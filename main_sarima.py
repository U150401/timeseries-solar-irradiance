"""
SARIMA baseline for solar kt forecasting — one direct-forecast model per horizon.

Methodology
-----------
The native data is 15-minute resolution.  Fitting a SARIMA with a 96-step
seasonal period on a multi-year series is intractable (the Kalman filter state
grows quadratically with the seasonal period).  We therefore aggregate the
clearsky-index series to **hourly means** for SARIMA only — this gives the
classical s=24 daily seasonality that SARIMA was designed for.

For every forecast horizon (in hours) we fit a *separate* SARIMA(p,d,q)(P,D,Q)[s]
on the target station's hourly kt series and produce the h-step ahead direct
forecast at each test anchor by applying the fitted parameters to a recent
window and calling `forecast(steps=h)`.  Fitting one model per horizon avoids
recursive 1-step error accumulation for the 6h and 24h horizons.

The horizons are commensurate with the GRU/GNN setup (1h, 6h, 24h ahead in
wall-clock time).  Metrics are computed at the hourly resolution and labelled
identically so the three models can be compared.

Usage
-----
    python main_sarima.py
    python main_sarima.py --horizons 1 6 24 --fit_days 90
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.loader   import load_all
from src.features import engineer
from src.metrics  import evaluate_all, print_results
from src.utils    import persistence_baseline

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT / "dataset"
TARGET_ID = "41.93"

# Per-horizon SARIMA configuration.  The seasonal period is always s=24 (one
# day at hourly resolution).  The fit length scales with the horizon so that
# longer-horizon models see more seasonal cycles.
DEFAULT_ORDER          = (1, 0, 1)
DEFAULT_SEASONAL_ORDER = (1, 1, 1)        # P, D, Q  (s=24 added at runtime)
HORIZON_FIT_DAYS       = {1: 30, 6: 60, 24: 120}


def _build_sarimax(series, order, seasonal_order):
    return SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )


def fit_sarima_for_horizon(kt, train_end, fit_len, order, seasonal_order):
    """Fit one SARIMA on the last `fit_len` hourly observations before train_end."""
    series = kt[max(0, train_end - fit_len) : train_end]
    return _build_sarimax(series, order, seasonal_order).fit(disp=False, maxiter=200)


def predict_direct_h(
    fit_result,
    kt: np.ndarray,
    val_end: int,
    n_test: int,
    h: int,
    batch_steps: int,
    warmup: int,
) -> np.ndarray:
    """Direct h-step ahead forecast at every test point (hourly granularity).

    For each daily anchor we slice a recent window of observed kt, apply the
    fitted parameters to that window, then call `forecast(steps=batch + h)`.
    The element at offset `h-1+j` from the anchor is the direct h-step ahead
    forecast for test point i = b_start + j.
    """
    preds = np.full(n_test, np.nan, dtype=np.float64)
    warmup = max(warmup, 96)   # at least 4 daily cycles
    for b_start in range(0, n_test, batch_steps):
        b_end = min(b_start + batch_steps, n_test)
        anchor = val_end + b_start
        lo = max(0, anchor + 1 - warmup)
        window = kt[lo : anchor + 1]
        applied = fit_result.apply(endog=window, refit=False)
        n_steps_needed = (b_end - b_start) + h
        fc = np.asarray(applied.forecast(steps=n_steps_needed))
        for j in range(b_end - b_start):
            idx = j + h - 1
            if idx < len(fc):
                preds[b_start + j] = fc[idx]
    return preds


def main(args: argparse.Namespace) -> None:
    # ── 1. Load 15-min target series, then aggregate to hourly ──────────────
    target_raw, _ = load_all(args.data_dir, args.target_id)
    target_df     = engineer(target_raw)

    hourly = target_df[["kt", "Clearsky GHI"]].resample("h").mean()
    kt        = hourly["kt"].values.astype(np.float64)
    clearsky  = hourly["Clearsky GHI"].values

    s = 24
    seasonal_order = (*DEFAULT_SEASONAL_ORDER, s)

    horizons_h = args.horizons      # already in hours; at hourly resolution h_steps == h_hours
    print(f"[sarima] Hourly series, T={len(kt)}  s={s}  horizons(h)={horizons_h}")

    # ── 2. Splits (chronological) ──────────────────────────────────────────
    T          = len(kt)
    train_end  = int(T * 0.70)
    val_end    = int(T * 0.85)
    max_h      = max(horizons_h)
    n_test     = T - val_end - max_h
    print(f"[sarima] train_end={train_end}  val_end={val_end}  n_test={n_test}")

    # ── 3. Fit one SARIMA per horizon ──────────────────────────────────────
    fitted: dict[int, object] = {}
    fit_lens: dict[int, int] = {}

    for h in horizons_h:
        fit_days = args.fit_days if args.fit_days is not None else HORIZON_FIT_DAYS.get(h, 60)
        fit_len  = min(fit_days * 24, train_end)
        fit_lens[h] = fit_len
        print(
            f"\n[sarima] Fitting horizon={h}h  "
            f"order={DEFAULT_ORDER}  seasonal_order={seasonal_order}  "
            f"fit_days={fit_days}  fit_len={fit_len}"
        )
        result = fit_sarima_for_horizon(
            kt, train_end, fit_len, DEFAULT_ORDER, seasonal_order
        )
        fitted[h] = result
        print(f"[sarima] AIC={result.aic:.2f}  BIC={result.bic:.2f}  "
              f"params={dict(zip(result.param_names, np.round(result.params, 4)))}")

    # ── 4. Direct h-step predictions for each horizon ──────────────────────
    y_pred = np.zeros((n_test, len(horizons_h)), dtype=np.float64)
    for h_idx, h in enumerate(horizons_h):
        print(f"\n[sarima] Forecasting horizon={h}h …", flush=True)
        y_pred[:, h_idx] = predict_direct_h(
            fit_result = fitted[h],
            kt         = kt,
            val_end    = val_end,
            n_test     = n_test,
            h          = h,
            batch_steps= 24,                                  # re-anchor every day
            warmup     = args.warmup_days * 24,
        )
    y_pred = np.clip(y_pred, 0.0, 1.5).astype(np.float32)

    # ── 5. Targets, daytime mask, persistence ─────────────────────────────
    y_true = np.stack(
        [kt[val_end + h : val_end + h + n_test] for h in horizons_h], axis=-1,
    ).astype(np.float32)
    is_day = np.stack(
        [(clearsky[val_end + h : val_end + h + n_test] > 1.0).astype(np.float32)
         for h in horizons_h], axis=-1,
    )
    anchors = val_end + np.arange(n_test)
    y_pers  = persistence_baseline(kt, horizons_h)[anchors].astype(np.float32)

    # ── 6. Evaluate ─────────────────────────────────────────────────────────
    horizon_labels = [f"{h}h" for h in horizons_h]
    results = evaluate_all(y_true, y_pred, y_pers, horizon_labels, is_day.astype(bool))
    print("\n── SARIMA Test Results ──────────────────────────────────────────")
    print_results(results)

    # ── 7. Save ─────────────────────────────────────────────────────────────
    res_path = ROOT / "results_sarima.npz"
    np.savez(
        res_path,
        y_true           = y_true,
        y_pred           = y_pred,
        y_pers           = y_pers,
        is_day           = is_day,
        train_loss       = np.array([fitted[h].aic for h in horizons_h]),
        val_loss         = np.array([fitted[h].bic for h in horizons_h]),
        horizons         = np.array(horizons_h),
        order            = np.array(DEFAULT_ORDER),
        seasonal_order   = np.array(seasonal_order),
        fit_lens         = np.array([fit_lens[h] for h in horizons_h]),
        aic_per_horizon  = np.array([fitted[h].aic for h in horizons_h]),
        bic_per_horizon  = np.array([fitted[h].bic for h in horizons_h]),
        granularity      = np.array(["hourly"]),
    )
    print(f"\n[saved] {res_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, default=str(DATA_DIR))
    parser.add_argument("--target_id",  type=str, default=TARGET_ID)
    parser.add_argument("--horizons",   type=int, nargs="+", default=[1, 6, 24],
                        help="Forecast horizons in hours (e.g. --horizons 1 6 24)")
    parser.add_argument("--fit_days",   type=int, default=None,
                        help="If set, use the same fit window (in days) for every "
                             "horizon. Otherwise scale per HORIZON_FIT_DAYS.")
    parser.add_argument("--warmup_days",type=int, default=14,
                        help="Days of recent observations passed to apply() per anchor")
    main(parser.parse_args())
