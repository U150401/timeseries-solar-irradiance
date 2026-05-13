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

def main(args: argparse.Namespace) -> None:

    # ── 1. Load & engineer ───────────────────────────────────────────────────
    target_raw, _ = load_all(DATA_DIR, TARGET_ID)
    target_df     = engineer(target_raw)
    kt            = target_df["kt"].values.astype(np.float64)
    clearsky      = target_raw["Clearsky GHI"].values

    # ── Auto-detect sampling interval and derive seasonal period ─────────────
    idx = target_df.index
    interval_minutes = (idx[1] - idx[0]).total_seconds() / 60
    steps_per_day    = int(round(24 * 60 / interval_minutes))  # e.g. 96 (15-min) or 24 (hourly)
    seasonal_order   = (1, 1, 1, steps_per_day)
    print(f"[sarima] Interval: {interval_minutes:.0f} min  →  steps/day = {steps_per_day}  "
          f"(seasonal period s={steps_per_day})")

    # Default fit_len: one full year of training data
    one_year = steps_per_day * 365
    fit_len_default = one_year
    fit_len = min(args.fit_len if args.fit_len is not None else fit_len_default,
                  int(len(kt) * 0.70))

    T         = len(kt)
    train_end = int(T * 0.70)
    val_end   = int(T * 0.85)
    max_h     = max(HORIZONS)
    n_test    = T - val_end - max_h

    print(f"[sarima] T={T}  train={train_end}  val_end={val_end}  n_test={n_test}")

    # ── 2. Fit on the last fit_len steps of training data ───────────────────
    kt_fit = kt[train_end - fit_len : train_end]
    print(f"[sarima] Fitting SARIMA{ORDER}x{seasonal_order} on {fit_len} observations "
          f"({fit_len / steps_per_day:.1f} days) …")

    model = SARIMAX(
        kt_fit,
        order=ORDER,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    res_path_existing = ROOT / "results_sarima.npz"
    if args.no_refit and res_path_existing.exists():
        print("[sarima] --no_refit: loading saved parameters (skipping optimisation)")
        saved        = dict(np.load(res_path_existing, allow_pickle=True))
        saved_params = saved["params"].astype(float)
        result       = model.filter(saved_params)      # filter only, no MLE
        _aic         = float(saved["aic"])
        _bic         = float(saved["bic"])
        _param_names = saved["param_names"]
        _conf_lower  = saved["param_conf_lower"].astype(float)
        _conf_upper  = saved["param_conf_upper"].astype(float)
    else:
        result       = model.fit(disp=False, maxiter=200)
        print(result.summary())
        _aic         = result.aic
        _bic         = result.bic
        _param_names = np.array(result.param_names, dtype=object)
        _conf_lower  = np.asarray(result.conf_int())[:, 0]
        _conf_upper  = np.asarray(result.conf_int())[:, 1]

    print(f"[sarima] AIC={_aic:.2f}  BIC={_bic:.2f}")

    # ── 3. Windowed Kalman filter (warm-up + test period only) ──────────────
    # Running the filter over the full 3-year series is expensive (O(n·s²)).
    # The Kalman gain converges within a few seasonal periods, so we warm up
    # for WARM_UP steps before the test set and filter only that window.
    WARM_UP     = 4 * steps_per_day          # 4 days = enough for convergence
    win_start   = max(0, val_end - WARM_UP)
    actual_wu   = val_end - win_start        # actual warm-up length (may be < WARM_UP)
    win_end     = val_end + n_test + max_h
    kt_win      = kt[win_start : win_end]

    print(f"[sarima] Windowed Kalman filter: {len(kt_win)} steps "
          f"(warm-up {actual_wu} + test {n_test} + horizon {max_h}) …")
    win_model = SARIMAX(
        kt_win,
        order=ORDER,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result_win = win_model.filter(result.params)

    # ── 4. One-step predictions (1-step horizon) ─────────────────────────────
    # Kalman-filter one-step-ahead: pred[t] = E[y_t | y_{0..t-1}]
    one_step = np.asarray(result_win.get_prediction(dynamic=False).predicted_mean)
    y_pred   = np.zeros((n_test, len(HORIZONS)), dtype=np.float64)
    y_pred[:, 0] = one_step[actual_wu + 1 : actual_wu + 1 + n_test]

    # ── 5. Multi-step predictions (+6, +24 steps) ────────────────────────────
    # Re-anchor every h steps (rolling window of width h).
    # Maximum conditioning staleness per batch = h-1 steps (vs. the old
    # daily-batch approach where it was up to steps_per_day-1 = 95 steps).
    print("[sarima] Computing multi-step forecasts (rolling BATCH=h) …")

    for h_idx, h in enumerate(HORIZONS[1:], start=1):
        print(f"[sarima]   +{h} steps …", flush=True)
        h_preds = np.zeros(n_test)

        for batch_start in range(0, n_test, h):
            batch_end = min(batch_start + h, n_test)
            t_w      = actual_wu + batch_start   # window-relative anchor
            n_ahead  = (batch_end - batch_start) + h
            end_w    = min(t_w + n_ahead - 1, len(kt_win) - 1)

            # dynamic=True: recursive forecast from t_w; no future data used.
            # (Never use dynamic=<int>; it silently falls back to one-step.)
            dyn_raw = np.asarray(result_win.get_prediction(
                start=t_w, end=end_w, dynamic=True,
            ).predicted_mean)

            if len(dyn_raw) < n_ahead:
                dyn_raw = np.concatenate(
                    [dyn_raw, np.full(n_ahead - len(dyn_raw), dyn_raw[-1])]
                )

            for j in range(batch_end - batch_start):
                if j + h < len(dyn_raw):
                    h_preds[batch_start + j] = dyn_raw[j + h]

        y_pred[:, h_idx] = h_preds


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
    # Persistence: predict kt[t+h] = kt[t] for all h.
    # Use kt[val_end + i] (last known value at test point i) for all horizons.
    y_pers = np.tile(
        kt[val_end : val_end + n_test, np.newaxis], (1, len(HORIZONS))
    ).astype(np.float32)

    # ── 6. Evaluate ─────────────────────────────────────────────────────────
    horizon_labels = [f"{h}h" for h in horizons_h]
    results = evaluate_all(y_true, y_pred, y_pers, horizon_labels, is_day.astype(bool))
    print("\n── SARIMA Test Results ──────────────────────────────────────────")
    print_results(results)

    # ── 7. Save ─────────────────────────────────────────────────────────────
    res_path = ROOT / "results_sarima.npz"
    np.savez(
        res_path,
        y_true      = y_true,
        y_pred      = y_pred,
        y_pers      = y_pers,
        is_day      = is_day,
        # No epoch-based training curve — store AIC/BIC for the report
        train_loss  = np.array([_aic]),
        val_loss    = np.array([_bic]),
        horizons    = np.array(HORIZONS),
        aic         = np.array([_aic]),
        bic         = np.array([_bic]),
        params      = np.asarray(result.params),
        param_names = np.array(_param_names, dtype=object),
        param_conf_lower = _conf_lower,
        param_conf_upper = _conf_upper,
        order            = np.array(ORDER),
        seasonal_order   = np.array(seasonal_order),
        fit_lens         = np.array([fit_lens[h] for h in horizons_h]),
        aic_per_horizon  = np.array([fitted[h].aic for h in horizons_h]),
        bic_per_horizon  = np.array([fitted[h].bic for h in horizons_h]),
        granularity      = np.array(["hourly"]),
    )
    print(f"\n[saved] {res_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit_len", type=int, default=FIT_LEN,
                        help="Number of training steps used for fitting "
                             "(default: auto = 1 year of data)")
    parser.add_argument("--no_refit", action="store_true",
                        help="Skip MLE optimisation; load params from existing "
                             "results_sarima.npz (fast — use after first full run)")
    main(parser.parse_args())
