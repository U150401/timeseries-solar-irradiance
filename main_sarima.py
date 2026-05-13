"""
SARIMA baseline for solar kt forecasting.

The seasonal period s is auto-detected from the data time index so that it
always equals one full day regardless of the recording interval:
  • 60-min data  → s = 24
  • 15-min data  → s = 96   (default for NSRDB)

Fits SARIMA(1,0,1)(1,1,1)[s] on the training split and evaluates on
the same test split used by the neural network models, making metrics
directly comparable.

Forecast methodology
--------------------
• +1 step  : true one-step-ahead Kalman-filter prediction at every test
             point (uses actual observations up to t, predicts t+1).
• +6/+24 steps : dynamic (recursive) forecast anchored every s steps
             (one full day). Within each day window the model forecasts
             recursively; at the next day boundary it is re-anchored to
             real data, keeping the recursive chain short.

Usage
-----
    python main_sarima.py
    python main_sarima.py --fit_len 35040  # fit on last year only (15-min)
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.loader   import load_all
from src.features import engineer
from src.metrics  import evaluate_all, print_results

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT / "data"
TARGET_ID = "41.93"
HORIZONS  = [1, 6, 24]
ORDER     = (1, 0, 1)
# SEASONAL_ORDER is built dynamically in main() once the interval is known.
# FIT_LEN default: 1 year of data (auto-set after interval detection)
FIT_LEN   = None   # None → auto (1 year of training data)


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

    y_pred = np.clip(y_pred, 0.0, 1.5).astype(np.float32)

    # ── 6. Targets, is_day, persistence ─────────────────────────────────────
    y_true = np.array(
        [[kt[val_end + i + h] for h in HORIZONS] for i in range(n_test)],
        dtype=np.float32,
    )
    is_day = np.array(
        [[float(clearsky[val_end + i + h] > 1.0) for h in HORIZONS]
         for i in range(n_test)],
        dtype=np.float32,
    )
    # Persistence: predict kt[t+h] = kt[t] for all h.
    # Use kt[val_end + i] (last known value at test point i) for all horizons.
    y_pers = np.tile(
        kt[val_end : val_end + n_test, np.newaxis], (1, len(HORIZONS))
    ).astype(np.float32)

    # ── 7. Evaluate ──────────────────────────────────────────────────────────
    horizon_labels = [f"{h}h" for h in HORIZONS]
    results = evaluate_all(y_true, y_pred, y_pers, horizon_labels, is_day.astype(bool))
    print("\n── SARIMA Test Results ──────────────────────────────────────────")
    print_results(results)

    # ── 8. Save ──────────────────────────────────────────────────────────────
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
        fit_len          = np.array([fit_len]),
        steps_per_day    = np.array([steps_per_day]),
        interval_minutes = np.array([interval_minutes]),
    )
    print(f"[saved] {res_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit_len", type=int, default=FIT_LEN,
                        help="Number of training steps used for fitting "
                             "(default: auto = 1 year of data)")
    parser.add_argument("--no_refit", action="store_true",
                        help="Skip MLE optimisation; load params from existing "
                             "results_sarima.npz (fast — use after first full run)")
    main(parser.parse_args())
