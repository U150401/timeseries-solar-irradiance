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


def persistence_baseline(kt: np.ndarray, horizons: list[int]) -> np.ndarray:
    N, out = len(kt), []
    for h in horizons:
        col = np.zeros(N)
        col[h:] = kt[:-h]
        out.append(col)
    return np.stack(out, axis=-1)


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

    model  = SARIMAX(
        kt_fit,
        order=ORDER,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False, maxiter=200)
    print(result.summary())
    print(f"[sarima] AIC={result.aic:.2f}  BIC={result.bic:.2f}")

    # ── 3. Apply fitted params to the full series via Kalman filter ──────────
    print("[sarima] Running Kalman filter over full series …")
    kt_full    = kt[: val_end + n_test + max_h]
    full_model = SARIMAX(
        kt_full,
        order=ORDER,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result_full = full_model.filter(result.params)

    # ── 4. One-step predictions (1h horizon) ─────────────────────────────────
    # get_prediction(dynamic=False) returns the Kalman-filter one-step-ahead
    # predictions: pred[t] = E[y_t | y_0, …, y_{t-1}].
    # For 1h at test point i (prediction point t = val_end+i):
    #   y_pred[i,0] = pred[val_end + i + 1]
    one_step = np.asarray(result_full.get_prediction(dynamic=False).predicted_mean)
    y_pred   = np.zeros((n_test, len(HORIZONS)), dtype=np.float64)
    y_pred[:, 0] = one_step[val_end + 1 : val_end + 1 + n_test]

    # ── 5. Multi-step predictions (6h and 24h horizons) ──────────────────────
    # Strategy: anchor to actual observations every 24 h (daily batches).
    # Within each 24-h batch the model forecasts recursively; at the next
    # day boundary it is re-anchored to the true kt.  This keeps the
    # recursive chain short and prevents error accumulation.
    BATCH = steps_per_day   # re-anchor every full day
    print(f"[sarima] Computing multi-step forecasts (batch={BATCH} steps = 1 day) …")

    for h_idx, h in enumerate(HORIZONS[1:], start=1):
        print(f"[sarima]   horizon {h}h …", flush=True)
        h_preds = np.zeros(n_test)

        for batch_start in range(0, n_test, BATCH):
            batch_end = min(batch_start + BATCH, n_test)
            t = val_end + batch_start          # anchor point (actual observation)

            # Dynamic (recursive) forecast starting from t, conditioned on
            # actual data up to t-1.  dynamic=True means the very first
            # predicted step (position t) is already recursive — no future
            # test observations leak into the conditioning set.
            #
            # NOTE: dynamic=<absolute_index> is a statsmodels bug trap: it is
            # compared against the prediction-window length (not nobs), so a
            # large absolute index is always "after the end" and silently
            # falls back to one-step predictions. Always use dynamic=True.
            n_ahead  = (batch_end - batch_start) + h
            end_idx  = min(t + n_ahead - 1, len(kt_full) - 1)
            dyn_raw  = np.asarray(result_full.get_prediction(
                start=t,
                end=end_idx,
                dynamic=True,   # start recursive from the first predicted step
            ).predicted_mean)
            # Pad with last value if series was clipped near end
            if len(dyn_raw) < n_ahead:
                pad = np.full(n_ahead - len(dyn_raw), dyn_raw[-1])
                dyn_raw = np.concatenate([dyn_raw, pad])
            dyn = dyn_raw

            for j in range(batch_end - batch_start):
                if j + h < len(dyn):
                    h_preds[batch_start + j] = dyn[j + h]

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
    # Match the same persistence indexing as main.py (offset by lookback=24)
    persistence = persistence_baseline(kt, HORIZONS)
    y_pers = persistence[val_end + 24 : val_end + 24 + n_test].astype(np.float32)

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
        train_loss  = np.array([result.aic]),
        val_loss    = np.array([result.bic]),
        horizons    = np.array(HORIZONS),
        aic         = np.array([result.aic]),
        bic         = np.array([result.bic]),
        params      = np.asarray(result.params),
        param_names = np.array(result.param_names, dtype=object),
        param_conf_lower = np.asarray(result.conf_int())[:, 0],
        param_conf_upper = np.asarray(result.conf_int())[:, 1],
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
    main(parser.parse_args())
