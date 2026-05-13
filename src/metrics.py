"""Forecast evaluation metrics."""
import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalised RMSE (by mean of observed)."""
    denom = np.mean(np.abs(y_true)) + 1e-8
    return rmse(y_true, y_pred) / denom


def skill_score(y_true: np.ndarray, y_pred: np.ndarray, y_ref: np.ndarray) -> float:
    """
    Forecast Skill Score relative to a reference (e.g. persistence).
    SS = 1 - RMSE_model / RMSE_ref
    """
    rmse_model = rmse(y_true, y_pred)
    rmse_ref   = rmse(y_true, y_ref)
    if rmse_ref < 1e-8:
        return 0.0
    return float(1.0 - rmse_model / rmse_ref)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-8))


def evaluate_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_persistence: np.ndarray,
    horizon_labels: list[str],
    daytime_mask: np.ndarray | None = None,
) -> dict:
    """
    Compute all metrics per horizon. If daytime_mask provided, also compute
    daytime-only metrics (where forecasting is meaningful).

    y_true, y_pred, y_persistence : (N, n_horizons)
    daytime_mask                  : (N, n_horizons) bool
    """
    results = {}
    n_horizons = y_true.shape[1]

    for i, label in enumerate(horizon_labels):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        yr = y_persistence[:, i]

        entry = dict(
            MAE   = mae(yt, yp),
            RMSE  = rmse(yt, yp),
            nRMSE = nrmse(yt, yp),
            Skill = skill_score(yt, yp, yr),
            R2    = r2(yt, yp),
        )

        if daytime_mask is not None:
            mask = daytime_mask[:, i].astype(bool)
            if mask.sum() > 0:
                entry["MAE_day"]   = mae(yt[mask], yp[mask])
                entry["RMSE_day"]  = rmse(yt[mask], yp[mask])
                entry["Skill_day"] = skill_score(yt[mask], yp[mask], yr[mask])

        results[label] = entry

    return results


def persistence_baseline(kt: np.ndarray, horizons: list[int]) -> np.ndarray:
    """
    Lag-h persistence baseline.

    persistence[t, h_idx] = kt[t - horizons[h_idx]]

    Index at raw time t to get the baseline prediction for horizon h:
    predict kt[t + h] = kt[t] by querying persistence[t + h, h_idx].
    """
    N, out = len(kt), []
    for h in horizons:
        col = np.zeros(N)
        col[h:] = kt[:-h]
        out.append(col)
    return np.stack(out, axis=-1)


def print_results(results: dict) -> None:
    header = f"{'Horizon':<10}" + "".join(f"{k:>12}" for k in ["MAE", "RMSE", "nRMSE", "Skill", "R2"])
    print(header)
    print("-" * len(header))
    for horizon, metrics in results.items():
        row = f"{horizon:<10}" + "".join(
            f"{metrics.get(k, float('nan')):>12.4f}"
            for k in ["MAE", "RMSE", "nRMSE", "Skill", "R2"]
        )
        print(row)
