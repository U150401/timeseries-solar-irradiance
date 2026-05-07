"""PyTorch Dataset for multi-site solar irradiance forecasting."""
import numpy as np
import torch
from torch.utils.data import Dataset


class SolarDataset(Dataset):
    """
    Sliding-window dataset that returns:
      x_target   : (lookback, n_target_features)  — target station history
      x_neighbors: (lookback, n_neighbor_features) — all neighbors concatenated
      y          : (n_horizons,)                   — kt at each forecast horizon
      is_day     : (n_horizons,)                   — 1 if Clearsky GHI > 0 at that step

    Parameters
    ----------
    target_arr      : np.ndarray (T, n_target_features)
    neighbor_arrs   : list of np.ndarray (T, n_neighbor_features_i)
    kt_series       : np.ndarray (T,)  — raw kt of target (unscaled)
    clearsky_series : np.ndarray (T,)  — Clearsky GHI of target
    lookback        : int   — number of past steps fed to the model
    horizons        : list[int]  — steps ahead to predict (e.g. [1, 6, 24])
    """

    def __init__(
        self,
        target_arr: np.ndarray,
        neighbor_arrs: list[np.ndarray],
        kt_series: np.ndarray,
        clearsky_series: np.ndarray,
        lookback: int = 24,
        horizons: list[int] = [1, 6, 24],
    ):
        self.target_arr = target_arr.astype(np.float32)
        self.neighbor_arr = (
            np.concatenate(neighbor_arrs, axis=-1).astype(np.float32)
            if neighbor_arrs else np.zeros((len(target_arr), 0), dtype=np.float32)
        )
        self.kt = kt_series.astype(np.float32)
        self.clearsky = clearsky_series.astype(np.float32)
        self.lookback = lookback
        self.horizons = horizons
        self.max_horizon = max(horizons)

        # valid start indices: need lookback past + max_horizon future
        self.indices = np.arange(lookback, len(target_arr) - self.max_horizon)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        t = self.indices[i]
        x_target    = self.target_arr[t - self.lookback : t]          # (L, F_t)
        x_neighbors = self.neighbor_arr[t - self.lookback : t]        # (L, F_n)
        y           = np.array([self.kt[t + h] for h in self.horizons])
        is_day      = np.array([float(self.clearsky[t + h] > 1.0) for h in self.horizons])

        return (
            torch.from_numpy(x_target),
            torch.from_numpy(x_neighbors),
            torch.from_numpy(y),
            torch.from_numpy(is_day),
        )


def time_split(
    n: int,
    train_frac: float = 0.70,
    val_frac: float   = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return index arrays for train / val / test splits (chronological)."""
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    return np.arange(t1), np.arange(t1, t2), np.arange(t2, n)
