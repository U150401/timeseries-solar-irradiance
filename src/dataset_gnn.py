"""
PyTorch Dataset for the GNN model.

Returns tensors shaped for a graph: all N stations share the same
feature columns, so the input has shape (lookback, N, n_features).
The target node is always at index 0.

Feature columns are the same for every station — unlike the GRU model
which used a reduced set for neighbours. The GCN will learn how much
to weight each station's features through its graph convolution weights.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

# Features used for ALL stations in the graph (same columns everywhere)
GNN_FEAT_COLS = [
    "kt",
    "Cloud Type",
    "Temperature",
    "Relative Humidity",
    "Pressure",
    "Wind Speed",
    "sin_wind",
    "cos_wind",
    "Dew Point",
    "Solar Zenith Angle",
    "sin_hour",
    "cos_hour",
    "sin_doy",
    "cos_doy",
]


class GraphSolarDataset(Dataset):
    """
    Sliding-window dataset for the GNN.

    Each sample contains:
      x       : (lookback, N, n_features)  — all stations
      y       : (n_horizons,)              — kt at target (node 0) per horizon
      is_day  : (n_horizons,)              — 1.0 if daytime at target node

    The adjacency matrix is fixed and stored separately (not per-sample).

    Parameters
    ----------
    station_arrs    : list of np.ndarray (T, F), target first then neighbours
    kt_target       : np.ndarray (T,) — raw clearsky index of target station
    clearsky_target : np.ndarray (T,) — Clearsky GHI of target station
    lookback        : int
    horizons        : list[int]
    """

    def __init__(
        self,
        station_arrs: list[np.ndarray],
        kt_target: np.ndarray,
        clearsky_target: np.ndarray,
        lookback: int = 24,
        horizons: list[int] = [1, 6, 24],
    ):
        # Stack stations: (T, N, F)
        self.graph_arr = np.stack(
            [a.astype(np.float32) for a in station_arrs], axis=1
        )
        self.kt        = kt_target.astype(np.float32)
        self.clearsky  = clearsky_target.astype(np.float32)
        self.lookback  = lookback
        self.horizons  = horizons
        self.max_h     = max(horizons)

        T = len(self.graph_arr)
        self.indices = np.arange(lookback, T - self.max_h)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        t = self.indices[i]
        x      = self.graph_arr[t - self.lookback : t]                     # (L, N, F)
        y      = np.array([self.kt[t + h]          for h in self.horizons])
        is_day = np.array([float(self.clearsky[t + h] > 1.0) for h in self.horizons])

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(is_day),
        )
