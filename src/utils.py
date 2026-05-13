"""Shared helpers across SARIMA, GRU, and GNN entry points."""
from __future__ import annotations

import numpy as np
import pandas as pd


def detect_steps_per_hour(df: pd.DataFrame) -> int:
    """Infer the sampling rate (steps per hour) from a DatetimeIndex DataFrame."""
    interval_min = (df.index[1] - df.index[0]).total_seconds() / 60.0
    if interval_min <= 0:
        raise ValueError(f"Non-positive index interval: {interval_min} min")
    return int(round(60.0 / interval_min))


def hours_to_steps(horizons_h: list[int], steps_per_hour: int) -> list[int]:
    """Convert a list of horizons in hours to step counts at the data rate."""
    return [h * steps_per_hour for h in horizons_h]


def persistence_baseline(kt: np.ndarray, horizons_steps: list[int]) -> np.ndarray:
    """Naive persistence forecast indexed by **anchor** time.

    For prediction anchor index t and horizon h, the persistence forecast for
    kt[t+h] is kt[t]. The returned array has shape (T, H) and is indexed
    directly by the anchor index:

        y_pers_at_anchor_t = persistence_baseline(kt, horizons_steps)[t]   # (H,)

    All H columns are equal to kt[t] (true naive persistence).
    """
    n = len(kt)
    H = len(horizons_steps)
    base = np.asarray(kt, dtype=np.float32)
    return np.tile(base[:, None], (1, H))
