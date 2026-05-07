"""Feature engineering: clearsky index, cyclical encodings, normalisation."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Columns used as model input for the TARGET station
TARGET_FEAT_COLS = [
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

# Columns used for NEIGHBOR stations (smaller to avoid overfitting)
NEIGHBOR_FEAT_COLS = [
    "kt",
    "Cloud Type",
    "Relative Humidity",
    "Wind Speed",
    "sin_wind",
    "cos_wind",
]


def compute_clearsky_index(df: pd.DataFrame, eps: float = 1.0) -> pd.Series:
    """
    kt = GHI / Clearsky_GHI, clipped to [0, 1.5].
    At night (Clearsky GHI < eps) kt is set to 0.
    """
    cs = df["Clearsky GHI"].clip(lower=0)
    kt = df["GHI"] / (cs + eps)
    kt = kt.clip(0.0, 1.5)
    kt[cs < eps] = 0.0
    return kt.rename("kt")


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hour = df.index.hour + df.index.minute / 60.0
    doy = df.index.dayofyear
    wd_rad = np.deg2rad(df["Wind Direction"])
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
    df["sin_doy"]  = np.sin(2 * np.pi * doy / 365.0)
    df["cos_doy"]  = np.cos(2 * np.pi * doy / 365.0)
    df["sin_wind"] = np.sin(wd_rad)
    df["cos_wind"] = np.cos(wd_rad)
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline for one station DataFrame."""
    df = df.copy()
    df["kt"] = compute_clearsky_index(df)
    df = add_cyclical_features(df)
    df = df.fillna(0.0)
    return df


class MultiSiteScaler:
    """
    Fits a StandardScaler on the target station training split,
    then applies the same scaler to target and (separately) neighbors.
    kt is NOT scaled — it already lives in [0, 1.5].
    """

    def __init__(self):
        self.target_scaler = StandardScaler()
        self.neighbor_scalers: list[StandardScaler] = []

    def fit_transform_target(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        non_kt = [c for c in cols if c != "kt"]
        df = df.copy()
        df[non_kt] = self.target_scaler.fit_transform(df[non_kt])
        return df

    def transform_target(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        non_kt = [c for c in cols if c != "kt"]
        df = df.copy()
        df[non_kt] = self.target_scaler.transform(df[non_kt])
        return df

    def fit_transform_neighbor(self, df: pd.DataFrame, cols: list[str], idx: int) -> pd.DataFrame:
        while len(self.neighbor_scalers) <= idx:
            self.neighbor_scalers.append(StandardScaler())
        non_kt = [c for c in cols if c != "kt"]
        df = df.copy()
        df[non_kt] = self.neighbor_scalers[idx].fit_transform(df[non_kt])
        return df

    def transform_neighbor(self, df: pd.DataFrame, cols: list[str], idx: int) -> pd.DataFrame:
        non_kt = [c for c in cols if c != "kt"]
        df = df.copy()
        df[non_kt] = self.neighbor_scalers[idx].transform(df[non_kt])
        return df
