"""Load and align NSRDB CSV files."""
import pandas as pd
from pathlib import Path


NEIGHBOR_COLS = ["kt", "Cloud Type", "Relative Humidity", "Wind Speed", "sin_wind", "cos_wind"]


def load_nsrdb(path: str | Path) -> pd.DataFrame:
    """
    Load one NSRDB CSV. Skips the 2 metadata rows, parses datetime from
    Year/Month/Day/Hour/Minute columns, returns a DatetimeIndex DataFrame.
    """
    df = pd.read_csv(path, skiprows=2)
    df["datetime"] = pd.to_datetime(
        dict(year=df["Year"], month=df["Month"], day=df["Day"],
             hour=df["Hour"], minute=df["Minute"])
    )
    df = df.set_index("datetime").sort_index()
    df = df.drop(columns=["Year", "Month", "Day", "Hour", "Minute"], errors="ignore")
    return df


def _station_key(path: Path) -> str:
    """Return the station identifier (everything before the last '_YEAR' suffix)."""
    # Filename format: {stationID}_{lat}_{lon}_{year}.csv
    parts = path.stem.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else path.stem


def _load_station(paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate (possibly multi-year) CSVs for one station."""
    frames = [load_nsrdb(p) for p in sorted(paths)]
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames).sort_index()


def load_all(data_dir: str | Path, target_id: str) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """
    Load all CSVs found recursively under data_dir. Files are grouped by
    station (the filename prefix before the year suffix) and concatenated
    across years. The station whose name contains target_id is the target;
    the rest become neighbors.

    Returns (target_df, [neighbor_df, ...])
    """
    data_dir = Path(data_dir)
    paths = sorted(data_dir.rglob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    # Group by station key
    stations: dict[str, list[Path]] = {}
    for p in paths:
        key = _station_key(p)
        stations.setdefault(key, []).append(p)

    target_key = None
    for key in stations:
        if target_id in key:
            target_key = key
            break

    if target_key is None:
        raise FileNotFoundError(f"No CSV with '{target_id}' in name found in {data_dir}")

    target_df = _load_station(stations[target_key])
    print(f"[loader] Target   : {target_key}  ({len(target_df)} rows, {len(stations[target_key])} file(s))")

    neighbor_dfs = []
    for key, station_paths in stations.items():
        if key == target_key:
            continue
        df = _load_station(station_paths)
        neighbor_dfs.append(df)
        print(f"[loader] Neighbor : {key}  ({len(df)} rows, {len(station_paths)} file(s))")

    return target_df, neighbor_dfs
