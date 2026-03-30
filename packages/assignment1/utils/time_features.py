"""Timestamp parsing and time-feature extraction for the Jena dataset."""

import pandas as pd


def parse_timestamps(df: pd.DataFrame, col: str = "Date Time") -> pd.DataFrame:
    """Parse the Date Time column (dd.mm.yyyy HH:MM:SS) and set it as index.

    Returns a new DataFrame with a DatetimeIndex sorted chronologically.
    """
    df = df.copy()
    df[col] = pd.to_datetime(df[col], format="%d.%m.%Y %H:%M:%S")
    df = df.set_index(col).sort_index()

    # Sanity checks
    duplicates = df.index.duplicated().sum()
    if duplicates:
        print(f"WARNING: {duplicates} duplicate timestamps found — keeping first.")
        df = df[~df.index.duplicated(keep="first")]

    diffs = pd.Series(df.index).diff().dropna().unique()
    print(f"Unique time deltas: {sorted(diffs)}")

    return df


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day_of_week, month, and is_weekend columns from the index."""
    df = df.copy()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek   # 0=Monday … 6=Sunday
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    return df


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 10-minute data to hourly by taking the mean of each hour."""
    return df.resample("h").mean()
