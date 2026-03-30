"""Feature engineering: lag features, rolling statistics, time-of-day category."""

import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    col: str = "T (degC)",
    lags: list[int] = [1, 2, 3],
) -> pd.DataFrame:
    """Add lag features for *col* at the specified lag offsets (in rows).

    For hourly data lag_1 = 1 h ago, lag_2 = 2 h ago, etc.
    """
    df = df.copy()
    for lag in lags:
        df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    col: str = "T (degC)",
    windows: list[int] = [3, 6, 12],
) -> pd.DataFrame:
    """Add rolling mean features for *col* over *windows* (in rows).

    For 10-minute raw data: 3→30 min, 6→60 min, 12→120 min.
    For hourly-resampled data: 3→3 h, 6→6 h, 12→12 h.
    """
    df = df.copy()
    for w in windows:
        df[f"{col}_rolling_{w}"] = df[col].rolling(window=w, min_periods=1).mean()
    return df


def add_time_of_day(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'time_of_day' string category and an ordinal integer encoding.

    Categories
    ----------
    night     0–5   → ordinal 0
    morning   6–11  → ordinal 1
    afternoon 12–17 → ordinal 2
    evening   18–23 → ordinal 3
    """
    if "hour" not in df.columns:
        raise ValueError("Run extract_time_features() first to add the 'hour' column.")

    df = df.copy()

    def _categorise(hour: int) -> str:
        if hour < 6:
            return "night"
        elif hour < 12:
            return "morning"
        elif hour < 18:
            return "afternoon"
        return "evening"

    _ordinal = {"night": 0, "morning": 1, "afternoon": 2, "evening": 3}

    df["time_of_day"] = df["hour"].map(_categorise)
    df["time_of_day_ord"] = df["time_of_day"].map(_ordinal)
    return df


def drop_lag_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that contain NaN introduced by lag / rolling features."""
    before = len(df)
    df = df.dropna()
    print(f"Dropped {before - len(df):,} rows with NaN (lag/rolling edge effects). "
          f"Remaining: {len(df):,}")
    return df
