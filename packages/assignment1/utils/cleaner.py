"""Missing-value handling and outlier detection/treatment for Jena data."""

import pandas as pd
import numpy as np


# Sentinel value used in the raw dataset for invalid wind measurements
_SENTINEL = -9999.0

# Known columns that contain sentinel values
_SENTINEL_COLS = ["wv (m/s)", "max. wv (m/s)", "wd (deg)"]

# Physically plausible ranges for key columns (inclusive)
_PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "T (degC)": (-30.0, 45.0),
    "Tpot (K)": (250.0, 330.0),
    "Tdew (degC)": (-40.0, 30.0),
    "rh (%)": (0.0, 100.0),
    "VPmax (mbar)": (0.0, 100.0),
    "VPact (mbar)": (0.0, 50.0),
    "VPdef (mbar)": (0.0, 80.0),
    "sh (g/kg)": (0.0, 30.0),
    "H2OC (mmol/mol)": (0.0, 50.0),
    "rho (g/m**3)": (1000.0, 1500.0),
    "wv (m/s)": (0.0, 60.0),
    "max. wv (m/s)": (0.0, 80.0),
    "wd (deg)": (0.0, 360.0),
}


def replace_sentinels(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    sentinel: float = _SENTINEL,
) -> pd.DataFrame:
    """Replace sentinel values (default -9999) with NaN.

    If cols is None, applies to all known sentinel columns that exist in df.
    """
    df = df.copy()
    target_cols = cols if cols is not None else [c for c in _SENTINEL_COLS if c in df.columns]
    replaced_total = 0
    for col in target_cols:
        mask = df[col] == sentinel
        count = mask.sum()
        if count:
            df.loc[mask, col] = np.nan
            replaced_total += count
            print(f"  {col}: replaced {count:,} sentinel values with NaN")
    print(f"Total sentinel replacements: {replaced_total:,}")
    return df


def interpolate_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaNs using time-based linear interpolation, then forward/back fill edges."""
    df = df.copy()
    before = df.isnull().sum().sum()
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="time")
    # Fill any remaining NaNs at the edges
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    after = df.isnull().sum().sum()
    print(f"Interpolated {before - after:,} missing values ({after} remaining)")
    return df


def handle_outliers(
    df: pd.DataFrame,
    method: str = "physical",
    iqr_multiplier: float = 3.0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Detect and cap outliers.

    Parameters
    ----------
    method : "physical" (cap at known physical limits) or "iqr" (IQR-based)
    iqr_multiplier : multiplier for IQR fence (used only when method="iqr")

    Returns (cleaned_df, outlier_counts_per_column)
    """
    df = df.copy()
    outlier_counts: dict[str, int] = {}

    if method == "physical":
        for col, (lo, hi) in _PHYSICAL_BOUNDS.items():
            if col not in df.columns:
                continue
            mask = (df[col] < lo) | (df[col] > hi)
            count = int(mask.sum())
            if count:
                df[col] = df[col].clip(lower=lo, upper=hi)
                outlier_counts[col] = count
                print(f"  {col}: capped {count:,} values to [{lo}, {hi}]")

    elif method == "iqr":
        numeric_cols = df.select_dtypes(include="number").columns
        for col in numeric_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
            mask = (df[col] < lo) | (df[col] > hi)
            count = int(mask.sum())
            if count:
                df[col] = df[col].clip(lower=lo, upper=hi)
                outlier_counts[col] = count
                print(f"  {col}: IQR capped {count:,} values")
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'physical' or 'iqr'.")

    total = sum(outlier_counts.values())
    print(f"Total outlier values capped: {total:,}")
    return df, outlier_counts
