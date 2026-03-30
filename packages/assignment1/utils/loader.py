"""Data loading and initial inspection for the Jena Climate dataset."""

import pandas as pd
from pathlib import Path


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the Jena climate CSV (semicolon-separated) and return a DataFrame.

    The raw file uses `;` as separator and the Date Time column is in
    `dd.mm.yyyy HH:MM:SS` format. This function loads it without parsing
    timestamps — call parse_timestamps() from time_features.py next.
    """
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def inspect(df: pd.DataFrame) -> None:
    """Print shape, dtypes, describe, and missing-value summary."""
    print("=== Shape ===")
    print(df.shape)

    print("\n=== dtypes ===")
    print(df.dtypes)

    print("\n=== .info() ===")
    df.info()

    print("\n=== .describe() ===")
    print(df.describe())

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("\nNo missing values detected.")
    else:
        print("\n=== Missing values ===")
        print(missing)
