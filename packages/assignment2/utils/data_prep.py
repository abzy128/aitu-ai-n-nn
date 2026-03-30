"""Loading, scaling, and chronological splitting for the Jena dataset."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TARGET = "T (degC)"

# Columns to drop: target, categorical string, and ordinal encoding of the same
_DROP_COLS = [TARGET, "time_of_day"]


def load_and_split(
    csv_path: str | Path,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed_data.csv and return (train, val, test) DataFrames.

    Split is strictly chronological — no shuffling.
    NaNs from lag/rolling features are dropped *before* splitting.
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.sort_index()

    # Drop rows with NaN (lag/rolling edge effects)
    before = len(df)
    df = df.dropna()
    if before - len(df):
        print(f"Dropped {before - len(df)} NaN rows before splitting.")

    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]

    print(f"Total rows: {n:,}")
    print(f"Train : {len(train):,}  ({train.index[0]} → {train.index[-1]})")
    print(f"Val   : {len(val):,}  ({val.index[0]} → {val.index[-1]})")
    print(f"Test  : {len(test):,}  ({test.index[0]} → {test.index[-1]})")

    return train, val, test


def scale_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    scaler_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on train features; transform all splits.

    Target (`T (degC)`) is kept unscaled for interpretable RMSE in °C.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test, fitted_scaler
    """
    feature_cols = [c for c in train.columns if c not in _DROP_COLS]

    X_tr = train[feature_cols].values
    X_va = val[feature_cols].values
    X_te = test[feature_cols].values

    y_tr = train[TARGET].values
    y_va = val[TARGET].values
    y_te = test[TARGET].values

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)

    if scaler_path is not None:
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {scaler_path}")

    print(f"Feature matrix shape — train: {X_tr.shape}, val: {X_va.shape}, test: {X_te.shape}")
    print(f"Features used ({len(feature_cols)}): {feature_cols}")
    return X_tr, y_tr, X_va, y_va, X_te, y_te, scaler
