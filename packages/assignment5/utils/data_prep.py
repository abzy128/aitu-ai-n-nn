"""Data loading, scaling, and chronological splitting — identical to Stages 3 & 4."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TARGET = "T (degC)"
_DROP_COLS = ["time_of_day"]


def load_and_split(
    csv_path: str | Path,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.sort_index().dropna()

    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]

    print(f"Total rows : {n:,}")
    print(f"Train : {len(train):,}  ({train.index[0]} → {train.index[-1]})")
    print(f"Val   : {len(val):,}  ({val.index[0]} → {val.index[-1]})")
    print(f"Test  : {len(test):,}  ({test.index[0]} → {test.index[-1]})")

    return train, val, test


def scale_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    scaler_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, list[str]]:
    """Reload Stage-2 scaler and transform all splits.

    Returns X_train, X_val, X_test, feature_scaler, feature_cols.
    The autoencoder operates on X only (no target needed for training).
    """
    with open(scaler_path, "rb") as f:
        feature_scaler: StandardScaler = pickle.load(f)

    feature_cols = [c for c in train.columns if c not in _DROP_COLS + [TARGET]]

    X_tr = feature_scaler.transform(train[feature_cols].values).astype(np.float32)
    X_va = feature_scaler.transform(val[feature_cols].values).astype(np.float32)
    X_te = feature_scaler.transform(test[feature_cols].values).astype(np.float32)

    print(f"Feature matrix — train: {X_tr.shape}, val: {X_va.shape}, test: {X_te.shape}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    return X_tr, X_va, X_te, feature_scaler, feature_cols


def get_window_timestamps(df: pd.DataFrame, window_size: int) -> pd.DatetimeIndex:
    """Return the timestamp of the last row in each sliding window.

    For window i covering rows [i, i+window_size), the last row is i+window_size-1.
    n_seq = len(df) - window_size sequences are produced.
    """
    n_seq = len(df) - window_size
    return df.index[window_size - 1 : window_size - 1 + n_seq]
