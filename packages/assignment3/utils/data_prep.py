"""Data loading, scaling (with target), and chronological splitting.

Key difference from Stage 2: the target T (degC) is also scaled here
so LSTM training sees values in the same range as the features.
Predictions are inverse-transformed before computing RMSE/MAE in °C.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TARGET = "T (degC)"
_DROP_COLS = ["time_of_day"]   # string column; ordinal encoding kept


def load_and_split(
    csv_path: str | Path,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed_data.csv and return (train, val, test) DataFrames.

    Split is strictly chronological by index position — identical
    boundaries to Stage 2 so comparisons are fair.
    """
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, int]:
    """Reload the Stage-2 scaler and transform all splits.

    The Stage-2 scaler was fit on features only (target excluded).
    Here we reload it and extend its application to include the target
    column by fitting a *separate* target scaler on train — this avoids
    refitting the feature scaler (no leakage) while still normalising T.

    Returns
    -------
    X_train, y_train_scaled, X_val, y_val_scaled,
    X_test, y_test_scaled, target_scaler, target_col_idx
    """
    # Reload Stage-2 feature scaler (transform only, never refit)
    with open(scaler_path, "rb") as f:
        feature_scaler: StandardScaler = pickle.load(f)

    feature_cols = [c for c in train.columns if c not in _DROP_COLS + [TARGET]]

    X_tr = feature_scaler.transform(train[feature_cols].values)
    X_va = feature_scaler.transform(val[feature_cols].values)
    X_te = feature_scaler.transform(test[feature_cols].values)

    # Separate scaler for the target (fit on train only)
    target_scaler = StandardScaler()
    y_tr = target_scaler.fit_transform(train[[TARGET]].values).flatten()
    y_va = target_scaler.transform(val[[TARGET]].values).flatten()
    y_te = target_scaler.transform(test[[TARGET]].values).flatten()

    print(f"Feature matrix — train: {X_tr.shape}, val: {X_va.shape}, test: {X_te.shape}")
    print(f"Features used ({len(feature_cols)}): {feature_cols}")

    # Index of TARGET in the feature matrix (needed for sequencer)
    target_col_idx = feature_cols.index(TARGET) if TARGET in feature_cols else -1

    return X_tr, y_tr, X_va, y_va, X_te, y_te, target_scaler, target_col_idx
