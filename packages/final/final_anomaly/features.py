from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


LOW_POWER_THRESHOLD = 10.0
JUMP_THRESHOLD = 5.0
SEQUENCE_LENGTH = 60


@dataclass(frozen=True)
class LabelConfig:
    low_power_threshold: float = LOW_POWER_THRESHOLD
    jump_threshold: float = JUMP_THRESHOLD


def load_furnace1(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "DateTime" not in df.columns or "active_power" not in df.columns:
        raise ValueError("Furnace1.csv must contain DateTime and active_power columns")
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="raise")
    return df.sort_values("DateTime").reset_index(drop=True)


def add_proxy_labels(df: pd.DataFrame, config: LabelConfig = LabelConfig()) -> pd.DataFrame:
    out = df.copy()
    diff = out["active_power"].diff().abs().fillna(0.0)
    out["proxy_anomaly"] = (
        (out["active_power"] < config.low_power_threshold) | (diff > config.jump_threshold)
    ).astype(int)
    return out


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["DateTime", "active_power", "proxy_anomaly"]].copy()
    power = out["active_power"]

    for lag in (1, 5, 15, 60):
        out[f"active_power_lag_{lag}"] = power.shift(lag)

    out["active_power_diff_1"] = power.diff()
    out["active_power_diff_5"] = power.diff(5)

    for window in (5, 15, 60):
        shifted = power.shift(1)
        rolling = shifted.rolling(window=window, min_periods=window)
        out[f"active_power_roll{window}_mean"] = rolling.mean()
        out[f"active_power_roll{window}_std"] = rolling.std()
        out[f"active_power_roll{window}_min"] = rolling.min()
        out[f"active_power_roll{window}_max"] = rolling.max()

    minute_of_day = out["DateTime"].dt.hour * 60 + out["DateTime"].dt.minute
    out["minute_sin"] = np.sin(2.0 * np.pi * minute_of_day / 1440.0)
    out["minute_cos"] = np.cos(2.0 * np.pi * minute_of_day / 1440.0)
    out["hour_sin"] = np.sin(2.0 * np.pi * out["DateTime"].dt.hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * out["DateTime"].dt.hour / 24.0)

    return out.dropna().reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0.0 < train_frac < 1.0 or not 0.0 < val_frac < 1.0:
        raise ValueError("train_frac and val_frac must be between 0 and 1")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must leave a test split")

    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return (
        df.iloc[:train_end].reset_index(drop=True),
        df.iloc[train_end:val_end].reset_index(drop=True),
        df.iloc[val_end:].reset_index(drop=True),
    )


def detector_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"DateTime", "proxy_anomaly"}
    return [col for col in df.columns if col not in excluded]


def forecast_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in detector_feature_columns(df) if col != "active_power"]


def make_sequences(values: np.ndarray, labels: np.ndarray, length: int = SEQUENCE_LENGTH) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim != 1:
        raise ValueError("values must be a one-dimensional array")
    if len(values) != len(labels):
        raise ValueError("values and labels must have the same length")
    if len(values) < length:
        return np.empty((0, length, 1), dtype=np.float32), np.empty((0,), dtype=np.int64)

    windows = []
    endpoint_labels = []
    for end in range(length, len(values) + 1):
        start = end - length
        windows.append(values[start:end])
        endpoint_labels.append(labels[end - 1])
    x = np.asarray(windows, dtype=np.float32)[:, :, None]
    y = np.asarray(endpoint_labels, dtype=np.int64)
    return x, y
