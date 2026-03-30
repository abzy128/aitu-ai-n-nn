"""LSTM model factory functions."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_1layer(
    window_size: int,
    n_features: int,
    units: int = 64,
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Single-layer LSTM baseline.

    Architecture: Input → LSTM(units) → Dropout → Dense(1, linear)
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(window_size, n_features)),
            layers.LSTM(units, return_sequences=False),
            layers.Dropout(dropout_rate),
            layers.Dense(1, activation="linear"),
        ],
        name="LSTM_1L",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_lstm_2layer(
    window_size: int,
    n_features: int,
    units_1: int = 100,
    units_2: int = 50,
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Two-layer stacked LSTM.

    Architecture:
        Input → LSTM(units_1, return_sequences=True)
              → Dropout → LSTM(units_2) → Dropout → Dense(1, linear)

    `return_sequences=True` on the first layer passes the full hidden-state
    sequence to the second LSTM instead of only the last step.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(window_size, n_features)),
            layers.LSTM(units_1, return_sequences=True),
            layers.Dropout(dropout_rate),
            layers.LSTM(units_2, return_sequences=False),
            layers.Dropout(dropout_rate),
            layers.Dense(1, activation="linear"),
        ],
        name="LSTM_2L",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model
