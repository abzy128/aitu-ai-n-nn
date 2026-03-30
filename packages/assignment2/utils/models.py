"""MLP and DNN model factory functions."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_mlp(input_dim: int, learning_rate: float = 1e-3) -> keras.Model:
    """3-hidden-layer MLP: Input → 64 → 32 → 16 → 1 (linear)."""
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="linear"),
        ],
        name="MLP",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_dnn(input_dim: int, dropout_rate: float = 0.2, learning_rate: float = 1e-3) -> keras.Model:
    """6-hidden-layer DNN with Dropout: Input → 128 → 64 → 32 → 16 → 8 → 1 (linear)."""
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(64, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(32, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(16, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(8, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(1, activation="linear"),
        ],
        name="DNN",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model
