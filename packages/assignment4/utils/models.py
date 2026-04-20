"""1D CNN model factory functions for Stage 4."""

from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_simple(
    window_size: int,
    n_features: int,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Simple 1D CNN baseline.

    Architecture:
        Input → Conv1D(64, k=3, causal) → MaxPooling1D(2)
              → Conv1D(32, k=3, causal) → GlobalAveragePooling1D
              → Dense(32, relu) → Dense(1, linear)

    causal padding ensures convolutions only look at past timesteps.
    GlobalAveragePooling1D is more robust to window-size changes than Flatten.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(window_size, n_features)),
            layers.Conv1D(64, kernel_size=3, activation="relu", padding="causal"),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(32, kernel_size=3, activation="relu", padding="causal"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="linear"),
        ],
        name="CNN_Simple",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_cnn_deep(
    window_size: int,
    n_features: int,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Multi-scale 1D CNN with two parallel branches.

    Architecture:
        Input ──┬── Conv1D(64, k=3, causal) → MaxPooling1D(2) ──┐
                └── Conv1D(64, k=7, causal) → MaxPooling1D(2) ──┤
                                                                 Concatenate
                                                        → Conv1D(32, k=3, causal)
                                                        → GlobalAveragePooling1D
                                                        → Dense(32, relu) → Dropout(0.2)
                                                        → Dense(1, linear)

    Short kernels (k=3) capture rapid fluctuations; long kernels (k=7)
    capture slower daily patterns.
    """
    inputs = layers.Input(shape=(window_size, n_features))

    branch_short = layers.Conv1D(64, kernel_size=3, activation="relu", padding="causal")(inputs)
    branch_short = layers.MaxPooling1D(pool_size=2)(branch_short)

    branch_long = layers.Conv1D(64, kernel_size=7, activation="relu", padding="causal")(inputs)
    branch_long = layers.MaxPooling1D(pool_size=2)(branch_long)

    merged = layers.Concatenate()([branch_short, branch_long])
    x = layers.Conv1D(32, kernel_size=3, activation="relu", padding="causal")(merged)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="CNN_MultiScale")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model
