"""Training loop with EarlyStopping and ModelCheckpoint (reused from Stage 2)."""

from pathlib import Path

import numpy as np
from tensorflow import keras


def compile_and_fit(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    checkpoint_path: str | Path,
    epochs: int = 200,
    batch_size: int = 64,
    patience: int = 10,
    verbose: int = 1,
) -> keras.callbacks.History:
    """Fit *model* with EarlyStopping and ModelCheckpoint.

    Best weights (lowest val_loss) are restored via
    EarlyStopping(restore_best_weights=True) and saved to *checkpoint_path*.
    batch_size default is 64 (LSTM training is more stable with smaller batches
    than the 256 used for MLP/DNN in Stage 2).
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )
    return history
