"""Sliding-window sequence builder — identical to Stages 3 & 4."""

import numpy as np


def make_sequences(X: np.ndarray, window_size: int) -> np.ndarray:
    """Build (n_seq, window_size, n_features) arrays for autoencoder training.

    No target y — both input and output of the autoencoder are X_seq.
    """
    n = len(X)
    if n <= window_size:
        raise ValueError(f"window_size={window_size} >= n_samples={n}")

    n_seq = n - window_size
    n_features = X.shape[1]
    X_seq = np.empty((n_seq, window_size, n_features), dtype=np.float32)
    for i in range(n_seq):
        X_seq[i] = X[i : i + window_size]
    return X_seq
