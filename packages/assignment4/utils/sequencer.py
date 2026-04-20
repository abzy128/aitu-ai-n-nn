"""Sliding-window sequence builder — identical to Stage 3."""

import numpy as np


def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (samples, window_size, n_features) input arrays.

    For each position i:
        X_seq[i] = X[i : i + window_size]   shape (window_size, n_features)
        y_seq[i] = y[i + window_size]        scalar
    """
    n = len(X)
    if n <= window_size:
        raise ValueError(f"window_size={window_size} >= n_samples={n}")

    n_seq = n - window_size
    n_features = X.shape[1]

    X_seq = np.empty((n_seq, window_size, n_features), dtype=np.float32)
    y_seq = np.empty(n_seq, dtype=np.float32)

    for i in range(n_seq):
        X_seq[i] = X[i : i + window_size]
        y_seq[i] = y[i + window_size]

    return X_seq, y_seq
