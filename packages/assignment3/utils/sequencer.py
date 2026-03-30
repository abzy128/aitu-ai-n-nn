"""Sliding-window sequence builder for LSTM input.

Applied separately to each split so no cross-boundary leakage occurs.
"""

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

    Parameters
    ----------
    X : (n_samples, n_features) scaled feature array
    y : (n_samples,) scaled target array
    window_size : number of past timesteps per sample

    Returns
    -------
    X_seq : (n_samples - window_size, window_size, n_features)
    y_seq : (n_samples - window_size,)
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
