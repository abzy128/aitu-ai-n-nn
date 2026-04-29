"""Anomaly label derivation and contrastive pair construction for Stage 6."""

from __future__ import annotations

import numpy as np
from tensorflow import keras


# ── Anomaly labels ──────────────────────────────────────────────────────────

def compute_reconstruction_mse(
    model: keras.Model,
    X: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Per-window mean squared reconstruction error, shape (n_windows,)."""
    X_recon = model.predict(X, batch_size=batch_size, verbose=0)
    return np.mean((X - X_recon) ** 2, axis=(1, 2))


def derive_anomaly_labels(
    autoencoder: keras.Model,
    X_train_seq: np.ndarray,
    X_val_seq: np.ndarray,
    X_test_seq: np.ndarray,
    percentile: float = 95.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute per-window reconstruction MSE and assign binary anomaly labels.

    Threshold = percentile of training MSE (same rule as Stage 5).
    Returns labels_train, labels_val, labels_test, threshold.
    Label 1 = anomalous, 0 = normal.
    """
    mse_train = compute_reconstruction_mse(autoencoder, X_train_seq)
    mse_val   = compute_reconstruction_mse(autoencoder, X_val_seq)
    mse_test  = compute_reconstruction_mse(autoencoder, X_test_seq)

    threshold = float(np.percentile(mse_train, percentile))

    labels_train = (mse_train > threshold).astype(np.int32)
    labels_val   = (mse_val   > threshold).astype(np.int32)
    labels_test  = (mse_test  > threshold).astype(np.int32)

    print(f"Anomaly threshold (p{percentile:.0f} of train MSE): {threshold:.6f}")
    for split, labels in [("Train", labels_train), ("Val", labels_val), ("Test", labels_test)]:
        n_anom = labels.sum()
        print(f"  {split}: {n_anom:,} anomalies / {len(labels):,} windows ({100*n_anom/len(labels):.1f}%)")

    return labels_train, labels_val, labels_test, threshold


# ── Pair construction ────────────────────────────────────────────────────────

def build_pairs(
    X_seq: np.ndarray,
    labels: np.ndarray,
    neg_per_anomaly: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct balanced positive/negative pairs from a single split.

    Positive pair (pair_label=0): two normal windows — should be close in embedding space.
    Negative pair (pair_label=1): one normal + one anomalous — should be far apart.

    Returns:
        pairs_A  — shape (n_pairs, window_size, n_features)
        pairs_B  — shape (n_pairs, window_size, n_features)
        pair_labels — shape (n_pairs,)  0=similar, 1=dissimilar
    """
    rng = np.random.default_rng(seed)

    normal_idx  = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]

    if len(anomaly_idx) == 0:
        raise ValueError("No anomalous windows in this split — cannot build negative pairs.")

    # Negative pairs: each anomaly paired with neg_per_anomaly random normals
    neg_normal_idx = rng.choice(normal_idx, size=len(anomaly_idx) * neg_per_anomaly, replace=True)
    neg_anomaly_idx = np.repeat(anomaly_idx, neg_per_anomaly)

    n_neg = len(neg_anomaly_idx)

    # Positive pairs: n_neg random normal-normal pairs for balance
    pos_a_idx = rng.choice(normal_idx, size=n_neg, replace=True)
    pos_b_idx = rng.choice(normal_idx, size=n_neg, replace=True)

    # Concatenate all pairs
    A_idx = np.concatenate([pos_a_idx, neg_normal_idx])
    B_idx = np.concatenate([pos_b_idx, neg_anomaly_idx])
    pair_labels = np.concatenate([
        np.zeros(n_neg, dtype=np.float32),
        np.ones(n_neg,  dtype=np.float32),
    ])

    # Shuffle
    perm = rng.permutation(len(pair_labels))
    pairs_A = X_seq[A_idx[perm]]
    pairs_B = X_seq[B_idx[perm]]
    pair_labels = pair_labels[perm]

    return pairs_A.astype(np.float32), pairs_B.astype(np.float32), pair_labels
