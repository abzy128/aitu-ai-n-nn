"""Gaussian noise injection for the denoising experiment."""

import numpy as np


def add_gaussian_noise(X: np.ndarray, std: float = 0.1, seed: int = 42) -> np.ndarray:
    """Add i.i.d. N(0, std²) noise to every element of X.

    std=0.1 is appropriate when X is StandardScaler-normalised to ~N(0,1):
    it adds roughly 10% of typical signal variance — visible but not destructive.
    """
    rng = np.random.default_rng(seed)
    return (X + rng.normal(0.0, std, X.shape)).astype(np.float32)


def signal_to_noise_ratio(X_clean: np.ndarray, X_noisy: np.ndarray) -> float:
    """Mean per-feature SNR in dB across all samples and timesteps."""
    signal_power = np.mean(X_clean ** 2, axis=(0, 1))
    noise_power = np.mean((X_noisy - X_clean) ** 2, axis=(0, 1))
    snr_per_feature = 10 * np.log10(signal_power / (noise_power + 1e-10))
    return float(np.mean(snr_per_feature))
