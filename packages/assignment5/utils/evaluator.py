"""Visualization and evaluation utilities for Stage 5 Autoencoder."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from tensorflow import keras


# ── Reconstruction ──────────────────────────────────────────────────────────

def compute_reconstruction_mse(
    model: keras.Model,
    X: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Per-sample mean squared reconstruction error, shape (n_samples,)."""
    X_recon = model.predict(X, batch_size=batch_size, verbose=0)
    return np.mean((X - X_recon) ** 2, axis=(1, 2))


def plot_reconstruction_examples(
    autoencoder: keras.Model,
    X_clean: np.ndarray,
    X_noisy: np.ndarray,
    feature_idx: int,
    feature_name: str,
    n_examples: int = 4,
    seed: int = 0,
    figsize: tuple = (14, 10),
    save_path: Path | None = None,
) -> None:
    """Plot original / noisy / denoised for n random test windows."""
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(X_clean), size=n_examples, replace=False)

    X_denoised = autoencoder.predict(X_noisy[idxs], verbose=0)

    fig, axes = plt.subplots(n_examples, 1, figsize=figsize, sharex=True)
    for row, i in enumerate(idxs):
        ax = axes[row]
        ax.plot(X_clean[i, :, feature_idx], label="Original", linewidth=1.5)
        ax.plot(X_noisy[i, :, feature_idx], label="Noisy", linewidth=1.0, alpha=0.7, linestyle="--")
        ax.plot(X_denoised[row, :, feature_idx], label="Denoised", linewidth=1.5, linestyle=":")
        ax.set_ylabel(feature_name, fontsize=8)
        ax.legend(fontsize=7, loc="upper right")

    axes[0].set_title(f"Reconstruction examples — feature: {feature_name}")
    axes[-1].set_xlabel("Timestep within window")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_noise_comparison(
    X_clean: np.ndarray,
    X_noisy: np.ndarray,
    feature_idx: int,
    feature_name: str,
    window_idx: int = 0,
    figsize: tuple = (12, 4),
    save_path: Path | None = None,
) -> None:
    """Side-by-side plot of one window: clean vs noisy."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].plot(X_clean[window_idx, :, feature_idx], color="steelblue")
    axes[0].set_title(f"Original — {feature_name}")
    axes[0].set_xlabel("Timestep")

    axes[1].plot(X_noisy[window_idx, :, feature_idx], color="tomato")
    axes[1].set_title(f"Noisy (σ=0.1) — {feature_name}")
    axes[1].set_xlabel("Timestep")

    plt.suptitle("Gaussian noise visualisation (single window)", fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_training_loss(
    history: keras.callbacks.History,
    figsize: tuple = (8, 4),
    save_path: Path | None = None,
) -> None:
    """Plot autoencoder train vs validation reconstruction loss."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(history.history["loss"], label="Train loss")
    ax.plot(history.history["val_loss"], label="Val loss", linestyle="--")
    ax.set_title("Autoencoder reconstruction loss (MSE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (scaled)")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


# ── Anomaly detection ────────────────────────────────────────────────────────

def get_anomaly_mask(
    mse_all: np.ndarray,
    mse_train: np.ndarray,
    percentile: float = 95.0,
) -> tuple[np.ndarray, float]:
    """Flag windows whose reconstruction MSE exceeds the train-set percentile."""
    threshold = float(np.percentile(mse_train, percentile))
    return mse_all > threshold, threshold


def plot_anomaly_overlay(
    timestamps_all: pd.DatetimeIndex,
    T_all: np.ndarray,
    anomaly_mask: np.ndarray,
    threshold: float,
    mse_all: np.ndarray,
    figsize: tuple = (16, 6),
    save_path: Path | None = None,
) -> None:
    """Two-panel plot: T(degC) with anomaly shading + reconstruction MSE."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    ax1.plot(timestamps_all, T_all, linewidth=0.5, color="steelblue", label="T (degC)")
    # Shade anomaly windows
    for ts, is_anom in zip(timestamps_all, anomaly_mask):
        if is_anom:
            ax1.axvspan(ts, ts + pd.Timedelta(hours=1), color="red", alpha=0.3, linewidth=0)
    ax1.set_ylabel("T (°C)")
    ax1.set_title("Temperature with anomaly windows (red)")
    ax1.legend(fontsize=8)

    ax2.plot(timestamps_all, mse_all, linewidth=0.5, color="grey", label="Reconstruction MSE")
    ax2.axhline(threshold, color="red", linestyle="--", linewidth=1.0, label=f"Threshold (p95={threshold:.4f})")
    ax2.set_ylabel("MSE (scaled)")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


# ── Latent space ─────────────────────────────────────────────────────────────

def _season(month: int) -> str:
    return {12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer",
            9: "Autumn", 10: "Autumn", 11: "Autumn"}[month]


def _time_of_day(hour: int) -> str:
    if 0 <= hour < 6:
        return "Night"
    elif 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    else:
        return "Evening"


def plot_latent_space(
    Z: np.ndarray,
    timestamps: pd.DatetimeIndex,
    T_values: np.ndarray,
    figsize: tuple = (18, 5),
    save_path: Path | None = None,
) -> None:
    """Three PCA scatter plots of the 2D latent space colored by season, hour, temperature."""
    pca = PCA(n_components=2)
    Z2 = pca.fit_transform(Z)
    var_explained = pca.explained_variance_ratio_

    seasons = [_season(ts.month) for ts in timestamps]
    times = [_time_of_day(ts.hour) for ts in timestamps]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1 — Season
    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    season_colors = {"Winter": "#4a90d9", "Spring": "#5cb85c", "Summer": "#f0ad4e", "Autumn": "#d9534f"}
    for s in season_order:
        mask = np.array([x == s for x in seasons])
        axes[0].scatter(Z2[mask, 0], Z2[mask, 1], s=1, alpha=0.3, label=s, color=season_colors[s])
    axes[0].set_title(f"Season\n(PCA var: {var_explained[0]:.1%} + {var_explained[1]:.1%})")
    axes[0].legend(markerscale=5, fontsize=8)

    # Panel 2 — Time of day
    tod_order = ["Night", "Morning", "Afternoon", "Evening"]
    tod_colors = {"Night": "#2c3e50", "Morning": "#f39c12", "Afternoon": "#e74c3c", "Evening": "#8e44ad"}
    for t in tod_order:
        mask = np.array([x == t for x in times])
        axes[1].scatter(Z2[mask, 0], Z2[mask, 1], s=1, alpha=0.3, label=t, color=tod_colors[t])
    axes[1].set_title("Time of day")
    axes[1].legend(markerscale=5, fontsize=8)

    # Panel 3 — Temperature (continuous)
    sc = axes[2].scatter(Z2[:, 0], Z2[:, 1], c=T_values, s=1, alpha=0.4, cmap="RdBu_r")
    fig.colorbar(sc, ax=axes[2], label="T (°C)")
    axes[2].set_title("Temperature (°C)")

    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    plt.suptitle("Latent space (PCA 2D projection)", fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


# ── Clustering ───────────────────────────────────────────────────────────────

def plot_elbow(
    inertias: list[float],
    k_range: range,
    figsize: tuple = (7, 4),
    save_path: Path | None = None,
) -> None:
    """Elbow plot: K-Means inertia vs number of clusters."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(list(k_range), inertias, marker="o")
    ax.set_title("K-Means elbow — latent space")
    ax.set_xlabel("K (number of clusters)")
    ax.set_ylabel("Inertia")
    ax.set_xticks(list(k_range))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_cluster_profiles(
    centroids_decoded: np.ndarray,
    feature_idx: int,
    feature_name: str,
    figsize: tuple = (12, 5),
    save_path: Path | None = None,
) -> None:
    """Plot reconstructed feature profile for each cluster centroid."""
    K = len(centroids_decoded)
    fig, axes = plt.subplots(1, K, figsize=figsize, sharey=True)
    if K == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, K))
    for k, (ax, color) in enumerate(zip(axes, colors)):
        ax.plot(centroids_decoded[k, :, feature_idx], color=color, linewidth=2)
        ax.set_title(f"Cluster {k}")
        ax.set_xlabel("Timestep")
        if k == 0:
            ax.set_ylabel(feature_name)
    plt.suptitle(f"Decoded cluster centroids — {feature_name}", fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_cluster_timeline(
    timestamps: pd.DatetimeIndex,
    labels: np.ndarray,
    K: int,
    figsize: tuple = (16, 4),
    save_path: Path | None = None,
) -> None:
    """Stacked area chart of cluster assignment proportions over time."""
    df = pd.DataFrame({"ts": timestamps, "cluster": labels})
    df["month"] = df["ts"].dt.to_period("M")
    monthly = df.groupby(["month", "cluster"]).size().unstack(fill_value=0)
    monthly_pct = monthly.div(monthly.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    monthly_pct.plot.area(ax=ax, colormap="tab10", alpha=0.8)
    ax.set_title("Cluster distribution over time (monthly proportions)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Proportion")
    ax.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()
