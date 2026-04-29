"""Evaluation utilities for Stage 6 Siamese anomaly detection."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.metrics.pairwise import cosine_distances
from tensorflow import keras


# ── Training curves ──────────────────────────────────────────────────────────

def plot_training_curves(
    history: keras.callbacks.History,
    figsize: tuple = (12, 4),
    save_path: Path | None = None,
) -> None:
    """Plot contrastive loss and binary accuracy over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(history.history["loss"], label="Train")
    axes[0].plot(history.history["val_loss"], label="Val", linestyle="--")
    axes[0].set_title("Contrastive Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    if "binary_accuracy" in history.history:
        axes[1].plot(history.history["binary_accuracy"], label="Train")
        axes[1].plot(history.history["val_binary_accuracy"], label="Val", linestyle="--")
        axes[1].set_title("Binary Accuracy (distance < 0.5)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


# ── Threshold calibration ────────────────────────────────────────────────────

def calibrate_threshold(
    siamese: keras.Model,
    pairs_A_val: np.ndarray,
    pairs_B_val: np.ndarray,
    pair_labels_val: np.ndarray,
    n_thresholds: int = 200,
    figsize: tuple = (12, 4),
    save_path: Path | None = None,
) -> float:
    """
    Compute pairwise distances on validation pairs, plot distributions,
    and find the threshold maximizing F1 on the validation set.

    Returns the optimal threshold τ.
    """
    distances = siamese.predict(
        [pairs_A_val, pairs_B_val], batch_size=256, verbose=0
    ).squeeze()

    pos_mask = pair_labels_val == 0  # similar (normal-normal)
    neg_mask = pair_labels_val == 1  # dissimilar (normal-anomaly)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Distribution plot
    axes[0].hist(distances[pos_mask], bins=50, alpha=0.6, label="Positive (similar)", color="steelblue")
    axes[0].hist(distances[neg_mask], bins=50, alpha=0.6, label="Negative (dissimilar)", color="tomato")
    axes[0].set_title("Distance distributions — validation pairs")
    axes[0].set_xlabel("Euclidean distance")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # F1 over threshold grid
    thresholds = np.linspace(distances.min(), distances.max(), n_thresholds)
    f1_scores = []
    for τ in thresholds:
        y_pred = (distances >= τ).astype(int)
        f1_scores.append(f1_score(pair_labels_val, y_pred, zero_division=0))

    best_idx = int(np.argmax(f1_scores))
    best_tau = float(thresholds[best_idx])
    best_f1  = float(f1_scores[best_idx])

    axes[1].plot(thresholds, f1_scores, color="darkorange")
    axes[1].axvline(best_tau, color="red", linestyle="--", label=f"τ={best_tau:.3f}  F1={best_f1:.3f}")
    axes[1].set_title("F1 score vs distance threshold")
    axes[1].set_xlabel("Threshold τ")
    axes[1].set_ylabel("F1 score")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()

    y_pred_best = (distances >= best_tau).astype(int)
    prec = precision_score(pair_labels_val, y_pred_best, zero_division=0)
    rec  = recall_score(pair_labels_val, y_pred_best, zero_division=0)
    print(f"Optimal threshold τ = {best_tau:.4f}  |  F1={best_f1:.3f}  Precision={prec:.3f}  Recall={rec:.3f}")

    return best_tau


# ── Nearest-neighbor anomaly scoring ────────────────────────────────────────

def knn_anomaly_scores(
    shared_cnn: keras.Model,
    X_test_seq: np.ndarray,
    X_train_normal_seq: np.ndarray,
    k: int = 10,
    batch_size: int = 256,
) -> np.ndarray:
    """
    For each test window compute the mean cosine distance to its k nearest
    normal training windows.  Higher score → more anomalous.
    """
    Z_test  = shared_cnn.predict(X_test_seq,         batch_size=batch_size, verbose=0)
    Z_train = shared_cnn.predict(X_train_normal_seq, batch_size=batch_size, verbose=0)

    D = cosine_distances(Z_test, Z_train)  # (n_test, n_train_normal)
    k_nearest = np.sort(D, axis=1)[:, :k]
    return k_nearest.mean(axis=1)


# ── ROC / AUC ────────────────────────────────────────────────────────────────

def evaluate_roc(
    anomaly_scores: np.ndarray,
    true_labels: np.ndarray,
    tau: float,
    figsize: tuple = (6, 5),
    save_path: Path | None = None,
) -> float:
    """Plot ROC curve and return AUC.  true_labels: 1=anomalous."""
    fpr, tpr, _ = roc_curve(true_labels, anomaly_scores)
    auc = roc_auc_score(true_labels, anomaly_scores)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color="darkorange", label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Siamese Anomaly Detection")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()

    y_pred = (anomaly_scores >= tau).astype(int)
    prec = precision_score(true_labels, y_pred, zero_division=0)
    rec  = recall_score(true_labels, y_pred, zero_division=0)
    f1   = f1_score(true_labels, y_pred, zero_division=0)
    print(f"Test AUC={auc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")

    return auc


# ── Nearest-neighbor visualization ──────────────────────────────────────────

def plot_nn_comparison(
    shared_cnn: keras.Model,
    X_test_seq: np.ndarray,
    X_train_normal_seq: np.ndarray,
    test_labels: np.ndarray,
    feature_idx: int,
    feature_name: str,
    n_examples: int = 3,
    batch_size: int = 256,
    figsize: tuple = (14, 9),
    save_path: Path | None = None,
) -> None:
    """
    For n_examples anomalous test windows, plot the window and its nearest
    normal training neighbor side by side.
    """
    Z_test  = shared_cnn.predict(X_test_seq,         batch_size=batch_size, verbose=0)
    Z_train = shared_cnn.predict(X_train_normal_seq, batch_size=batch_size, verbose=0)

    anom_idx = np.where(test_labels == 1)[0]
    if len(anom_idx) == 0:
        print("No anomalous windows in test set — skipping NN visualization.")
        return

    chosen = anom_idx[:n_examples]
    D = cosine_distances(Z_test[chosen], Z_train)
    nn_idx = D.argmin(axis=1)

    fig, axes = plt.subplots(n_examples, 2, figsize=figsize, sharex=True)
    if n_examples == 1:
        axes = axes[np.newaxis, :]

    for row, (ti, ni) in enumerate(zip(chosen, nn_idx)):
        axes[row, 0].plot(X_test_seq[ti, :, feature_idx], color="tomato", linewidth=1.5)
        axes[row, 0].set_title(f"Anomalous test window #{ti}", fontsize=9)
        axes[row, 0].set_ylabel(feature_name, fontsize=8)

        axes[row, 1].plot(X_train_normal_seq[ni, :, feature_idx], color="steelblue", linewidth=1.5)
        axes[row, 1].set_title(f"Nearest normal training window #{ni}", fontsize=9)

    for ax in axes[-1]:
        ax.set_xlabel("Timestep within window")

    plt.suptitle(f"Anomalous windows vs nearest normal neighbors — {feature_name}", fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


# ── Error-distribution plot ──────────────────────────────────────────────────

def plot_mse_distribution(
    mse_train: np.ndarray,
    threshold: float,
    figsize: tuple = (8, 4),
    save_path: Path | None = None,
) -> None:
    """Reconstruction MSE histogram with threshold line."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(mse_train, bins=80, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"p95 threshold = {threshold:.5f}")
    ax.set_xlabel("Reconstruction MSE")
    ax.set_ylabel("Count")
    ax.set_title("Train reconstruction error distribution")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()
