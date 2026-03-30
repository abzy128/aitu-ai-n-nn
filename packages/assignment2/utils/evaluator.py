"""Metrics computation and evaluation plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras


def compute_metrics(
    model: keras.Model,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, dict[str, float]]:
    """Return RMSE and MAE (in °C) for each split.

    Parameters
    ----------
    splits : {"train": (X, y), "val": (X, y), "test": (X, y)}
    """
    results: dict[str, dict[str, float]] = {}
    for name, (X, y) in splits.items():
        y_pred = model.predict(X, verbose=0).flatten()
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y - y_pred)))
        results[name] = {"RMSE": rmse, "MAE": mae}
    return results


def metrics_table(
    mlp_metrics: dict[str, dict[str, float]],
    dnn_metrics: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Build a comparison DataFrame."""
    rows = []
    for model_name, metrics in [("MLP", mlp_metrics), ("DNN", dnn_metrics)]:
        row = {"Model": model_name}
        for split in ["train", "val", "test"]:
            row[f"{split.capitalize()} RMSE"] = round(metrics[split]["RMSE"], 4)
            row[f"{split.capitalize()} MAE"] = round(metrics[split]["MAE"], 4)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


def plot_loss_curves(
    histories: dict[str, keras.callbacks.History],
    figsize: tuple = (12, 5),
    save_path: Path | None = None,
) -> None:
    """Overlay train/val loss curves for all models in *histories*."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    colors = {"MLP": ("#1f77b4", "#aec7e8"), "DNN": ("#d62728", "#f7b6b6")}

    for model_name, history in histories.items():
        train_col, val_col = colors.get(model_name, ("#333", "#aaa"))
        axes[0].plot(history.history["loss"], color=train_col, label=f"{model_name} train")
        axes[0].plot(history.history["val_loss"], color=val_col, linestyle="--", label=f"{model_name} val")

        axes[1].plot(history.history["mae"], color=train_col, label=f"{model_name} train")
        axes[1].plot(history.history["val_mae"], color=val_col, linestyle="--", label=f"{model_name} val")

    for ax, title, ylabel in zip(axes, ["Loss (MSE)", "MAE"], ["MSE", "MAE (°C)"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    plt.suptitle("Training vs Validation curves — MLP vs DNN", fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_actual_vs_predicted(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    n: int = 500,
    figsize: tuple = (14, 5),
    save_path: Path | None = None,
) -> None:
    """Line plot of first *n* actual vs predicted temperatures on the test set."""
    y_pred = model.predict(X_test, verbose=0).flatten()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(y_test[:n], label="Actual", linewidth=1.0, alpha=0.8)
    ax.plot(y_pred[:n], label=f"Predicted ({model_name})", linewidth=1.0, alpha=0.8, linestyle="--")
    ax.set_title(f"Actual vs Predicted — {model_name} (first {n} test samples)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("T (°C)")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()
