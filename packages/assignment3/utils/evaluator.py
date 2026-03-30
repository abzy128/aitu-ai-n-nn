"""Metrics and plots for Stage 3 LSTM evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


def inverse_transform_target(
    y_scaled: np.ndarray,
    target_scaler: StandardScaler,
) -> np.ndarray:
    """Inverse-transform a 1-D scaled target array back to °C."""
    return target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()


def compute_metrics(
    model: keras.Model,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    target_scaler: StandardScaler,
) -> dict[str, dict[str, float]]:
    """Return RMSE and MAE in °C for each split after inverse-transforming.

    Parameters
    ----------
    splits : {"train": (X_seq, y_scaled), "val": ..., "test": ...}
    """
    results: dict[str, dict[str, float]] = {}
    for name, (X, y_scaled) in splits.items():
        y_pred_scaled = model.predict(X, verbose=0).flatten()
        y_true = inverse_transform_target(y_scaled, target_scaler)
        y_pred = inverse_transform_target(y_pred_scaled, target_scaler)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        results[name] = {"RMSE": rmse, "MAE": mae}
    return results


def metrics_table(
    stage2_mlp: dict,
    stage2_dnn: dict,
    lstm_1l: dict,
    lstm_2l: dict,
) -> pd.DataFrame:
    """Build cross-stage comparison DataFrame (test split only)."""
    rows = []
    for name, m in [
        ("MLP (Stage 2)", stage2_mlp),
        ("DNN (Stage 2)", stage2_dnn),
        ("LSTM 1-layer", lstm_1l),
        ("LSTM 2-layer", lstm_2l),
    ]:
        rows.append({
            "Model": name,
            "Test RMSE (°C)": round(m["test"]["RMSE"], 4),
            "Test MAE (°C)": round(m["test"]["MAE"], 4),
        })
    return pd.DataFrame(rows).set_index("Model")


def plot_loss_curves(
    histories: dict[str, keras.callbacks.History],
    figsize: tuple = (12, 5),
    save_path: Path | None = None,
) -> None:
    """Overlay train/val loss and MAE curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    colors = {
        "LSTM_1L": ("#1f77b4", "#aec7e8"),
        "LSTM_2L": ("#d62728", "#f7b6b6"),
    }

    for model_name, history in histories.items():
        tc, vc = colors.get(model_name, ("#333", "#aaa"))
        axes[0].plot(history.history["loss"], color=tc, label=f"{model_name} train")
        axes[0].plot(history.history["val_loss"], color=vc, linestyle="--", label=f"{model_name} val")
        axes[1].plot(history.history["mae"], color=tc, label=f"{model_name} train")
        axes[1].plot(history.history["val_mae"], color=vc, linestyle="--", label=f"{model_name} val")

    for ax, title, ylabel in zip(axes, ["Loss (MSE)", "MAE"], ["MSE", "MAE (scaled)"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    plt.suptitle("Training curves — LSTM 1-layer vs 2-layer", fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_actual_vs_predicted(
    model: keras.Model,
    X_test: np.ndarray,
    y_test_scaled: np.ndarray,
    target_scaler: StandardScaler,
    model_name: str = "LSTM",
    n: int = 500,
    figsize: tuple = (14, 5),
    save_path: Path | None = None,
) -> None:
    """Line plot of first *n* actual vs predicted temperatures (°C)."""
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_true = inverse_transform_target(y_test_scaled, target_scaler)
    y_pred = inverse_transform_target(y_pred_scaled, target_scaler)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(y_true[:n], label="Actual", linewidth=1.0, alpha=0.8)
    ax.plot(y_pred[:n], label=f"Predicted ({model_name})", linewidth=1.0, alpha=0.8, linestyle="--")
    ax.set_title(f"Actual vs Predicted — {model_name} (first {n} test samples)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("T (°C)")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_window_experiment(
    window_sizes: list[int],
    rmse_values: list[float],
    mae_values: list[float],
    figsize: tuple = (8, 4),
    save_path: Path | None = None,
) -> None:
    """Line plot of RMSE and MAE vs window size."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(window_sizes, rmse_values, marker="o", label="RMSE (°C)")
    ax.plot(window_sizes, mae_values, marker="s", linestyle="--", label="MAE (°C)")
    ax.set_title("Window size experiment — Test RMSE & MAE")
    ax.set_xlabel("Window size (timesteps)")
    ax.set_ylabel("Error (°C)")
    ax.set_xticks(window_sizes)
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()
