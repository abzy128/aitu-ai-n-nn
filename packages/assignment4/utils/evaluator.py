"""Metrics and plots for Stage 4 CNN evaluation."""

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


def compute_persistence_metrics(
    y_test_scaled: np.ndarray,
    target_scaler: StandardScaler,
) -> dict[str, float]:
    """Persistence baseline: predict next temp = last observed temp.

    Uses a 1-step y-lag: y_pred[i] = y_test[i-1].
    Compares y_test[1:] vs y_test[:-1] after inverse-transforming to °C.
    This is correct when T(degC) is not in the feature matrix.
    """
    y_true = inverse_transform_target(y_test_scaled[1:], target_scaler)
    y_pred = inverse_transform_target(y_test_scaled[:-1], target_scaler)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"RMSE": rmse, "MAE": mae}


def build_comparison_table(
    persistence: dict,
    mlp_s2: dict,
    dnn_s2: dict,
    lstm_1l: dict,
    lstm_2l: dict,
    cnn_simple: dict,
    cnn_deep: dict,
) -> pd.DataFrame:
    """Build cross-stage comparison DataFrame (test split only)."""
    rows = [
        ("Persistence", persistence),
        ("MLP (Stage 2)", mlp_s2["test"]),
        ("DNN (Stage 2)", dnn_s2["test"]),
        ("LSTM 1-layer (Stage 3)", lstm_1l["test"]),
        ("LSTM 2-layer (Stage 3)", lstm_2l["test"]),
        ("CNN Simple (Stage 4)", cnn_simple["test"]),
        ("CNN Multi-Scale (Stage 4)", cnn_deep["test"]),
    ]
    return pd.DataFrame(
        [{"Model": name, "Test RMSE (°C)": round(m["RMSE"], 4), "Test MAE (°C)": round(m["MAE"], 4)}
         for name, m in rows]
    ).set_index("Model")


def plot_loss_curves(
    histories: dict[str, keras.callbacks.History],
    title: str = "Training curves — Simple vs Multi-Scale CNN",
    figsize: tuple = (12, 5),
    save_path: Path | None = None,
) -> None:
    """Overlay train/val loss curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    colors = {
        "CNN_Simple": ("#1f77b4", "#aec7e8"),
        "CNN_MultiScale": ("#d62728", "#f7b6b6"),
    }

    for model_name, history in histories.items():
        tc, vc = colors.get(model_name, ("#333", "#aaa"))
        axes[0].plot(history.history["loss"], color=tc, label=f"{model_name} train")
        axes[0].plot(history.history["val_loss"], color=vc, linestyle="--", label=f"{model_name} val")
        axes[1].plot(history.history["mae"], color=tc, label=f"{model_name} train")
        axes[1].plot(history.history["val_mae"], color=vc, linestyle="--", label=f"{model_name} val")

    for ax, title_ax, ylabel in zip(axes, ["Loss (MSE)", "MAE"], ["MSE", "MAE (scaled)"]):
        ax.set_title(title_ax)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_actual_vs_predicted(
    model: keras.Model,
    X_test: np.ndarray,
    y_test_scaled: np.ndarray,
    target_scaler: StandardScaler,
    model_name: str = "CNN",
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


def plot_residuals(
    model: keras.Model,
    X_test: np.ndarray,
    y_test_scaled: np.ndarray,
    target_scaler: StandardScaler,
    model_name: str = "CNN",
    bins: int = 60,
    figsize: tuple = (8, 5),
    save_path: Path | None = None,
) -> None:
    """Histogram of residuals (y_pred - y_true) in °C.

    Should be centered near 0 with narrow spread for a well-calibrated model.
    """
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_true = inverse_transform_target(y_test_scaled, target_scaler)
    y_pred = inverse_transform_target(y_pred_scaled, target_scaler)
    residuals = y_pred - y_true

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(residuals, bins=bins, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="zero error")
    ax.axvline(residuals.mean(), color="orange", linestyle="-", linewidth=1.2,
               label=f"mean={residuals.mean():.3f} °C")
    ax.set_title(f"Residual distribution — {model_name}")
    ax.set_xlabel("Prediction error (°C)  [y_pred − y_true]")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()
