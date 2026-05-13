from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from final_anomaly.features import (
    LabelConfig,
    SEQUENCE_LENGTH,
    add_proxy_labels,
    build_feature_frame,
    chronological_split,
    detector_feature_columns,
    forecast_feature_columns,
    load_furnace1,
    make_sequences,
)
from final_anomaly.lstm import LSTMTrainingConfig, reconstruction_scores, train_lstm_autoencoder
from final_anomaly.metrics import choose_threshold, evaluate_at_threshold


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "dataset" / "Furnace1.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def train_all(
    dataset_path: Path = DATASET_PATH,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw = load_furnace1(str(dataset_path))
    labeled = add_proxy_labels(raw, LabelConfig())
    features = build_feature_frame(labeled)
    train_df, val_df, test_df = chronological_split(features)

    metrics: dict[str, Any] = {
        "dataset": str(dataset_path),
        "rows": {
            "raw": int(len(raw)),
            "feature_frame": int(len(features)),
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "proxy_labels": {
            "low_power_threshold": LabelConfig().low_power_threshold,
            "jump_threshold": LabelConfig().jump_threshold,
            "train_anomalies": int(train_df["proxy_anomaly"].sum()),
            "validation_anomalies": int(val_df["proxy_anomaly"].sum()),
            "test_anomalies": int(test_df["proxy_anomaly"].sum()),
        },
        "models": {},
    }

    _train_classical_detectors(train_df, val_df, test_df, models_dir, metrics)
    _train_forecast_residual(train_df, val_df, test_df, models_dir, metrics)
    _train_lstm(train_df, val_df, test_df, models_dir, metrics)

    ranked = sorted(
        metrics["models"].items(),
        key=lambda item: item[1]["test"]["f1"],
        reverse=True,
    )
    metrics["ranking_by_test_f1"] = [name for name, _ in ranked]

    metrics_json = reports_dir / "metrics.json"
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_markdown_report(metrics, reports_dir / "metrics.md")
    return metrics


def _train_classical_detectors(
    train_df,
    val_df,
    test_df,
    models_dir: Path,
    metrics: dict[str, Any],
) -> None:
    feature_cols = detector_feature_columns(train_df)
    x_train = train_df[feature_cols].to_numpy(dtype=np.float64)
    y_train = train_df["proxy_anomaly"].to_numpy(dtype=np.int64)
    x_val = val_df[feature_cols].to_numpy(dtype=np.float64)
    y_val = val_df["proxy_anomaly"].to_numpy(dtype=np.int64)
    x_test = test_df[feature_cols].to_numpy(dtype=np.float64)
    y_test = test_df["proxy_anomaly"].to_numpy(dtype=np.int64)
    normal_x_train = x_train[y_train == 0]

    candidates = {
        "isolation_forest": IsolationForest(
            n_estimators=300,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        ),
        "one_class_svm": OneClassSVM(kernel="rbf", gamma="scale", nu=0.03),
        "local_outlier_factor": LocalOutlierFactor(
            n_neighbors=35,
            contamination="auto",
            novelty=True,
        ),
    }

    for name, model in candidates.items():
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
        pipeline.fit(normal_x_train)
        val_scores = -pipeline.decision_function(x_val)
        threshold, val_metrics = choose_threshold(y_val, val_scores)
        test_scores = -pipeline.decision_function(x_test)
        test_metrics = evaluate_at_threshold(y_test, test_scores, threshold)
        artifact_path = models_dir / f"{name}.joblib"

        artifact = {
            "model_name": name,
            "model_type": type(model).__name__,
            "pipeline": pipeline,
            "feature_columns": feature_cols,
            "score_direction": "higher_is_more_anomalous",
            "threshold": threshold,
            "validation": val_metrics,
            "test": test_metrics,
            "proxy_label_config": LabelConfig().__dict__,
        }
        joblib.dump(artifact, artifact_path)
        metrics["models"][name] = _metric_entry(artifact_path, artifact)


def _train_forecast_residual(
    train_df,
    val_df,
    test_df,
    models_dir: Path,
    metrics: dict[str, Any],
) -> None:
    feature_cols = forecast_feature_columns(train_df)
    target_col = "active_power"
    y_train_labels = train_df["proxy_anomaly"].to_numpy(dtype=np.int64)
    normal_train = train_df[y_train_labels == 0]

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=250,
                    learning_rate=0.04,
                    l2_regularization=0.01,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(
        normal_train[feature_cols].to_numpy(dtype=np.float64),
        normal_train[target_col].to_numpy(dtype=np.float64),
    )

    y_val = val_df["proxy_anomaly"].to_numpy(dtype=np.int64)
    val_scores = np.abs(
        val_df[target_col].to_numpy(dtype=np.float64)
        - pipeline.predict(val_df[feature_cols].to_numpy(dtype=np.float64))
    )
    threshold, val_metrics = choose_threshold(y_val, val_scores)

    y_test = test_df["proxy_anomaly"].to_numpy(dtype=np.int64)
    test_scores = np.abs(
        test_df[target_col].to_numpy(dtype=np.float64)
        - pipeline.predict(test_df[feature_cols].to_numpy(dtype=np.float64))
    )
    test_metrics = evaluate_at_threshold(y_test, test_scores, threshold)

    name = "forecast_residual_hgb"
    artifact_path = models_dir / f"{name}.joblib"
    artifact = {
        "model_name": name,
        "model_type": "HistGradientBoostingRegressor residual detector",
        "pipeline": pipeline,
        "feature_columns": feature_cols,
        "target_column": target_col,
        "score_direction": "higher_is_more_anomalous",
        "threshold": threshold,
        "validation": val_metrics,
        "test": test_metrics,
        "proxy_label_config": LabelConfig().__dict__,
    }
    joblib.dump(artifact, artifact_path)
    metrics["models"][name] = _metric_entry(artifact_path, artifact)


def _train_lstm(
    train_df,
    val_df,
    test_df,
    models_dir: Path,
    metrics: dict[str, Any],
) -> None:
    name = "lstm_autoencoder"
    mean = float(train_df.loc[train_df["proxy_anomaly"] == 0, "active_power"].mean())
    std = float(train_df.loc[train_df["proxy_anomaly"] == 0, "active_power"].std())
    if std == 0.0:
        raise ValueError("Cannot train LSTM because normal active_power std is zero")

    train_values = ((train_df["active_power"].to_numpy(dtype=np.float64) - mean) / std)
    val_values = ((val_df["active_power"].to_numpy(dtype=np.float64) - mean) / std)
    test_values = ((test_df["active_power"].to_numpy(dtype=np.float64) - mean) / std)

    train_windows, train_labels = make_sequences(
        train_values,
        train_df["proxy_anomaly"].to_numpy(dtype=np.int64),
        SEQUENCE_LENGTH,
    )
    val_windows, val_labels = make_sequences(
        val_values,
        val_df["proxy_anomaly"].to_numpy(dtype=np.int64),
        SEQUENCE_LENGTH,
    )
    test_windows, test_labels = make_sequences(
        test_values,
        test_df["proxy_anomaly"].to_numpy(dtype=np.int64),
        SEQUENCE_LENGTH,
    )
    normal_windows = train_windows[train_labels == 0]
    if len(normal_windows) == 0:
        raise ValueError("Cannot train LSTM because no normal training windows were found")

    model = train_lstm_autoencoder(normal_windows, LSTMTrainingConfig())
    val_scores = reconstruction_scores(model, val_windows)
    threshold, val_metrics = choose_threshold(val_labels, val_scores)
    test_scores = reconstruction_scores(model, test_windows)
    test_metrics = evaluate_at_threshold(test_labels, test_scores, threshold)

    weights_path = models_dir / f"{name}.pt"
    torch.save(model.state_dict(), weights_path)
    artifact_path = models_dir / f"{name}.joblib"
    artifact = {
        "model_name": name,
        "model_type": "PyTorch LSTM autoencoder",
        "weights_file": weights_path.name,
        "state_dict_path": str(weights_path),
        "sequence_length": SEQUENCE_LENGTH,
        "scaler": {"mean": mean, "std": std},
        "score_direction": "higher_is_more_anomalous",
        "threshold": threshold,
        "validation": val_metrics,
        "test": test_metrics,
        "proxy_label_config": LabelConfig().__dict__,
        "architecture": {"hidden_size": 32, "latent_size": 16},
    }
    joblib.dump(artifact, artifact_path)
    metrics["models"][name] = _metric_entry(artifact_path, artifact)


def _metric_entry(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact": str(path),
        "model_type": artifact["model_type"],
        "threshold": float(artifact["threshold"]),
        "validation": artifact["validation"],
        "test": artifact["test"],
    }


def _write_markdown_report(metrics: dict[str, Any], path: Path) -> None:
    lines = [
        "# Furnace1 Anomaly Detection Metrics",
        "",
        f"Dataset: `{metrics['dataset']}`",
        "",
        "## Splits",
        "",
        "| Split | Rows | Proxy anomalies |",
        "|---|---:|---:|",
        f"| Train | {metrics['rows']['train']} | {metrics['proxy_labels']['train_anomalies']} |",
        f"| Validation | {metrics['rows']['validation']} | {metrics['proxy_labels']['validation_anomalies']} |",
        f"| Test | {metrics['rows']['test']} | {metrics['proxy_labels']['test_anomalies']} |",
        "",
        "## Models",
        "",
        "| Rank | Model | Precision | Recall | F1 | Threshold | Artifact |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for rank, name in enumerate(metrics["ranking_by_test_f1"], start=1):
        item = metrics["models"][name]
        test = item["test"]
        lines.append(
            f"| {rank} | `{name}` | {test['precision']:.4f} | {test['recall']:.4f} | "
            f"{test['f1']:.4f} | {item['threshold']:.6f} | `{item['artifact']}` |"
        )
    lines.append("")
    lines.append(
        "Metrics use proxy labels: active_power below 10.0 or absolute one-minute change above 5.0."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
