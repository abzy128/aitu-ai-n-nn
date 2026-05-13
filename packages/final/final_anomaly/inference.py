from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from final_anomaly.features import add_proxy_labels, build_feature_frame, load_furnace1, make_sequences
from final_anomaly.lstm import LSTMAutoencoder, reconstruction_scores
from final_anomaly.train import DATASET_PATH, MODELS_DIR, REPORTS_DIR, ROOT


STATIC_DIR = ROOT / "static"


class InferenceError(ValueError):
    """Raised when a request cannot be served as a valid prediction."""


@dataclass(frozen=True)
class ModelInfo:
    name: str
    model_type: str
    threshold: float
    artifact_path: str
    validation: dict[str, Any] | None
    test: dict[str, Any] | None
    is_default: bool


class PredictionService:
    def __init__(
        self,
        dataset_path: Path = DATASET_PATH,
        models_dir: Path = MODELS_DIR,
        reports_dir: Path = REPORTS_DIR,
    ) -> None:
        self.dataset_path = dataset_path
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        self.raw_df = load_furnace1(str(dataset_path))
        self.feature_df = build_feature_frame(add_proxy_labels(self.raw_df))
        self.metrics = self._load_metrics()
        self.default_model = self._choose_default_model()
        self.artifacts = self._load_artifacts()

    @property
    def min_datetime(self) -> pd.Timestamp:
        return pd.Timestamp(self.raw_df["DateTime"].min())

    @property
    def max_datetime(self) -> pd.Timestamp:
        return pd.Timestamp(self.raw_df["DateTime"].max())

    def list_models(self) -> list[ModelInfo]:
        infos = []
        for name in sorted(self.artifacts):
            artifact = self.artifacts[name]
            metric_entry = self.metrics.get("models", {}).get(name, {})
            infos.append(
                ModelInfo(
                    name=name,
                    model_type=str(artifact["model_type"]),
                    threshold=float(artifact["threshold"]),
                    artifact_path=str(self.models_dir / f"{name}.joblib"),
                    validation=metric_entry.get("validation"),
                    test=metric_entry.get("test"),
                    is_default=name == self.default_model,
                )
            )
        return infos

    def predict(self, start: datetime, end: datetime, model_name: str | None = None) -> dict[str, Any]:
        if start > end:
            raise InferenceError("startDate must be before or equal to endDate")

        selected_model = model_name or self.default_model
        if selected_model not in self.artifacts:
            raise InferenceError(f"Unknown modelName: {selected_model}")

        artifact = self.artifacts[selected_model]
        if selected_model == "lstm_autoencoder":
            points = self._predict_lstm(artifact, start, end)
        else:
            points = self._predict_tabular(artifact, start, end)

        return {
            "modelName": selected_model,
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "threshold": float(artifact["threshold"]),
            "points": points,
            "summary": self._summary(points),
        }

    def _predict_tabular(
        self, artifact: dict[str, Any], start: datetime, end: datetime
    ) -> list[dict[str, Any]]:
        window = self.feature_df[
            (self.feature_df["DateTime"] >= pd.Timestamp(start))
            & (self.feature_df["DateTime"] <= pd.Timestamp(end))
        ].copy()
        if window.empty:
            return []

        feature_cols = artifact["feature_columns"]
        x = window[feature_cols].to_numpy(dtype=np.float64)
        if artifact["model_name"] == "forecast_residual_hgb":
            predicted = artifact["pipeline"].predict(x)
            scores = np.abs(window["active_power"].to_numpy(dtype=np.float64) - predicted)
        else:
            scores = -artifact["pipeline"].decision_function(x)

        return _rows_to_points(window, scores, float(artifact["threshold"]))

    def _predict_lstm(
        self, artifact: dict[str, Any], start: datetime, end: datetime
    ) -> list[dict[str, Any]]:
        scaler = artifact["scaler"]
        values = (
            self.feature_df["active_power"].to_numpy(dtype=np.float64) - float(scaler["mean"])
        ) / float(scaler["std"])
        labels = self.feature_df["proxy_anomaly"].to_numpy(dtype=np.int64)
        windows, _ = make_sequences(values, labels, int(artifact["sequence_length"]))
        if len(windows) == 0:
            return []

        architecture = artifact.get("architecture", {})
        model = LSTMAutoencoder(
            hidden_size=int(architecture.get("hidden_size", 32)),
            latent_size=int(architecture.get("latent_size", 16)),
        )
        state_dict_path = Path(artifact["state_dict_path"])
        if not state_dict_path.exists():
            state_dict_path = self.models_dir / artifact["weights_file"]
        model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
        scores = reconstruction_scores(model, windows)

        sequence_length = int(artifact["sequence_length"])
        scored_df = self.feature_df.iloc[sequence_length - 1 :].reset_index(drop=True)
        window_df = scored_df[
            (scored_df["DateTime"] >= pd.Timestamp(start))
            & (scored_df["DateTime"] <= pd.Timestamp(end))
        ].copy()
        if window_df.empty:
            return []
        selected_scores = scores[window_df.index.to_numpy()]
        return _rows_to_points(window_df, selected_scores, float(artifact["threshold"]))

    def _load_metrics(self) -> dict[str, Any]:
        metrics_path = self.reports_dir / "metrics.json"
        if not metrics_path.exists():
            return {}
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    def _choose_default_model(self) -> str:
        ranking = self.metrics.get("ranking_by_test_f1") or []
        if ranking:
            return str(ranking[0])
        return "forecast_residual_hgb"

    def _load_artifacts(self) -> dict[str, dict[str, Any]]:
        artifacts = {}
        for path in sorted(self.models_dir.glob("*.joblib")):
            artifact = joblib.load(path)
            if not isinstance(artifact, dict) or "model_name" not in artifact:
                continue
            artifacts[str(artifact["model_name"])] = artifact
        if not artifacts:
            raise RuntimeError(f"No model artifacts found in {self.models_dir}")
        return artifacts

    def _summary(self, points: list[dict[str, Any]]) -> dict[str, Any]:
        if not points:
            return {
                "totalPoints": 0,
                "anomalyCount": 0,
                "minDateTime": None,
                "maxDateTime": None,
                "minActivePower": None,
                "maxActivePower": None,
                "datasetMinDateTime": self.min_datetime.isoformat(),
                "datasetMaxDateTime": self.max_datetime.isoformat(),
            }
        values = [point["activePower"] for point in points]
        return {
            "totalPoints": len(points),
            "anomalyCount": sum(1 for point in points if point["isAnomaly"]),
            "minDateTime": points[0]["dateTime"],
            "maxDateTime": points[-1]["dateTime"],
            "minActivePower": float(min(values)),
            "maxActivePower": float(max(values)),
            "datasetMinDateTime": self.min_datetime.isoformat(),
            "datasetMaxDateTime": self.max_datetime.isoformat(),
        }


def parse_request_datetime(value: str, *, end_of_day: bool = False) -> datetime:
    text = value.strip()
    if not text:
        raise InferenceError("Date value cannot be empty")
    normalized = text.replace("Z", "+00:00")
    try:
        if "T" not in normalized and " " not in normalized and len(normalized) == 10:
            date = datetime.fromisoformat(normalized).date()
            return datetime.combine(date, time.max if end_of_day else time.min)
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise InferenceError(f"Invalid date value: {value}") from exc


def _rows_to_points(
    df: pd.DataFrame, scores: np.ndarray, threshold: float
) -> list[dict[str, Any]]:
    points = []
    for row, score in zip(df.itertuples(index=False), scores, strict=True):
        points.append(
            {
                "dateTime": pd.Timestamp(row.DateTime).isoformat(),
                "activePower": float(row.active_power),
                "score": float(score),
                "isAnomaly": bool(score >= threshold),
            }
        )
    return points
