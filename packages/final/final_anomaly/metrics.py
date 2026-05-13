from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float | list[list[int]]]]:
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 201)))
    best_threshold = float(candidates[-1])
    best_metrics = score_predictions(y_true, scores >= best_threshold)

    for threshold in candidates:
        metrics = score_predictions(y_true, scores >= threshold)
        if (
            metrics["f1"] > best_metrics["f1"]
            or (
                metrics["f1"] == best_metrics["f1"]
                and metrics["recall"] > best_metrics["recall"]
            )
        ):
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | list[list[int]]]:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int).tolist(),
    }


def evaluate_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | list[list[int]]]:
    return score_predictions(y_true, scores >= threshold)
