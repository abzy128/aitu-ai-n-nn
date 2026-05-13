from __future__ import annotations

from datetime import time

from final_anomaly.inference import PredictionService, parse_request_datetime


def test_parse_request_datetime_end_of_day() -> None:
    parsed = parse_request_datetime("2026-03-25", end_of_day=True)

    assert parsed.date().isoformat() == "2026-03-25"
    assert parsed.time() == time.max


def test_prediction_service_lists_models() -> None:
    service = PredictionService()
    names = {model.name for model in service.list_models()}

    assert service.default_model == "forecast_residual_hgb"
    assert {
        "forecast_residual_hgb",
        "isolation_forest",
        "local_outlier_factor",
        "lstm_autoencoder",
        "one_class_svm",
    }.issubset(names)


def test_prediction_service_scores_each_model() -> None:
    service = PredictionService()
    start = parse_request_datetime("2026-03-30T10:30:00")
    end = parse_request_datetime("2026-03-30T12:30:00")

    for model in service.list_models():
        result = service.predict(start, end, model.name)
        assert result["modelName"] == model.name
        assert result["points"]
        assert {"dateTime", "activePower", "score", "isAnomaly"} <= set(result["points"][0])
