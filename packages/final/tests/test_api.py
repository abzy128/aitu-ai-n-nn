from __future__ import annotations

from fastapi.testclient import TestClient

from final_anomaly.api import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_stats_page() -> None:
    response = client.get("/stats")

    assert response.status_code == 200
    assert "Model Evaluation Stats" in response.text


def test_models_endpoint_lists_exported_models() -> None:
    response = client.get("/api/models")

    assert response.status_code == 200
    body = response.json()
    assert body["defaultModel"] == "forecast_residual_hgb"
    assert len(body["models"]) >= 5


def test_predictions_endpoint() -> None:
    response = client.get(
        "/api/predictions",
        params={
            "startDate": "2026-03-30T10:30:00",
            "endDate": "2026-03-30T12:30:00",
            "modelName": "forecast_residual_hgb",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["points"]
    assert body["summary"]["anomalyCount"] > 0


def test_predictions_reject_invalid_model() -> None:
    response = client.get(
        "/api/predictions",
        params={
            "startDate": "2026-03-30",
            "endDate": "2026-03-30",
            "modelName": "missing",
        },
    )

    assert response.status_code == 400
