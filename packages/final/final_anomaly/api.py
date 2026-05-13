from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from final_anomaly.inference import InferenceError, PredictionService, STATIC_DIR, parse_request_datetime


app = FastAPI(title="Furnace1 Anomaly Detection API")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@lru_cache(maxsize=1)
def get_service() -> PredictionService:
    return PredictionService()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/stats")
def stats() -> FileResponse:
    return FileResponse(STATIC_DIR / "stats.html")


@app.get("/health")
def health() -> dict[str, Any]:
    service = get_service()
    return {
        "status": "ok",
        "datasetRows": int(len(service.raw_df)),
        "featureRows": int(len(service.feature_df)),
        "modelCount": len(service.artifacts),
        "datasetMinDateTime": service.min_datetime.isoformat(),
        "datasetMaxDateTime": service.max_datetime.isoformat(),
    }


@app.get("/api/models")
def models() -> dict[str, Any]:
    service = get_service()
    return {
        "defaultModel": service.default_model,
        "models": [model.__dict__ for model in service.list_models()],
    }


@app.get("/api/predictions")
def predictions(
    startDate: str = Query(..., description="Start date/time, for example 2026-03-25T00:00:00"),
    endDate: str = Query(..., description="End date/time, date-only values include the whole day"),
    modelName: str | None = Query(None, description="Model name from /api/models"),
) -> dict[str, Any]:
    service = get_service()
    try:
        start = parse_request_datetime(startDate)
        end = parse_request_datetime(endDate, end_of_day=True)
        return service.predict(start, end, modelName)
    except InferenceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def main() -> None:
    uvicorn.run("final_anomaly.api:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
