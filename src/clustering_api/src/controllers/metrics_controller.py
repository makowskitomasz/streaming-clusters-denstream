from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter

from clustering_api.src.services.metrics_service import metrics_service

router = APIRouter(prefix="/v1/metrics", tags=["Metrics"])


@router.get("/latest", summary="Fetch latest clustering metrics")
def get_latest_metrics() -> dict[str, dict[str, dict]]:
    """Return the latest metrics per model."""
    latest = metrics_service.get_latest()
    payload = {name: asdict(record) for name, record in latest.items()}
    return {"latest": payload}
