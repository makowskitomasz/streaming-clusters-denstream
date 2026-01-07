from __future__ import annotations

from dataclasses import asdict, dataclass

from fastapi import APIRouter

from clustering_api.src.services.metrics_service import metrics_service

router = APIRouter(prefix="/v1/metrics", tags=["Metrics"])


@dataclass(frozen=True, slots=True)
class MetricsResponse:
    latest: dict[str, dict[str, object]]

    def to_dict(self) -> dict[str, dict[str, dict[str, object]]]:
        return {"latest": self.latest}


@router.get("/latest", summary="Fetch latest clustering metrics")
def get_latest_metrics() -> dict[str, dict[str, dict]]:
    """Return the latest metrics per model."""
    latest = metrics_service.get_latest()
    payload = {name: asdict(record) for name, record in latest.items()}
    response = MetricsResponse(latest=payload)
    return response.to_dict()
