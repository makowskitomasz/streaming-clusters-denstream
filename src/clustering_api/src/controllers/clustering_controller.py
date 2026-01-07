from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from clustering_api.src.models.data_models import Cluster, ClusterPoint
from clustering_api.src.services.denstream_service import denstream_service

router = APIRouter(prefix="/v1/clustering", tags=["Clustering"])


class ClusterBatchPayload(BaseModel):
    """Batch payload containing points for DenStream updates."""

    points: list[ClusterPoint]


class DenStreamConfigPayload(BaseModel):
    """Config payload for DenStream hyperparameters."""

    decay_factor: float | None = None
    epsilon: float | None = None
    beta: float | None = None
    mu: float | None = None
    n_samples_init: int | None = None
    stream_speed: int | None = None


@dataclass(frozen=True, slots=True)
class ConfigResponse:
    message: str
    config: dict[str, Any]

    def to_dict(self) -> dict[str, object]:
        return {"message": self.message, "config": self.config}


@router.post("/denstream/update", summary="Update DenStream with a new batch")
def update_denstream(payload: ClusterBatchPayload) -> dict[str, list[Cluster]]:
    """Update DenStream with a new batch."""
    return denstream_service.update_clusters(payload.points)


@router.get("/denstream/clusters", summary="Fetch current DenStream clusters")
def get_denstream_clusters() -> dict[str, list[Cluster]]:
    """Return the latest DenStream clusters."""
    return denstream_service.get_current_clusters()


@router.post("/denstream/configure", summary="Update DenStream hyperparameters")
def configure_denstream(payload: DenStreamConfigPayload) -> dict[str, Any]:
    """Configure DenStream with provided overrides."""
    updated = denstream_service.configure(**payload.model_dump(exclude_none=True))
    response = ConfigResponse(
        message="DenStream configuration updated",
        config=updated,
    )
    return response.to_dict()


@router.get("/denstream/config", summary="Get current DenStream configuration")
def get_denstream_config() -> dict[str, Any]:
    """Return current DenStream configuration."""
    return denstream_service.get_config()
