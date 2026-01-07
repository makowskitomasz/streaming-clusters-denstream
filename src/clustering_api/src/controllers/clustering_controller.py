from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from clustering_api.src.models.data_models import Cluster, ClusterPoint
from clustering_api.src.services.denstream_service import denstream_service

router = APIRouter(prefix="/v1/clustering", tags=["Clustering"])


class ClusterBatchPayload(BaseModel):
    points: list[ClusterPoint]


class DenStreamConfigPayload(BaseModel):
    decay_factor: float | None = None
    epsilon: float | None = None
    beta: float | None = None
    mu: float | None = None
    n_samples_init: int | None = None
    stream_speed: int | None = None


@router.post("/denstream/update", summary="Update DenStream with a new batch")
def update_denstream(payload: ClusterBatchPayload) -> dict[str, list[Cluster]]:
    return denstream_service.update_clusters(payload.points)


@router.get("/denstream/clusters", summary="Fetch current DenStream clusters")
def get_denstream_clusters() -> dict[str, list[Cluster]]:
    return denstream_service.get_current_clusters()


@router.post("/denstream/configure", summary="Update DenStream hyperparameters")
def configure_denstream(payload: DenStreamConfigPayload) -> dict[str, Any]:
    updated = denstream_service.configure(**payload.model_dump(exclude_none=True))
    return {"message": "DenStream configuration updated", "config": updated}


@router.get("/denstream/config", summary="Get current DenStream configuration")
def get_denstream_config() -> dict[str, Any]:
    return denstream_service.get_config()
