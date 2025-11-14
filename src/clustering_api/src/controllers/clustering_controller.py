from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from clustering_api.src.models.data_models import ClusterPoint
from clustering_api.src.services.denstream_service import denstream_service


router = APIRouter(prefix="/v1/clustering", tags=["Clustering"])


class ClusterBatchPayload(BaseModel):
    points: List[ClusterPoint]


class DenStreamConfigPayload(BaseModel):
    decay_factor: Optional[float] = None
    epsilon: Optional[float] = None
    beta: Optional[float] = None
    mu: Optional[float] = None
    n_samples_init: Optional[int] = None
    stream_speed: Optional[int] = None


@router.post("/denstream/update", summary="Update DenStream with a new batch")
def update_denstream(payload: ClusterBatchPayload):
    return denstream_service.update_clusters(payload.points)


@router.get("/denstream/clusters", summary="Fetch current DenStream clusters")
def get_denstream_clusters():
    return denstream_service.get_current_clusters()


@router.post("/denstream/configure", summary="Update DenStream hyperparameters")
def configure_denstream(payload: DenStreamConfigPayload):
    updated = denstream_service.configure(**payload.dict(exclude_none=True))
    return {"message": "DenStream configuration updated", "config": updated}


@router.get("/denstream/config", summary="Get current DenStream configuration")
def get_denstream_config():
    return denstream_service.get_config()
