from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Body

from clustering_api.src.services.stream_service import stream_service

router = APIRouter(prefix="/v1/stream", tags=["Stream"])


@dataclass(frozen=True, slots=True)
class BatchResponse:
    batch_id: int
    points_generated: int
    points: list[object]

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_id": self.batch_id,
            "points_generated": self.points_generated,
            "points": self.points,
        }


@dataclass(frozen=True, slots=True)
class SaveResponse:
    message: str
    file_path: str

    def to_dict(self) -> dict[str, str]:
        return {"message": self.message, "file_path": self.file_path}


@dataclass(frozen=True, slots=True)
class StateResponse:
    state: dict[str, object]

    def to_dict(self) -> dict[str, dict[str, object]]:
        return {"state": self.state}


@router.get("/generate", summary="Generate synthetic data batch (in-memory)")
def generate_batch() -> dict[str, object]:
    """Generate a batch of synthetic points."""
    data = stream_service.generate_batch()
    response = BatchResponse(
        batch_id=stream_service.batch_id,
        points_generated=len(data),
        points=data,
    )
    return response.to_dict()


@router.get(
    "/generate-cluster-points",
    summary="Generate synthetic data as cluster points",
)
def generate_cluster_points() -> dict[str, object]:
    """Generate a batch of synthetic cluster points."""
    data = stream_service.generate_batch_cluster_points()
    response = BatchResponse(
        batch_id=stream_service.batch_id,
        points_generated=len(data),
        points=data,
    )
    return response.to_dict()


@router.get("/generate/save", summary="Generate and save synthetic data batch")
def generate_and_save() -> dict[str, str]:
    """Generate and persist a synthetic batch to disk."""
    saved_path = stream_service.save_batch()
    response = SaveResponse(
        message=f"Batch {stream_service.batch_id} saved successfully.",
        file_path=saved_path,
    )
    return response.to_dict()


@router.get(
    "/generate-cluster-points/save",
    summary="Generate and save synthetic data batch with cluster points",
)
def generate_and_save_cluster_points() -> dict[str, str]:
    """Generate and persist a cluster-points batch to disk."""
    saved_path = stream_service.save_batch_cluster_points()
    response = SaveResponse(
        message=f"Cluster points batch {stream_service.batch_id} saved successfully.",
        file_path=saved_path,
    )
    return response.to_dict()


@router.get("/state", summary="Get current stream generator state")
def get_stream_state() -> dict[str, object]:
    """Return current stream configuration/state."""
    response = StateResponse(state=stream_service.get_state())
    return response.state


@router.post("/configure", summary="Configure stream generator parameters")
def configure_stream(
    n_clusters: Annotated[int | None, Body()] = None,
    points_per_cluster: Annotated[int | None, Body()] = None,
    noise_ratio: Annotated[float | None, Body()] = None,
    drift: Annotated[float | None, Body()] = None,
) -> dict[str, dict | str]:
    """Dynamically configure generator parameters."""
    stream_service.configure(
        n_clusters=n_clusters,
        points_per_cluster=points_per_cluster,
        noise_ratio=noise_ratio,
        drift=drift,
    )
    response = StateResponse(state=stream_service.get_state())
    return {
        "message": "Stream configuration updated",
        "state": response.state,
    }


@router.post("/reset", summary="Reset the stream to initial state")
def reset_stream() -> dict[str, object]:
    """Reset stream (batch counter and centroids)."""
    stream_service.reset_stream()
    response = StateResponse(state=stream_service.get_state())
    return {"message": "Stream reset successfully", "state": response.state}


@router.post("/pause", summary="Pause the stream")
def pause_stream() -> dict[str, object]:
    """Pause stream (state-only flag)."""
    stream_service.pause_stream()
    response = StateResponse(state=stream_service.get_state())
    return {"message": "Stream paused successfully", "state": response.state}
