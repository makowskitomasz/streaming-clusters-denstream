from fastapi import APIRouter, Body

from clustering_api.src.services.stream_service import stream_service

router = APIRouter(prefix="/v1/stream", tags=["Stream"])


@router.get("/generate", summary="Generate synthetic data batch (in-memory)")
def generate_batch():
    data = stream_service.generate_batch()
    return {
        "batch_id": stream_service.batch_id,
        "points_generated": len(data),
        "points": data,
    }
    

@router.get("/generate-cluster-points", summary="Generate synthetic data as cluster points")
def generate_cluster_points():
    data = stream_service.generate_batch_cluster_points()
    return {
        "batch_id": stream_service.batch_id,
        "points_generated": len(data),
        "points": data,
    }


@router.get("/generate/save", summary="Generate and save synthetic data batch")
def generate_and_save():
    saved_path = stream_service.save_batch()
    return {
        "message": f"Batch {stream_service.batch_id} saved successfully.",
        "file_path": saved_path,
    }


@router.get("/generate-cluster-points/save", summary="Generate and save synthetic data batch with cluster points")
def generate_and_save_cluster_points():
    saved_path = stream_service.save_batch_cluster_points()
    return {
        "message": f"Cluster points batch {stream_service.batch_id} saved successfully.",
        "file_path": saved_path,
    }


@router.get("/state", summary="Get current stream generator state")
def get_stream_state():
    return stream_service.get_state()


@router.post("/configure", summary="Configure stream generator parameters")
def configure_stream(
    n_clusters: int = Body(None),
    points_per_cluster: int = Body(None),
    noise_ratio: float = Body(None),
    drift: float = Body(None),
):
    """Dynamically configure generator parameters."""
    stream_service.configure(
        n_clusters=n_clusters,
        points_per_cluster=points_per_cluster,
        noise_ratio=noise_ratio,
        drift=drift,
    )
    return {"message": "Stream configuration updated", "state": stream_service.get_state()}


@router.post("/reset", summary="Reset the stream to initial state")
def reset_stream():
    """Reset stream (batch counter and centroids)."""
    stream_service.reset_stream()
    return {"message": "Stream reset successfully", "state": stream_service.get_state()}
