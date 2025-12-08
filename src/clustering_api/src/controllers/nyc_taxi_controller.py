from fastapi import APIRouter
from clustering_api.src.services.nyc_taxi_service import NycTaxiService

router = APIRouter(prefix="/v1/nyc-taxi", tags=["NYC Taxi"])

service = NycTaxiService(
    file_path="./data/yellow_tripdata_2016-01_batch.csv",
    batch_size=500,
)


@router.get("/next-batch")
def next_batch():
    batch = service.next_batch()
    if batch is None:
        return {
            "message": "No more data available.",
            "batch_id": service.batch_id,
            "data": [],
        }
    return {
        "message": "Batch generated.",
        "batch_id": service.batch_id,
        "size": len(batch),
        "points": [p.model_dump() for p in batch],
    }
    
    
@router.get("/next-batch-cluster-points")
def next_batch_cluster_points():
    batch = service.next_batch_cluster_points()
    if batch is None:
        return {
            "message": "No more data available.",
            "batch_id": service.batch_id,
            "data": [],
        }
    return {
        "message": "Batch generated.",
        "batch_id": service.batch_id,
        "size": len(batch),
        "points": [p.model_dump() for p in batch],
    }


@router.post("/reset")
def reset_stream():
    service.reset()
    return {"message": "NYC Taxi stream reset.", "batch_id": service.batch_id}
