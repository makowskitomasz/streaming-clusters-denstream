from fastapi import APIRouter

from clustering_api.src.services.nyc_taxi_service import NycTaxiService

router = APIRouter(prefix="/nyc-taxi", tags=["NYC Taxi"])

# Example path – you will place your CSV here
service = NycTaxiService(
    file_path="data/raw/nyc_taxi/yellow_tripdata_2016-01_batch.csv",
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
        "preview": [p.dict() for p in batch[:10]],
    }


@router.post("/reset")
def reset_stream():
    service.reset()
    return {"message": "NYC Taxi stream reset.", "batch_id": service.batch_id}
