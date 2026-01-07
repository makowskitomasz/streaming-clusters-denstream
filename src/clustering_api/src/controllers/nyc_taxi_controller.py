from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from clustering_api.src.services.nyc_taxi_service import NycTaxiService

router = APIRouter(prefix="/v1/nyc-taxi", tags=["NYC Taxi"])

service = NycTaxiService(
    file_path="data/raw/nyc_taxi/yellow_tripdata_2016-01_batch.csv",
    batch_size=500,
)


class NextBatchEmpty(BaseModel):
    """Response when no more data is available."""

    message: Literal["No more data available."]
    batch_id: int
    data: list[dict]


class NextBatchOk(BaseModel):
    """Response when a batch is returned."""

    message: Literal["Batch generated."]
    batch_id: int
    size: int
    points: list[dict]


NextBatchResponse = NextBatchEmpty | NextBatchOk


@router.get("/next-batch")
def next_batch() -> NextBatchResponse:
    """Return the next batch from the NYC taxi stream."""
    batch = service.next_batch()
    if batch is None:
        return NextBatchEmpty(
            message="No more data available.",
            batch_id=service.batch_id,
            data=[],
        )

    return NextBatchOk(
        message="Batch generated.",
        batch_id=service.batch_id,
        size=len(batch),
        points=[p.model_dump() for p in batch],
    )


@router.get("/next-batch-cluster-points")
def next_batch_cluster_points() -> NextBatchResponse:
    """Return the next batch of cluster points."""
    batch = service.next_batch_cluster_points()
    if batch is None:
        return NextBatchEmpty(
            message="No more data available.",
            batch_id=service.batch_id,
            data=[],
        )
    return NextBatchOk(
        message="Batch generated.",
        batch_id=service.batch_id,
        size=len(batch),
        points=[p.model_dump() for p in batch],
    )


@router.post("/reset")
def reset_stream() -> dict[str, str | int]:
    """Reset the NYC Taxi stream iterator."""
    service.reset()
    return {"message": "NYC Taxi stream reset.", "batch_id": service.batch_id}
