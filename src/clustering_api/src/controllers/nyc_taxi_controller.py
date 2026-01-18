from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Body
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


@router.get("/next-second")
def next_second() -> NextBatchResponse:
    """Return the next second of NYC taxi points."""
    batch = service.next_second()
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


@router.get("/next-second-cluster-points")
def next_second_cluster_points() -> NextBatchResponse:
    """Return the next second of NYC taxi cluster points."""
    batch = service.next_second_cluster_points()
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


@router.post("/configure")
def configure_stream(
    file_path: Annotated[str | None, Body()] = None,
    batch_size: Annotated[int | None, Body()] = None,
) -> dict[str, str | int]:
    """Configure NYC Taxi stream source."""
    service.configure(file_path=file_path, batch_size=batch_size)
    return {
        "message": "NYC Taxi stream configured.",
        "batch_id": service.batch_id,
        "file_path": str(service.file_path),
        "batch_size": service.batch_size,
    }


@router.get("/bounds")
def get_bounds() -> dict[str, float] | dict[str, str]:
    """Return min/max bounds for pickup longitude/latitude."""
    bounds = service.bounds()
    if bounds is None:
        return {"message": "No bounds available."}
    min_lon, max_lon, min_lat, max_lat = bounds
    return {
        "min_lon": min_lon,
        "max_lon": max_lon,
        "min_lat": min_lat,
        "max_lat": max_lat,
    }
