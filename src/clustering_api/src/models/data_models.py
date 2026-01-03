from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class DataPoint(BaseModel):
    x: float
    y: float
    timestamp: float
    cluster_id: int
    source: str = "nyc_taxi"


class ClusterPoint(BaseModel):
    """Representation of a single sample assigned to a cluster."""

    x: float
    y: float
    cluster_id: str | None = None
    timestamp: float | None = None
    weight: float = 1.0

    @field_validator("weight")
    @classmethod
    def _non_negative_weight(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("ClusterPoint weight must be positive")
        return value


class Cluster(BaseModel):
    """Unified cluster representation across DenStream/HDBSCAN outputs."""

    id: str = Field(..., description="Unique identifier for the cluster")
    centroid: tuple[float, float]
    size: int = Field(..., ge=0)
    density: float = Field(..., ge=0.0)
    status: Literal["active", "decayed"] = "active"
    points: list[ClusterPoint] = Field(default_factory=list)

    @field_validator("centroid")
    def _centroid_length(cls, value: tuple[float, float]) -> tuple[float, float]:
        if len(value) != 2:
            raise ValueError("Centroid must be a 2D coordinate")
        return value

    @field_validator("points")
    def _size_consistency(cls, points: list[ClusterPoint], info):
        expected_size = info.data.get("size") if info.data else None
        if expected_size is None:
            return points
        if points and expected_size < len(points):
            raise ValueError("Cluster size cannot be smaller than number of points")
        return points


class ClusterSummary(BaseModel):
    """Aggregated statistics for a set of clusters."""

    total_clusters: int = Field(..., ge=0)
    total_points: int = Field(..., ge=0)
    avg_density: float = Field(..., ge=0.0)
    noise_ratio: float = Field(..., ge=0.0, le=1.0)

    @classmethod
    def from_clusters(cls, clusters: list[Cluster], noise_points: int = 0) -> ClusterSummary:
        total_clusters = len(clusters)
        total_points = sum(cluster.size for cluster in clusters)
        avg_density = sum(cluster.density for cluster in clusters) / total_clusters if total_clusters else 0.0
        total_items = total_points + noise_points
        noise_ratio = noise_points / total_items if total_items else 0.0
        return cls(
            total_clusters=total_clusters,
            total_points=total_points,
            avg_density=avg_density,
            noise_ratio=noise_ratio,
        )
