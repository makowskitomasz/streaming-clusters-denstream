from __future__ import annotations

from typing import List, Optional, Tuple, Literal

from pydantic import BaseModel, Field, validator


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
    cluster_id: Optional[str] = None
    timestamp: Optional[float] = None
    weight: float = 1.0

    @validator("weight")
    def _non_negative_weight(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("ClusterPoint weight must be positive")
        return value


class Cluster(BaseModel):
    """Unified cluster representation across DenStream/HDBSCAN outputs."""

    id: str = Field(..., description="Unique identifier for the cluster")
    centroid: Tuple[float, float]
    size: int = Field(..., ge=0)
    density: float = Field(..., ge=0.0)
    status: Literal["active", "decayed"] = "active"
    points: List[ClusterPoint] = Field(default_factory=list)

    @validator("centroid")
    def _centroid_length(cls, value: Tuple[float, float]) -> Tuple[float, float]:
        if len(value) != 2:
            raise ValueError("Centroid must be a 2D coordinate")
        return value

    @validator("points", always=True)
    def _size_consistency(cls, points: List[ClusterPoint], values):
        expected_size = values.get("size")
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
    def from_clusters(cls, clusters: List[Cluster], noise_points: int = 0) -> "ClusterSummary":
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
