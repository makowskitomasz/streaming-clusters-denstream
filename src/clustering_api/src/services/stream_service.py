from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from clustering_api.src.models.data_models import (
    ClusterPoint,
    DataPoint,
    map_batch_to_clusterpoints,
)


class StreamConfigError(ValueError):
    """Raised when stream configuration is invalid."""


@dataclass(frozen=True, slots=True)
class StreamState:
    """Snapshot of the stream configuration and state."""

    n_clusters: int
    points_per_cluster: int
    noise_ratio: float
    drift: float
    batch_id: int
    centroids: list[list[float]]
    paused: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "n_clusters": self.n_clusters,
            "points_per_cluster": self.points_per_cluster,
            "noise_ratio": self.noise_ratio,
            "drift": self.drift,
            "batch_id": self.batch_id,
            "centroids": self.centroids,
            "paused": self.paused,
        }


class StreamService:
    """Singleton service generating drifting clustered data batches."""

    def __init__(
        self,
        n_clusters: int = 3,
        points_per_cluster: int = 100,
        noise_ratio: float = 0.05,
        drift: float = 0.05,
        output_dir: str = "./data",
    ) -> None:
        self._n_clusters = n_clusters
        self._points_per_cluster = points_per_cluster
        self._noise_ratio = noise_ratio
        self._drift = drift
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._centroids = self._initialize_centroids(self._n_clusters)
        self._batch_id = 0
        self._paused = False

    @property
    def batch_id(self) -> int:
        return self._batch_id

    @property
    def n_clusters(self) -> int:
        return self._n_clusters

    @property
    def points_per_cluster(self) -> int:
        return self._points_per_cluster

    @property
    def noise_ratio(self) -> float:
        return self._noise_ratio

    @property
    def drift(self) -> float:
        return self._drift

    def generate_batch(self) -> list[DataPoint]:
        """Generate one batch consisting of clustered and noise points."""
        return self._generate_records()

    def generate_batch_cluster_points(self) -> list[ClusterPoint]:
        """Generate one batch of clustered/noise points as ClusterPoint."""
        data_points = self._generate_records()
        return map_batch_to_clusterpoints(data_points)

    def generate_custom_batch(
        self,
        *,
        centroids: np.ndarray,
        points_per_cluster: int,
        noise_ratio: float,
        rng: np.random.Generator,
        batch_id: int | None = None,
        noise_bounds: tuple[float, float] = (-8.0, 8.0),
    ) -> list[ClusterPoint]:
        """Generate a deterministic batch from supplied centroids and noise ratio."""
        if centroids.ndim != 2 or centroids.shape[1] != 2:
            msg = (
                "centroids must be a 2D array with shape (k, 2), "
                f"got {centroids.shape}"
            )
            raise StreamConfigError(msg)
        if points_per_cluster <= 0:
            msg = f"points_per_cluster must be greater than 0, got {points_per_cluster}"
            raise StreamConfigError(msg)
        if noise_ratio < 0:
            msg = f"noise_ratio must be non-negative, got {noise_ratio}"
            raise StreamConfigError(msg)
        if noise_bounds[0] >= noise_bounds[1]:
            msg = f"noise_bounds must be an increasing range, got {noise_bounds}"
            raise StreamConfigError(msg)
        timestamp = time.time()
        total_clusters = centroids.shape[0]
        noise_count = int(points_per_cluster * total_clusters * noise_ratio)
        cluster_points = rng.normal(
            loc=0.0,
            scale=0.5,
            size=(total_clusters, points_per_cluster, 2),
        )
        cluster_points = centroids[:, None, :] + cluster_points
        noise_points = rng.uniform(
            noise_bounds[0], noise_bounds[1], size=(noise_count, 2),
        )
        records: list[ClusterPoint] = []

        records.extend(
            ClusterPoint(
                x=float(point[0]),
                y=float(point[1]),
                cluster_id=str(cluster_id),
                timestamp=timestamp,
                batch_id=batch_id,
                noise=False,
            )
            for cluster_id, points in enumerate(cluster_points)
            for point in points
        )

        records.extend(
            ClusterPoint(
                x=float(point[0]),
                y=float(point[1]),
                cluster_id=None,
                timestamp=timestamp,
                batch_id=batch_id,
                noise=True,
            )
            for point in noise_points
        )

        return records

    def save_batch(self, filename: str | None = None) -> str:
        """Generate a batch and persist it as a JSON file."""
        batch = self.generate_batch()
        df = pd.DataFrame([point.model_dump() for point in batch])
        output_name = filename or f"synthetic_batch_{self.batch_id}.json"
        path = self._output_dir / output_name
        df.to_json(path, orient="records", indent=4)
        return str(path)

    def save_batch_cluster_points(self, filename: str | None = None) -> str:
        """Generate a batch and persist it as a JSON file with ClusterPoint format."""
        batch = self.generate_batch_cluster_points()
        df = pd.DataFrame([point.model_dump() for point in batch])
        output_name = filename or f"synthetic_cluster_points_batch_{self.batch_id}.json"
        path = self._output_dir / output_name
        df.to_json(path, orient="records", indent=4)
        return str(path)

    def configure(
        self,
        n_clusters: int | None = None,
        points_per_cluster: int | None = None,
        noise_ratio: float | None = None,
        drift: float | None = None,
    ) -> None:
        """Update configuration dynamically."""
        if n_clusters is not None and n_clusters != self._n_clusters:
            self._n_clusters = n_clusters
            self._centroids = self._initialize_centroids(n_clusters)
        if points_per_cluster is not None:
            self._points_per_cluster = points_per_cluster
        if noise_ratio is not None:
            self._noise_ratio = noise_ratio
        if drift is not None:
            self._drift = drift

    def reset_stream(self) -> None:
        """Reset stream to initial state (batch counter and centroids)."""
        self._batch_id = 0
        self._centroids = self._initialize_centroids(self._n_clusters)
        self._paused = False

    def pause_stream(self) -> None:
        """Pause the stream (state-only flag)."""
        self._paused = True

    def get_state(self) -> dict:
        """Return current configuration and state."""
        state = StreamState(
            n_clusters=self._n_clusters,
            points_per_cluster=self._points_per_cluster,
            noise_ratio=self._noise_ratio,
            drift=self._drift,
            batch_id=self._batch_id,
            centroids=self._centroids.tolist(),
            paused=self._paused,
        )
        return state.to_dict()

    def _total_points_per_batch(self) -> int:
        return self._n_clusters * self._points_per_cluster + self._noise_points_count()

    def _noise_points_count(self) -> int:
        return int(self._points_per_cluster * self._n_clusters * self._noise_ratio)

    def _update_centroids(self) -> None:
        """Apply random drift to all centroids."""
        rng = np.random.default_rng()
        drift_vector = rng.uniform(
            -self._drift, self._drift, size=self._centroids.shape
        )
        self._centroids = self._centroids + drift_vector

    def _initialize_centroids(self, n_clusters: int) -> np.ndarray:
        rng = np.random.default_rng()
        return rng.uniform(-5, 5, size=(n_clusters, 2))

    def _populate_cluster_records(
        self, records: list[DataPoint | None], timestamp: float,
    ) -> int:
        cluster_points = self._generate_cluster_points()
        cluster_ids = np.repeat(np.arange(self._n_clusters), self._points_per_cluster)
        flattened = cluster_points.reshape(-1, 2)
        for idx, (point, cluster_id) in enumerate(zip(flattened, cluster_ids)):
            records[idx] = self._build_record(point, timestamp, cluster_id, False)
        return len(flattened)

    def _populate_noise_records(
        self,
        records: list[DataPoint | None],
        timestamp: float,
        start_index: int,
    ) -> None:
        noise_points = self._generate_noise_points()
        for offset, point in enumerate(noise_points):
            records[start_index + offset] = self._build_record(
                point, timestamp, -1, True
            )

    def _generate_cluster_points(self) -> np.ndarray:
        rng = np.random.default_rng()
        noise_component = rng.normal(
            loc=0.0,
            scale=0.5,
            size=(self._n_clusters, self._points_per_cluster, 2),
        )
        return self._centroids[:, None, :] + noise_component

    def _generate_noise_points(self) -> np.ndarray:
        rng = np.random.default_rng()
        return rng.uniform(-8, 8, size=(self._noise_points_count(), 2))

    def _build_record(
        self, point: np.ndarray, timestamp: float, cluster_id: int, noise: bool,
    ) -> DataPoint:
        return DataPoint(
            x=float(point[0]),
            y=float(point[1]),
            timestamp=timestamp,
            cluster_id=cluster_id,
            source="synthetic",
            batch_id=self._batch_id,
            noise=noise,
        )

    def _generate_records(self) -> list[DataPoint]:
        self._batch_id += 1
        self._update_centroids()
        timestamp = time.time()
        total_points = self._total_points_per_batch()
        records: list[DataPoint | None] = [None] * total_points

        next_index = self._populate_cluster_records(records, timestamp)
        self._populate_noise_records(records, timestamp, next_index)

        return [cast("DataPoint", record) for record in records]


stream_service = StreamService()
