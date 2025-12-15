import time
from pathlib import Path

import numpy as np
import pandas as pd


class StreamService:
    """Singleton service for generating synthetic streaming data.
    Supports parameter configuration and stream reset.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        points_per_cluster: int = 100,
        noise_ratio: float = 0.05,
        drift: float = 0.05,
        output_dir: str = "../../../data/",
    ):
        if not hasattr(self, "_initialized"):
            self.n_clusters = n_clusters
            self.points_per_cluster = points_per_cluster
            self.noise_ratio = noise_ratio
            self.drift = drift
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.centroids = [np.random.uniform(-5, 5, 2) for _ in range(n_clusters)]
            self.batch_id = 0
            self._initialized = True

    def _update_centroids(self):
        """Apply random drift to centroids."""
        for i in range(self.n_clusters):
            self.centroids[i] += np.random.uniform(-self.drift, self.drift, 2)

    def generate_batch(self) -> list[dict]:
        """Generate one batch of drifting clustered data."""
        self.batch_id += 1
        self._update_centroids()
        timestamp = time.time()
        points = []

        for cid, center in enumerate(self.centroids):
            cluster_points = np.random.normal(
                loc=center, scale=0.5, size=(self.points_per_cluster, 2),
            )
            for x, y in cluster_points:
                points.append(
                    {
                        "x": float(x),
                        "y": float(y),
                        "timestamp": timestamp,
                        "cluster_id": cid,
                        "batch_id": self.batch_id,
                        "noise": False,
                    },
                )

        n_noise = int(self.points_per_cluster * self.n_clusters * self.noise_ratio)
        noise_points = np.random.uniform(-8, 8, size=(n_noise, 2))
        for x, y in noise_points:
            points.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "timestamp": timestamp,
                    "cluster_id": -1,
                    "batch_id": self.batch_id,
                    "noise": True,
                },
            )

        return points

    def save_batch(self, filename: str | None = None) -> str:
        """Generate a batch and save it as a CSV file."""
        batch = self.generate_batch()
        df = pd.DataFrame(batch)
        if filename is None:
            filename = f"synthetic_batch_{self.batch_id}.csv"

        path = self.output_dir / filename
        df.to_csv(path, index=False)
        return str(path)

    def configure(
        self,
        n_clusters: int | None = None,
        points_per_cluster: int | None = None,
        noise_ratio: float | None = None,
        drift: float | None = None,
    ):
        """Update configuration dynamically."""
        if n_clusters is not None and n_clusters != self.n_clusters:
            self.n_clusters = n_clusters
            self.centroids = [np.random.uniform(-5, 5, 2) for _ in range(n_clusters)]
        if points_per_cluster is not None:
            self.points_per_cluster = points_per_cluster
        if noise_ratio is not None:
            self.noise_ratio = noise_ratio
        if drift is not None:
            self.drift = drift

    def reset_stream(self):
        """Reset stream to initial state (batch counter and centroids)."""
        self.batch_id = 0
        self.centroids = [np.random.uniform(-5, 5, 2) for _ in range(self.n_clusters)]

    def get_state(self) -> dict:
        """Return current configuration and state."""
        return {
            "n_clusters": self.n_clusters,
            "points_per_cluster": self.points_per_cluster,
            "noise_ratio": self.noise_ratio,
            "drift": self.drift,
            "batch_id": self.batch_id,
            "centroids": [c.tolist() for c in self.centroids],
        }


stream_service = StreamService()
