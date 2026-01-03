from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import asdict
from typing import Any

import numpy as np
from loguru import logger

from clustering_api.src.adapters.base_clusterer import BaseClusterer
from clustering_api.src.adapters.denstream_clusterer import DenStreamClusterer
from clustering_api.src.config import config
from clustering_api.src.models.data_models import Cluster
from clustering_api.src.services.metrics_service import MetricsService, metrics_service


class DenStreamService:
    """Service orchestrating streaming updates for DenStream."""

    DEFAULT_CONFIG = asdict(config.denstream)

    def __init__(
        self,
        clusterer: BaseClusterer | None = None,
        clusterer_factory: Callable[..., BaseClusterer] | None = None,
        metrics: MetricsService | None = None,
        **config,
    ):
        self._config = {**self.DEFAULT_CONFIG, **config}
        self._factory = clusterer_factory or (lambda **cfg: DenStreamClusterer(**cfg))
        self.clusterer = clusterer or self._factory(**self._config)
        self._metrics = metrics or metrics_service
        self._active_clusters: list[Cluster] = []
        self._decayed_clusters: list[Cluster] = []

    def _refresh_cache(self) -> dict[str, list[Cluster]]:
        clusters = self.clusterer.get_clusters()
        self._active_clusters = clusters.get("active", [])
        self._decayed_clusters = clusters.get("decayed", [])
        return self.get_current_clusters()

    def update_clusters(self, batch: Iterable[Any]) -> dict[str, list[Cluster]]:
        batch_list = list(batch)
        start = time.perf_counter()
        if not batch_list:
            self._metrics.evaluate(
                np.empty((0, 2)),
                np.array([], dtype=int),
                model_name="denstream",
                batch_id=None,
            )
            self._log_batch_stats(
                n_samples=0,
                noise_ratio=0.0,
                batch_id=None,
                latency_ms=(time.perf_counter() - start) * 1000,
            )
            return self.get_current_clusters()
        self.clusterer.update(batch_list)
        response = self._refresh_cache()
        metrics_record = self._evaluate_metrics(batch_list)
        self._log_batch_stats(
            n_samples=metrics_record.n_samples,
            noise_ratio=metrics_record.noise_ratio,
            batch_id=metrics_record.batch_id,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        return response

    def get_current_clusters(self) -> dict[str, list[Cluster]]:
        return {
            "active_clusters": list(self._active_clusters),
            "decayed_clusters": list(self._decayed_clusters),
        }

    def configure(self, **config) -> dict[str, float]:
        updated = False
        for key, value in config.items():
            if value is not None and key in self._config:
                self._config[key] = value
                updated = True
        if updated:
            self.clusterer = self._factory(**self._config)
            self._active_clusters = []
            self._decayed_clusters = []
        return dict(self._config)

    def get_config(self) -> dict[str, float]:
        return dict(self._config)

    def _evaluate_metrics(self, batch: list[Any]):
        features = self._batch_to_features(batch)
        labels = self._assign_labels(features)
        batch_id = self._extract_batch_id(batch)
        return self._metrics.evaluate(
            features,
            labels,
            model_name="denstream",
            batch_id=batch_id,
        )

    def _batch_to_features(self, batch: list[Any]) -> np.ndarray:
        features: list[list[float]] = []
        for item in batch:
            if hasattr(item, "x") and hasattr(item, "y"):
                features.append([float(item.x), float(item.y)])
            elif isinstance(item, dict) and "x" in item and "y" in item:
                features.append([float(item["x"]), float(item["y"])])
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                features.append([float(item[0]), float(item[1])])
            else:
                raise TypeError(
                    f"Unsupported data type for DenStreamService: {type(item)}"
                )
        return np.asarray(features)

    def _assign_labels(self, features: np.ndarray) -> np.ndarray:
        if features.size == 0:
            return np.array([], dtype=int)
        if not self._active_clusters:
            logger.warning("DenStream metrics: no active clusters, labeling as noise")
            return np.full((features.shape[0],), -1, dtype=int)
        centroids = np.array([cluster.centroid for cluster in self._active_clusters])
        distances = np.linalg.norm(
            features[:, None, :] - centroids[None, :, :], axis=2
        )
        nearest = np.argmin(distances, axis=1)
        return nearest.astype(int)

    def _extract_batch_id(self, batch: list[Any]) -> str | None:
        batch_ids = []
        for item in batch:
            if hasattr(item, "batch_id"):
                batch_ids.append(getattr(item, "batch_id"))
        unique_ids = {value for value in batch_ids if value is not None}
        if len(unique_ids) == 1:
            return next(iter(unique_ids))
        return None

    def _log_batch_stats(
        self,
        *,
        n_samples: int,
        noise_ratio: float,
        batch_id: str | int | None,
        latency_ms: float,
    ) -> None:
        avg_density = self._average_density()
        logger.bind(
            event="clustering_batch",
            model_name="denstream",
            batch_id=batch_id,
            n_samples=n_samples,
            active_clusters=len(self._active_clusters),
            avg_density=avg_density,
            noise_ratio=noise_ratio,
            latency_ms=latency_ms,
            timestamp=time.time(),
        ).info("DenStream batch processed")

    def _average_density(self) -> float | None:
        if not self._active_clusters:
            return None
        return float(
            sum(cluster.density for cluster in self._active_clusters)
            / len(self._active_clusters)
        )


denstream_service = DenStreamService()
