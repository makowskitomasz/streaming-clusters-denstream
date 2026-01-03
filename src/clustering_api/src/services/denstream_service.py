from __future__ import annotations

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
        if not batch_list:
            self._metrics.evaluate(
                np.empty((0, 2)),
                np.array([], dtype=int),
                model_name="denstream",
                batch_id=None,
            )
            return self.get_current_clusters()
        self.clusterer.update(batch_list)
        response = self._refresh_cache()
        self._evaluate_metrics(batch_list)
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

    def _evaluate_metrics(self, batch: list[Any]) -> None:
        features = self._batch_to_features(batch)
        labels = self._assign_labels(features)
        batch_id = self._extract_batch_id(batch)
        self._metrics.evaluate(
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


denstream_service = DenStreamService()
