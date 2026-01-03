from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict
from typing import Any

from clustering_api.src.adapters.base_clusterer import BaseClusterer
from clustering_api.src.adapters.denstream_clusterer import DenStreamClusterer
from clustering_api.src.config import config
from clustering_api.src.models.data_models import Cluster


class DenStreamService:
    """Service orchestrating streaming updates for DenStream."""

    DEFAULT_CONFIG = asdict(config.denstream)

    def __init__(
        self,
        clusterer: BaseClusterer | None = None,
        clusterer_factory: Callable[..., BaseClusterer] | None = None,
        **config,
    ):
        self._config = {**self.DEFAULT_CONFIG, **config}
        self._factory = clusterer_factory or (lambda **cfg: DenStreamClusterer(**cfg))
        self.clusterer = clusterer or self._factory(**self._config)
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
            return self.get_current_clusters()
        self.clusterer.update(batch_list)
        return self._refresh_cache()

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


denstream_service = DenStreamService()
