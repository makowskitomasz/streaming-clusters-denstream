from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict, Iterable, List, Optional

from clustering_api.src.adapters.base_clusterer import BaseClusterer
from clustering_api.src.adapters.denstream_clusterer import DenStreamClusterer
from clustering_api.src.config import config
from clustering_api.src.models.data_models import Cluster


class DenStreamService:
    """Service orchestrating streaming updates for DenStream."""

    DEFAULT_CONFIG = asdict(config.denstream)

    def __init__(
        self,
        clusterer: Optional[BaseClusterer] = None,
        clusterer_factory: Optional[Callable[..., BaseClusterer]] = None,
        **config,
    ):
        self._config = {**self.DEFAULT_CONFIG, **config}
        self._factory = clusterer_factory or self._default_factory
        self._clusterer = clusterer or self._factory(**self._config)
        self._active_clusters: List[Cluster] = []
        self._decayed_clusters: List[Cluster] = []

    @property
    def clusterer(self) -> BaseClusterer:
        return self._clusterer

    def _default_factory(self, **config: Any) -> BaseClusterer:
        return DenStreamClusterer(**config)

    def _refresh_cache(self) -> Dict[str, List[Cluster]]:
        clusters = self._clusterer.get_clusters()
        self._active_clusters = clusters.get("active", [])
        self._decayed_clusters = clusters.get("decayed", [])
        return self.get_current_clusters()

    def update_clusters(self, batch: Iterable[Any]) -> Dict[str, List[Cluster]]:
        batch_list = list(batch)
        if not batch_list:
            return self.get_current_clusters()
        self._clusterer.update(batch_list)
        return self._refresh_cache()

    def get_current_clusters(self) -> Dict[str, List[Cluster]]:
        return {
            "active_clusters": list(self._active_clusters),
            "decayed_clusters": list(self._decayed_clusters),
        }

    def configure(self, **config) -> Dict[str, float]:
        updated = False
        for key, value in config.items():
            if value is not None and key in self._config:
                self._config[key] = value
                updated = True
        if updated:
            self._clusterer = self._factory(**self._config)
            self._active_clusters = []
            self._decayed_clusters = []
        return dict(self._config)

    def get_config(self) -> Dict[str, float]:
        return dict(self._config)


denstream_service = DenStreamService()
