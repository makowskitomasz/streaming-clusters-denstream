from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

from clustering_api.src.adapters.base_clusterer import BaseClusterer
from clustering_api.src.adapters.denstream_clusterer import DenStreamClusterer
from clustering_api.src.models.data_models import Cluster


class DenStreamService:
    """Service orchestrating streaming updates for DenStream."""

    DEFAULT_CONFIG = {
        "decay_factor": 0.01,
        "epsilon": 0.5,
        "beta": 0.5,
        "mu": 2.5,
        "n_samples_init": 200,
        "stream_speed": 50,
    }

    def __init__(
        self,
        clusterer: Optional[BaseClusterer] = None,
        clusterer_factory: Optional[Callable[..., BaseClusterer]] = None,
        **config,
    ):
        self._config = {**self.DEFAULT_CONFIG, **config}
        self._factory = clusterer_factory or (lambda **cfg: DenStreamClusterer(**cfg))
        self.clusterer = clusterer or self._factory(**self._config)
        self._active_clusters: List[Cluster] = []
        self._decayed_clusters: List[Cluster] = []

    def _refresh_cache(self) -> Dict[str, List[Cluster]]:
        clusters = self.clusterer.get_clusters()
        self._active_clusters = clusters.get("active", [])
        self._decayed_clusters = clusters.get("decayed", [])
        return self.get_current_clusters()

    def update_clusters(self, batch: Iterable[Any]) -> Dict[str, List[Cluster]]:
        batch_list = list(batch)
        if not batch_list:
            return self.get_current_clusters()
        self.clusterer.update(batch_list)
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
            self.clusterer = self._factory(**self._config)
            self._active_clusters = []
            self._decayed_clusters = []
        return dict(self._config)

    def get_config(self) -> Dict[str, float]:
        return dict(self._config)


denstream_service = DenStreamService()
