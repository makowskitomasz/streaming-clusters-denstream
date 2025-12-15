from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from river import cluster as river_cluster

from clustering_api.src.adapters.base_clusterer import BaseClusterer
from clustering_api.src.models.data_models import Cluster, ClusterPoint, DataPoint

RawPoint = ClusterPoint | DataPoint | dict[str, Any] | Sequence[float]


class DenStreamClusterer(BaseClusterer):
    """Adapter that wraps river's DenStream implementation."""

    def __init__(
        self,
        decay_factor: float = 0.01,
        epsilon: float = 0.5,
        beta: float = 0.5,
        mu: float = 2.5,
        n_samples_init: int = 200,
        stream_speed: int = 50,
    ):
        self.config = {
            "decay_factor": decay_factor,
            "epsilon": epsilon,
            "beta": beta,
            "mu": mu,
            "n_samples_init": n_samples_init,
            "stream_speed": stream_speed,
        }
        self._model = self._create_model()

    def _create_model(self):
        return river_cluster.DenStream(
            decaying_factor=self.config["decay_factor"],
            epsilon=self.config["epsilon"],
            beta=self.config["beta"],
            mu=self.config["mu"],
            n_samples_init=self.config["n_samples_init"],
            stream_speed=self.config["stream_speed"],
        )

    def _iter_features(self, data: Iterable[RawPoint]) -> Iterable[dict[str, float]]:
        for item in data:
            if isinstance(item, ClusterPoint) or isinstance(item, DataPoint):
                yield {"x": float(item.x), "y": float(item.y)}
            elif isinstance(item, dict):
                yield {"x": float(item["x"]), "y": float(item["y"])}
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                yield {"x": float(item[0]), "y": float(item[1])}
            else:
                raise TypeError(f"Unsupported data type for DenStreamClusterer: {type(item)}")

    def _micro_clusters_to_cluster(self, clusters_dict, status: str) -> list[Cluster]:
        timestamp = getattr(self._model, "timestamp", 0)
        clusters: list[Cluster] = []
        for idx, micro_cluster in clusters_dict.items():
            center = micro_cluster.calc_center(timestamp)
            centroid = (
                float(center.get("x", center.get(0, 0.0))),
                float(center.get("y", center.get(1, 0.0))),
            )
            density = float(micro_cluster.calc_weight(timestamp))
            clusters.append(
                Cluster(
                    id=f"{status}-{idx}",
                    centroid=centroid,
                    size=int(micro_cluster.N),
                    density=density,
                    status=status,
                ),
            )
        return clusters

    def fit(self, data: Iterable[RawPoint]) -> None:
        self._model = self._create_model()
        self.update(data)

    def update(self, data: Iterable[RawPoint]) -> None:
        for features in self._iter_features(data):
            self._model.learn_one(features)

    def get_clusters(self) -> dict[str, list[Cluster]]:
        active = self._micro_clusters_to_cluster(self._model.p_micro_clusters, status="active")
        decayed = self._micro_clusters_to_cluster(self._model.o_micro_clusters, status="decayed")
        return {"active": active, "decayed": decayed}
