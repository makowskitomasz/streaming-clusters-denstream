from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from river import cluster as river_cluster

from clustering_api.src.adapters.base_clusterer import BaseClusterer
from clustering_api.src.models.data_models import Cluster, ClusterPoint, DataPoint

RawPoint = ClusterPoint | DataPoint | dict[str, Any] | Sequence[float]
FeatureVector = dict[str, float]
ConfigValue = int | float
DIMENSIONS = 2


@dataclass(frozen=True, slots=True)
class DenStreamConfig:
    decay_factor: float
    epsilon: float
    beta: float
    mu: float
    n_samples_init: int
    stream_speed: int

    @classmethod
    def from_dict(cls, payload: dict[str, ConfigValue]) -> DenStreamConfig:
        return cls(
            decay_factor=float(payload["decay_factor"]),
            epsilon=float(payload["epsilon"]),
            beta=float(payload["beta"]),
            mu=float(payload["mu"]),
            n_samples_init=int(payload["n_samples_init"]),
            stream_speed=int(payload["stream_speed"]),
        )

    def to_dict(self) -> dict[str, ConfigValue]:
        return {
            "decay_factor": self.decay_factor,
            "epsilon": self.epsilon,
            "beta": self.beta,
            "mu": self.mu,
            "n_samples_init": self.n_samples_init,
            "stream_speed": self.stream_speed,
        }


class DenStreamClusterer(BaseClusterer):
    """Adapter wrapping River's DenStream implementation."""

    def __init__(
        self,
        decay_factor: float = 0.01,
        epsilon: float = 0.5,
        beta: float = 0.5,
        mu: float = 2.5,
        n_samples_init: int = 200,
        stream_speed: int = 50,
    ) -> None:
        self._config = DenStreamConfig(
            decay_factor=decay_factor,
            epsilon=epsilon,
            beta=beta,
            mu=mu,
            n_samples_init=n_samples_init,
            stream_speed=stream_speed,
        )
        self._model: river_cluster.DenStream = self._create_model()

    @property
    def config(self) -> dict[str, float]:
        """Return a copy of the clusterer configuration."""
        return dict(self._config.to_dict())

    def _create_model(self) -> river_cluster.DenStream:
        cfg = self._config
        return river_cluster.DenStream(
            decaying_factor=cfg.decay_factor,
            epsilon=cfg.epsilon,
            beta=cfg.beta,
            mu=cfg.mu,
            n_samples_init=cfg.n_samples_init,
            stream_speed=cfg.stream_speed,
        )

    def _iter_features(self, data: Iterable[RawPoint]) -> Iterable[FeatureVector]:
        for item in data:
            yield self._point_to_features(item)

    def _point_to_features(self, item: RawPoint) -> FeatureVector:
        match item:
            case ClusterPoint() | DataPoint():
                return {"x": float(item.x), "y": float(item.y)}
            case {"x": x, "y": y}:
                return {"x": float(x), "y": float(y)}
            case (x, y) if len(item) == DIMENSIONS:
                return {"x": float(x), "y": float(y)}
            case _:
                msg = f"Unsupported data type for DenStreamClusterer: {type(item)}"
                raise TypeError(msg)

    def _micro_clusters_to_cluster(
        self,
        clusters_dict: dict[Any, Any],
        status: str,
    ) -> list[Cluster]:
        timestamp = getattr(self._model, "timestamp", 0)
        clusters: list[Cluster] = []
        for idx, micro_cluster in clusters_dict.items():
            centroid = self._micro_cluster_centroid(micro_cluster, timestamp)
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

    def _micro_cluster_centroid(
        self,
        micro_cluster: object,
        timestamp: float,
    ) -> tuple[float, float]:
        cluster = cast("Any", micro_cluster)
        center = cluster.calc_center(timestamp)
        return (
            float(center.get("x", center.get(0, 0.0))),
            float(center.get("y", center.get(1, 0.0))),
        )

    def fit(self, data: Iterable[RawPoint]) -> None:
        self._model = self._create_model()
        self.update(data)

    def update(self, data: Iterable[RawPoint]) -> None:
        for features in self._iter_features(data):
            self._model.learn_one(features)

    def get_clusters(self) -> dict[str, list[Cluster]]:
        active = self._micro_clusters_to_cluster(
            self._model.p_micro_clusters,
            status="active",
        )
        decayed = self._micro_clusters_to_cluster(
            self._model.o_micro_clusters,
            status="decayed",
        )
        return {"active": active, "decayed": decayed}
