from typing import Any, Dict, Iterable, List, Sequence, Union

from river import cluster as river_cluster

from clustering_api.src.adapters.base_clusterer import BaseClusterer
from clustering_api.src.models.data_models import Cluster, ClusterPoint, DataPoint

RawPoint = Union[ClusterPoint, DataPoint, Dict[str, Any], Sequence[float]]
FeatureVector = Dict[str, float]
ConfigValue = Union[int, float]


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
        self._config: Dict[str, ConfigValue] = {
            "decay_factor": decay_factor,
            "epsilon": epsilon,
            "beta": beta,
            "mu": mu,
            "n_samples_init": n_samples_init,
            "stream_speed": stream_speed,
        }
        self._model: river_cluster.DenStream = self._create_model()

    @property
    def config(self) -> Dict[str, float]:
        """Return a copy of the clusterer configuration."""
        return dict(self._config)

    def _create_model(self) -> river_cluster.DenStream:
        return river_cluster.DenStream(
            decaying_factor=self._config["decay_factor"],
            epsilon=self._config["epsilon"],
            beta=self._config["beta"],
            mu=self._config["mu"],
            n_samples_init=int(self._config["n_samples_init"]),
            stream_speed=int(self._config["stream_speed"]),
        )

    def _iter_features(self, data: Iterable[RawPoint]) -> Iterable[FeatureVector]:
        for item in data:
            yield self._point_to_features(item)

    def _point_to_features(self, item: RawPoint) -> FeatureVector:
        if isinstance(item, (ClusterPoint, DataPoint)):
            return {"x": float(item.x), "y": float(item.y)}
        if isinstance(item, dict):
            return {"x": float(item["x"]), "y": float(item["y"])}
        if isinstance(item, (list, tuple)) and len(item) == 2:
            return {"x": float(item[0]), "y": float(item[1])}
        raise TypeError(f"Unsupported data type for DenStreamClusterer: {type(item)}")

    def _micro_clusters_to_cluster(
        self, clusters_dict: Dict[Any, Any], status: str
    ) -> List[Cluster]:
        timestamp = getattr(self._model, "timestamp", 0)
        clusters: List[Cluster] = []
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
                )
            )
        return clusters

    def _micro_cluster_centroid(
        self, micro_cluster: Any, timestamp: float
    ) -> tuple[float, float]:
        center = micro_cluster.calc_center(timestamp)
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

    def get_clusters(self) -> Dict[str, List[Cluster]]:
        active = self._micro_clusters_to_cluster(
            self._model.p_micro_clusters, status="active"
        )
        decayed = self._micro_clusters_to_cluster(
            self._model.o_micro_clusters, status="decayed"
        )
        return {"active": active, "decayed": decayed}
