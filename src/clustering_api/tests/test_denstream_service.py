from clustering_api.src.adapters.base_clusterer import BaseClusterer
from clustering_api.src.models.data_models import Cluster, ClusterPoint
from clustering_api.src.services.denstream_service import DenStreamService
from clustering_api.src.services.metrics_service import MetricsService


class DummyClusterer(BaseClusterer):
    def __init__(self):
        self.updated_batches = []
        self.clusters_output = {"active": [], "decayed": []}

    def fit(self, data):
        self.updated_batches.append(list(data))

    def update(self, data):
        self.updated_batches.append(list(data))

    def get_clusters(self):
        return self.clusters_output


def test_update_clusters_refreshes_cache():
    dummy = DummyClusterer()
    dummy.clusters_output = {
        "active": [Cluster(id="a-1", centroid=(0.0, 0.0), size=3, density=0.5)],
        "decayed": [Cluster(id="d-1", centroid=(1.0, 1.0), size=1, density=0.1, status="decayed")],
    }
    metrics = MetricsService()
    service = DenStreamService(clusterer=dummy, metrics=metrics)

    payload = [ClusterPoint(x=0.1, y=0.2)]
    response = service.update_clusters(payload)

    assert len(dummy.updated_batches) == 1
    assert response["active_clusters"][0].id == "a-1"
    assert response["decayed_clusters"][0].status == "decayed"


def test_configure_rebuilds_clusterer():
    created_clusterers = []

    def factory(**cfg):
        dummy = DummyClusterer()
        dummy.config = cfg
        created_clusterers.append(dummy)
        return dummy

    metrics = MetricsService()
    service = DenStreamService(clusterer_factory=factory, metrics=metrics)
    first_clusterer = service.clusterer

    updated_config = service.configure(decay_factor=0.2)

    assert service.clusterer is not first_clusterer
    assert created_clusterers[-1].config["decay_factor"] == 0.2
    assert updated_config["decay_factor"] == 0.2


def test_metrics_stored_after_denstream_update():
    dummy = DummyClusterer()
    dummy.clusters_output = {
        "active": [Cluster(id="a-1", centroid=(0.0, 0.0), size=3, density=0.5)],
        "decayed": [],
    }
    metrics = MetricsService()
    service = DenStreamService(clusterer=dummy, metrics=metrics)

    payload = [ClusterPoint(x=0.1, y=0.2, batch_id=7)]
    service.update_clusters(payload)
    latest = metrics.get_latest("denstream")

    assert latest is not None
    assert latest.batch_id == 7
    assert latest.n_samples == 1
