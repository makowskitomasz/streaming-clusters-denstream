import numpy as np
from fastapi.testclient import TestClient

from clustering_api.src.app import create_app
from clustering_api.src.services.metrics_service import MetricsService, metrics_service


def test_metrics_service_handles_empty_batch():
    # Arrange
    service = MetricsService()
    features = np.empty((0, 2))
    labels = np.array([], dtype=int)

    # Act
    record = service.evaluate(features, labels, model_name="denstream", batch_id=None)

    # Assert
    assert record.n_samples == 0
    assert record.number_of_clusters == 0
    assert record.noise_ratio == 0.0
    assert record.silhouette_score is None


def test_metrics_service_handles_all_noise():
    # Arrange
    service = MetricsService()
    features = np.random.default_rng(1).normal(size=(5, 2))
    labels = np.full((5,), -1, dtype=int)

    # Act
    record = service.evaluate(features, labels, model_name="hdbscan", batch_id="n1")

    # Assert
    assert record.number_of_clusters == 0
    assert record.noise_ratio == 1.0
    assert record.silhouette_score is None


def test_metrics_latest_endpoint():
    # Arrange
    metrics_service.reset()
    client = TestClient(create_app())
    features = np.array([[0.0, 0.0], [1.0, 1.0]])
    labels = np.array([0, 0])
    metrics_service.evaluate(features, labels, model_name="denstream", batch_id="b1")

    # Act
    response = client.get("/v1/metrics/latest")

    # Assert
    assert response.status_code == 200
    payload = response.json()
    assert "latest" in payload
    assert payload["latest"]["denstream"]["batch_id"] == "b1"
