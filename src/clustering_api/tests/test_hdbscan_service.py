import numpy as np
import pytest
from loguru import logger
from sklearn.datasets import make_blobs

from clustering_api.src.services.hdbscan_service import HdbscanService
from clustering_api.src.services.metrics_service import MetricsService


def test_cluster_batch_returns_labels_length():
    # Arrange
    features, _ = make_blobs(
        n_samples=60, centers=2, cluster_std=0.3, random_state=7,
    )
    metrics = MetricsService()
    service = HdbscanService(min_cluster_size=5, random_state=7, metrics=metrics)

    # Act
    result = service.cluster_batch(features, batch_id="batch-1")

    # Assert
    assert len(result.labels) == len(features)


def test_metrics_exclude_noise():
    # Arrange
    features, _ = make_blobs(
        n_samples=80, centers=2, cluster_std=0.35, random_state=42,
    )
    rng = np.random.default_rng(42)
    noise = rng.uniform(-6, 6, size=(12, 2))
    batch = np.vstack([features, noise])
    metrics = MetricsService()
    service = HdbscanService(min_cluster_size=5, random_state=42, metrics=metrics)

    # Act
    result = service.cluster_batch(batch, batch_id="batch-2")

    # Assert
    labels = np.array(result.labels)
    unique_clusters = {label for label in labels if label != -1}
    expected_clusters = len(unique_clusters)
    expected_noise_ratio = float(np.sum(labels == -1)) / len(labels)
    assert result.number_of_clusters == expected_clusters
    assert result.noise_ratio == pytest.approx(expected_noise_ratio)


def test_silhouette_none_with_single_cluster(monkeypatch):
    # Arrange
    features, _ = make_blobs(
        n_samples=40, centers=1, cluster_std=0.2, random_state=1,
    )
    metrics = MetricsService()
    service = HdbscanService(min_cluster_size=5, random_state=1, metrics=metrics)
    forced_labels = np.zeros(len(features), dtype=int)

    def fake_fit_predict(_clusterer, _data):
        return forced_labels

    monkeypatch.setattr(service, "_fit_predict", fake_fit_predict)

    # Act
    result = service.cluster_batch(features)

    # Assert
    assert result.number_of_clusters == 1
    assert result.silhouette_score is None


def test_silhouette_none_all_noise():
    # Arrange
    rng = np.random.default_rng(3)
    features = rng.normal(0, 1, size=(5, 2))
    metrics = MetricsService()
    service = HdbscanService(min_cluster_size=10, random_state=3, metrics=metrics)

    # Act
    result = service.cluster_batch(features)

    # Assert
    assert result.number_of_clusters == 0
    assert result.silhouette_score is None
    assert result.noise_ratio == 1.0


def test_empty_batch_handled_gracefully():
    # Arrange
    features = np.empty((0, 2))
    metrics = MetricsService()
    service = HdbscanService(metrics=metrics)

    # Act
    result = service.cluster_batch(features)

    # Assert
    assert result.labels == []
    assert result.number_of_clusters == 0
    assert result.noise_ratio == 0.0
    assert result.silhouette_score is None
    assert result.cluster_size_summary is None


def test_history_size_cap():
    # Arrange
    features, _ = make_blobs(
        n_samples=20, centers=2, cluster_std=0.2, random_state=5,
    )
    metrics = MetricsService()
    service = HdbscanService(history_size=2, random_state=5, metrics=metrics)

    # Act
    service.cluster_batch(features, batch_id="batch-a")
    service.cluster_batch(features, batch_id="batch-b")
    service.cluster_batch(features, batch_id="batch-c")
    history = service.get_history()

    # Assert
    assert isinstance(history, tuple)
    assert len(history) == 2
    assert history[0].batch_id == "batch-b"
    assert history[1].batch_id == "batch-c"


def test_metrics_stored_after_cluster_batch():
    # Arrange
    features, _ = make_blobs(
        n_samples=30, centers=2, cluster_std=0.4, random_state=11,
    )
    metrics = MetricsService()
    service = HdbscanService(min_cluster_size=5, random_state=11, metrics=metrics)

    # Act
    result = service.cluster_batch(features, batch_id="batch-metrics")
    latest = metrics.get_latest("hdbscan")

    # Assert
    assert latest is not None
    assert latest.batch_id == "batch-metrics"
    assert latest.n_samples == len(result.labels)
    assert latest.number_of_clusters == result.number_of_clusters


def test_hdbscan_logs_batch_stats(monkeypatch):
    # Arrange
    features = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
    metrics = MetricsService()
    service = HdbscanService(min_cluster_size=2, metrics=metrics)
    forced_labels = np.array([0, 0, -1])

    def fake_fit_predict(_clusterer, _data):
        return forced_labels

    captured = {}

    def fake_bind(**kwargs):
        captured.update(kwargs)

        class _Logger:
            def info(self, _msg):
                return None

        return _Logger()

    monkeypatch.setattr(service, "_fit_predict", fake_fit_predict)
    monkeypatch.setattr(logger, "bind", fake_bind)

    # Act
    service.cluster_batch(features, batch_id="hdb-1")

    # Assert
    assert captured["event"] == "clustering_batch"
    assert captured["model_name"] == "hdbscan"
    assert captured["n_samples"] == 3
    assert captured["active_clusters"] == 1
    assert captured["noise_ratio"] == pytest.approx(1 / 3)
    assert "latency_ms" in captured
