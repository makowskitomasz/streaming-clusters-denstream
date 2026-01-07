import numpy as np

from clustering_api.src.models.data_models import Cluster
from clustering_api.src.services.denstream_service import DenStreamService
from clustering_api.src.services.metrics_service import MetricsService
from clustering_api.src.services.stream_service import StreamService
from clustering_api.src.utils.drift_tracker import DriftTracker


def _active_clusters(service: DenStreamService) -> list[Cluster]:
    return service.get_current_clusters()["active_clusters"]


def _centroid_map(clusters: list[Cluster]) -> dict[int, np.ndarray]:
    mapping: dict[int, np.ndarray] = {}
    fallback_ids = sorted(cluster.id for cluster in clusters)
    fallback_map = {cid: idx for idx, cid in enumerate(fallback_ids)}
    for cluster in clusters:
        cluster_id = cluster.id
        parsed = None
        if "-" in cluster_id:
            suffix = cluster_id.split("-", maxsplit=1)[-1]
            if suffix.isdigit():
                parsed = int(suffix)
        mapped_id = parsed if parsed is not None else fallback_map[cluster_id]
        mapping[mapped_id] = np.array(cluster.centroid, dtype=float)
    return mapping


def test_stable_distribution_keeps_cluster_count():
    # Arrange
    rng = np.random.default_rng(42)
    stream = StreamService()
    metrics = MetricsService()
    service = DenStreamService(metrics=metrics)
    centroids = np.array([[-4.0, 0.0], [0.0, 4.0], [4.0, 0.0]])
    counts = []

    # Act
    for batch_id in range(8):
        batch = stream.generate_custom_batch(
            centroids=centroids,
            points_per_cluster=60,
            noise_ratio=0.02,
            rng=rng,
            batch_id=batch_id,
        )
        service.update_clusters(batch)
        counts.append(len(_active_clusters(service)))

    # Assert
    steady_counts = counts[3:]
    median = float(np.median(steady_counts))
    assert all(count > 0 for count in steady_counts)
    assert all(abs(count - median) <= 3 for count in steady_counts)


def test_gradual_drift_increases_mean_drift_distance():
    # Arrange
    rng = np.random.default_rng(1)
    stream = StreamService()
    metrics = MetricsService()
    service = DenStreamService(metrics=metrics)
    tracker = DriftTracker()
    centroids = np.array([[-3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    mean_drifts = []

    # Act
    for step in range(6):
        shifted = centroids + np.array([0.2 * step, 0.1 * step])
        batch = stream.generate_custom_batch(
            centroids=shifted,
            points_per_cluster=50,
            noise_ratio=0.03,
            rng=rng,
            batch_id=step,
        )
        service.update_clusters(batch)
        clusters = _active_clusters(service)
        drift_update = tracker.update(_centroid_map(clusters))
        if drift_update.mean_drift_distance is not None:
            mean_drifts.append(drift_update.mean_drift_distance)

    # Assert
    assert len(mean_drifts) >= 3
    assert max(mean_drifts) > min(mean_drifts)


def test_cluster_appearance_and_disappearance():
    # Arrange
    rng = np.random.default_rng(7)
    stream = StreamService()
    metrics = MetricsService()
    service = DenStreamService(metrics=metrics)
    base = np.array([[-3.0, 0.0], [3.0, 0.0]])
    added = np.array([0.0, 4.0])
    counts = []

    # Act
    for step in range(9):
        if step < 3:
            centroids = base
        elif step < 6:
            centroids = np.vstack([base, added])
        else:
            centroids = np.array([base[0], added])
        batch = stream.generate_custom_batch(
            centroids=centroids,
            points_per_cluster=60,
            noise_ratio=0.02,
            rng=rng,
            batch_id=step,
        )
        service.update_clusters(batch)
        counts.append(len(_active_clusters(service)))

    # Assert
    first_avg = float(np.mean(counts[:3]))
    middle_avg = float(np.mean(counts[3:6]))
    last_avg = float(np.mean(counts[6:]))
    assert middle_avg > first_avg
    assert last_avg <= middle_avg + 1


def test_increasing_noise_ratio_does_not_explode_clusters():
    # Arrange
    rng = np.random.default_rng(9)
    stream = StreamService()
    metrics = MetricsService()
    service = DenStreamService(metrics=metrics)
    centroids = np.array([[-4.0, -2.0], [4.0, 2.0]])
    noise_ratios = [0.0, 0.05, 0.1, 0.2, 0.3]
    counts = []

    # Act
    for step, noise_ratio in enumerate(noise_ratios):
        batch = stream.generate_custom_batch(
            centroids=centroids,
            points_per_cluster=60,
            noise_ratio=noise_ratio,
            rng=rng,
            batch_id=step,
        )
        service.update_clusters(batch)
        counts.append(len(_active_clusters(service)))

    # Assert
    baseline_max = max(counts[:2])
    assert max(counts) <= baseline_max + 6
    assert noise_ratios == sorted(noise_ratios)
