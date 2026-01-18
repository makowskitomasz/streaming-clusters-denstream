import pytest

from clustering_api.src.models.data_models import (
    Cluster,
    ClusterPoint,
    ClusterSummary,
)


def test_cluster_point_weight_validation() -> None:
    # Arrange
    payload = {"x": 0.0, "y": 0.0, "weight": 0}
    # Act
    with pytest.raises(ValueError, match="weight must be positive"):
        # Assert
        ClusterPoint(**payload)


def test_cluster_size_vs_points_validation() -> None:
    # Arrange
    point = ClusterPoint(x=1.0, y=2.0)
    # Act
    with pytest.raises(ValueError, match="Cluster size cannot be smaller than number of points"):
        # Assert
        Cluster(id="c1", centroid=(0.0, 0.0), size=0, density=0.5, points=[point])


def test_cluster_status_flag() -> None:
    # Arrange
    cluster = Cluster(id="c2", centroid=(1.0, 1.0), size=5, density=0.8, status="decayed")
    # Act
    status = cluster.status
    # Assert
    assert status == "decayed"


def test_cluster_summary_from_clusters() -> None:
    # Arrange
    c1 = Cluster(id="1", centroid=(0.0, 0.0), size=10, density=0.4)
    c2 = Cluster(id="2", centroid=(1.0, 1.0), size=5, density=0.6)
    # Act
    summary = ClusterSummary.from_clusters([c1, c2], noise_points=5)
    # Assert
    assert summary.total_clusters == 2
    assert summary.total_points == 15
    assert summary.avg_density == pytest.approx(0.5)
    assert summary.noise_ratio == pytest.approx(5 / 20)
