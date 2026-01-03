import numpy as np
import pytest

from clustering_api.src.utils.drift_tracker import DriftTracker


def test_first_update_marks_appeared_only():
    # Arrange
    tracker = DriftTracker()
    centroids = {0: np.array([0.0, 0.0]), 1: np.array([1.0, 1.0])}

    # Act
    update = tracker.update(centroids)

    # Assert
    assert update.stable == []
    assert sorted(update.appeared) == [0, 1]
    assert update.disappeared == []
    assert update.per_cluster == {}


def test_displacement_distance_direction():
    # Arrange
    tracker = DriftTracker()
    tracker.update({0: np.array([0.0, 0.0])})

    # Act
    update = tracker.update({0: np.array([3.0, 4.0])})

    # Assert
    drift = update.per_cluster[0]
    assert drift.distance == pytest.approx(5.0)
    assert drift.direction is not None
    assert drift.direction.tolist() == pytest.approx([0.6, 0.8])


def test_speed_uses_timestamp_dt():
    # Arrange
    tracker = DriftTracker()
    tracker.update({0: np.array([0.0, 0.0])}, timestamp=1.0)

    # Act
    update = tracker.update({0: np.array([2.0, 0.0])}, timestamp=3.0)

    # Assert
    drift = update.per_cluster[0]
    assert update.dt == 2.0
    assert drift.speed == pytest.approx(1.0)


def test_appear_disappear_detection():
    # Arrange
    tracker = DriftTracker()
    tracker.update({0: np.array([0.0, 0.0]), 1: np.array([1.0, 1.0])})

    # Act
    update = tracker.update({1: np.array([1.0, 2.0]), 2: np.array([2.0, 2.0])})

    # Assert
    assert update.appeared == [2]
    assert update.disappeared == [0]
    assert update.stable == [1]


def test_zero_displacement_has_no_direction():
    # Arrange
    tracker = DriftTracker()
    tracker.update({0: np.array([1.0, 1.0])})

    # Act
    update = tracker.update({0: np.array([1.0, 1.0])})

    # Assert
    drift = update.per_cluster[0]
    assert drift.distance == 0.0
    assert drift.direction is None


def test_ema_distance_smoothing():
    # Arrange
    tracker = DriftTracker(ema_alpha=0.5, smooth_direction=False)
    tracker.update({0: np.array([0.0, 0.0])})
    tracker.update({0: np.array([2.0, 0.0])})

    # Act
    update = tracker.update({0: np.array([6.0, 0.0])})

    # Assert
    drift = update.per_cluster[0]
    assert drift.distance == pytest.approx(4.0)
    assert drift.ema_distance == pytest.approx(3.0)


def test_invalid_timestamp_order_raises():
    # Arrange
    tracker = DriftTracker()
    tracker.update({0: np.array([0.0, 0.0])}, timestamp=2.0)

    # Act / Assert
    with pytest.raises(ValueError):
        tracker.update({0: np.array([1.0, 0.0])}, timestamp=1.0)


def test_nan_centroid_raises():
    # Arrange
    tracker = DriftTracker()
    centroids = {0: np.array([np.nan, 0.0])}

    # Act / Assert
    with pytest.raises(ValueError):
        tracker.update(centroids)
