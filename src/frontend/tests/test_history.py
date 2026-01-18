from collections import deque

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
from history import CentroidSnapshot, append_history, compute_centroids_from_points
from plotting import build_centroid_trajectories


def test_append_history_trims() -> None:
    # Arrange
    history = deque(maxlen=3)
    # Act
    for idx in range(5):
        snapshot = CentroidSnapshot(
            batch_id=idx,
            timestamp="t",
            centroids={0: (float(idx), 0.0)},
        )
        append_history(history, snapshot)
    # Assert
    assert len(history) == 3
    assert next(iter(history)).batch_id == 2


def test_compute_centroids_excludes_noise() -> None:
    # Arrange
    points = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0]])
    labels = np.array([0, 0, -1])
    # Act
    centroids = compute_centroids_from_points(points, labels)
    # Assert
    assert centroids == {0: (0.5, 0.5)}


def test_build_centroid_trajectories_returns_figure() -> None:
    # Arrange
    history = [
        CentroidSnapshot(batch_id=0, timestamp="t0", centroids={0: (0.0, 0.0)}),
        CentroidSnapshot(batch_id=1, timestamp="t1", centroids={0: (1.0, 1.0)}),
    ]
    # Act
    fig = build_centroid_trajectories(history, show_labels=True)
    # Assert
    assert isinstance(fig, go.Figure)
