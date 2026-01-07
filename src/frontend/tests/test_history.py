import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
from frontend.history import CentroidSnapshot, append_history, compute_centroids_from_points
from frontend.plotting import build_centroid_trajectories


def test_append_history_trims() -> None:
    history = []
    for idx in range(5):
        snapshot = CentroidSnapshot(
            batch_id=idx,
            timestamp="t",
            centroids={0: (float(idx), 0.0)},
        )
        history = append_history(history, snapshot, max_len=3)
    assert len(history) == 3
    assert history[0].batch_id == 2


def test_compute_centroids_excludes_noise() -> None:
    points = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0]])
    labels = np.array([0, 0, -1])
    centroids = compute_centroids_from_points(points, labels)
    assert centroids == {0: (0.5, 0.5)}


def test_build_centroid_trajectories_returns_figure() -> None:
    history = [{0: (0.0, 0.0)}, {0: (1.0, 1.0)}]
    fig = build_centroid_trajectories(history, show_labels=True)
    assert isinstance(fig, go.Figure)
