import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotting import build_cluster_scatter


def test_build_cluster_scatter_creates_figure() -> None:
    # Arrange
    points = [(0.0, 0.0), (1.0, 1.0)]
    labels = [0, 0]
    # Act
    fig = build_cluster_scatter(points, labels, {0: (0.5, 0.5)})
    # Assert
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2


def test_noise_trace_has_low_opacity() -> None:
    # Arrange
    maximum_opacity = 0.4

    points = [(0.0, 0.0), (1.0, 1.0)]
    labels = [-1, 0]
    # Act
    fig = build_cluster_scatter(points, labels)
    noise_traces = [trace for trace in fig.data if trace.name == "Noise"]
    # Assert
    assert noise_traces
    assert noise_traces[0].marker.opacity <= maximum_opacity


def test_centroid_trace_has_labels() -> None:
    # Arrange
    points = [(0.0, 0.0), (1.0, 1.0)]
    labels = [0, 0]
    # Act
    fig = build_cluster_scatter(points, labels, {0: (0.5, 0.5)})
    centroid_traces = [trace for trace in fig.data if trace.name == "C0"]
    # Assert
    assert centroid_traces
    assert centroid_traces[0].text[0] == "C0"


def test_color_mapping_is_stable() -> None:
    # Arrange
    points = [(0.0, 0.0), (1.0, 1.0)]
    labels = [2, 2]
    # Act
    fig_one = build_cluster_scatter(points, labels)
    fig_two = build_cluster_scatter(points, labels)
    color_one = fig_one.data[0].marker.color
    color_two = fig_two.data[0].marker.color
    # Assert
    assert color_one == color_two
