import plotly.graph_objects as go  # type: ignore[import-untyped]
from frontend.plotting import build_cluster_scatter


def test_build_cluster_scatter_creates_figure():
    points = [(0.0, 0.0), (1.0, 1.0)]
    labels = [0, 0]
    fig = build_cluster_scatter(points, labels, {0: (0.5, 0.5)})
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2


def test_noise_trace_has_low_opacity():
    points = [(0.0, 0.0), (1.0, 1.0)]
    labels = [-1, 0]
    fig = build_cluster_scatter(points, labels)
    noise_traces = [trace for trace in fig.data if trace.name == "Noise"]
    assert noise_traces
    assert noise_traces[0].marker.opacity <= 0.4


def test_centroid_trace_has_labels():
    points = [(0.0, 0.0), (1.0, 1.0)]
    labels = [0, 0]
    fig = build_cluster_scatter(points, labels, {0: (0.5, 0.5)})
    centroid_traces = [trace for trace in fig.data if trace.name == "C0"]
    assert centroid_traces
    assert centroid_traces[0].text[0] == "C0"


def test_color_mapping_is_stable():
    points = [(0.0, 0.0), (1.0, 1.0)]
    labels = [2, 2]
    fig_one = build_cluster_scatter(points, labels)
    fig_two = build_cluster_scatter(points, labels)
    color_one = fig_one.data[0].marker.color
    color_two = fig_two.data[0].marker.color
    assert color_one == color_two
