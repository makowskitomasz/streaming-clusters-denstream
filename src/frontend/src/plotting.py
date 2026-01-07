from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly import colors as plotly_colors

if TYPE_CHECKING:
    from api_client import LogRecord


def build_cluster_scatter(
    points: list[tuple[float, float]],
    labels: list[int],
    centroids: dict[int, tuple[float, float]] | None = None,
) -> go.Figure:
    """Build a Plotly scatter for clusters with optional centroids."""
    if len(points) != len(labels):
        msg = "points and labels must have matching lengths, " f"got {len(points)} and {len(labels)}"
        raise ValueError(msg)
    fig = go.Figure()
    if not points:
        fig.update_layout(
            title="Current clusters",
            xaxis_title="x",
            yaxis_title="y",
            uirevision="cluster-scatter",
        )
        return fig

    points_array = np.asarray(points, dtype=float)
    labels_array = np.asarray(labels, dtype=int)

    _add_cluster_traces(fig, points_array, labels_array)
    centroid_map = centroids or _compute_centroids(points_array, labels_array)
    if centroid_map:
        _add_centroid_traces(fig, centroid_map)

    fig.update_layout(
        title="Current clusters",
        xaxis_title="x",
        yaxis_title="y",
        legend={"orientation": "v"},
        uirevision="cluster-scatter",
        height=600,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    return fig


def build_centroid_trajectories(
    history: list[dict[int, tuple[float, float]]],
    *,
    show_labels: bool = True,
    only_last_n: int | None = None,
) -> go.Figure:
    """Build a 2D trajectory plot for centroid history."""
    fig = go.Figure()
    if not history:
        fig.update_layout(
            title="Centroid trajectories",
            xaxis_title="x",
            yaxis_title="y",
            uirevision="centroid-trajectories",
        )
        return fig

    snapshots = history[-only_last_n:] if only_last_n else history
    cluster_ids = sorted({cid for snap in snapshots for cid in snap})
    for cluster_id in cluster_ids:
        xs: list[float | None] = []
        ys: list[float | None] = []
        for snap in snapshots:
            if cluster_id in snap:
                x, y = snap[cluster_id]
                xs.append(x)
                ys.append(y)
            else:
                xs.append(None)
                ys.append(None)
        color = _color_for_label(cluster_id)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=f"Cluster {cluster_id}",
                line={"color": color},
                marker={"color": color, "size": 6},
            ),
        )
        if show_labels:
            last_point = next(
                (snap[cluster_id] for snap in reversed(snapshots) if cluster_id in snap),
                None,
            )
            if last_point is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[last_point[0]],
                        y=[last_point[1]],
                        mode="markers+text",
                        name=f"C{cluster_id}",
                        text=[f"C{cluster_id}"],
                        textposition="top center",
                        marker={"color": color, "size": 10, "symbol": "circle-open"},
                        showlegend=False,
                    ),
                )

    fig.update_layout(
        title="Centroid trajectories",
        xaxis_title="x",
        yaxis_title="y",
        legend={"orientation": "v"},
        uirevision="centroid-trajectories",
        height=500,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    return fig


def build_logs_timeline(logs: list[LogRecord], series: str) -> go.Figure:
    """Build a timeline chart for a selected log metric."""
    fig = go.Figure()
    if not logs:
        fig.update_layout(title="Performance timeline")
        return fig
    xs: list[object] = []
    ys: list[float] = []
    for idx, log in enumerate(logs):
        x_value = log.timestamp or log.batch_id or idx
        value = getattr(log, series, None)
        if value is None:
            continue
        xs.append(x_value)
        ys.append(value)
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name=series,
        ),
    )
    fig.update_layout(
        title="Performance timeline",
        xaxis_title="timestamp",
        yaxis_title=series,
        height=260,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    return fig


def _add_cluster_traces(
    fig: go.Figure,
    points: np.ndarray,
    labels: np.ndarray,
) -> None:
    unique_labels = sorted({int(label) for label in labels.tolist()})
    for label in unique_labels:
        mask = labels == label
        if label == -1:
            fig.add_trace(
                go.Scatter(
                    x=points[mask, 0],
                    y=points[mask, 1],
                    mode="markers",
                    name="Noise",
                    marker={"color": "gray", "size": 5, "opacity": 0.3},
                ),
            )
        else:
            color = _color_for_label(label)
            fig.add_trace(
                go.Scatter(
                    x=points[mask, 0],
                    y=points[mask, 1],
                    mode="markers",
                    name=f"Cluster {label}",
                    marker={"color": color, "size": 6},
                ),
            )


def _add_centroid_traces(
    fig: go.Figure,
    centroids: dict[int, tuple[float, float]],
) -> None:
    for cluster_id, centroid in centroids.items():
        color = _color_for_label(cluster_id)
        fig.add_trace(
            go.Scatter(
                x=[centroid[0]],
                y=[centroid[1]],
                mode="markers+text",
                name=f"C{cluster_id}",
                text=[f"C{cluster_id}"],
                textposition="top center",
                marker={"color": color, "size": 14, "symbol": "x"},
            ),
        )


def _compute_centroids(
    points: np.ndarray,
    labels: np.ndarray,
) -> dict[int, tuple[float, float]]:
    centroids: dict[int, tuple[float, float]] = {}
    for label in sorted({int(label) for label in labels.tolist()}):
        if label == -1:
            continue
        mask = labels == label
        if not np.any(mask):
            continue
        center = points[mask].mean(axis=0)
        centroids[label] = (float(center[0]), float(center[1]))
    return centroids


@lru_cache(maxsize=256)
def _color_for_label(label: int) -> str:
    palette = plotly_colors.qualitative.Set2
    return palette[label % len(palette)]
