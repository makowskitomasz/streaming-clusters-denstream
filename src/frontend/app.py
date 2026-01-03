from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import streamlit as st
from plotly import colors as plotly_colors


def _init_state() -> None:
    if "running" not in st.session_state:
        st.session_state.running = False
    if "batch_id" not in st.session_state:
        st.session_state.batch_id = 0
    if "seed" not in st.session_state:
        st.session_state.seed = 7
    if "rng" not in st.session_state:
        st.session_state.rng = np.random.default_rng(st.session_state.seed)
    if "base_centroids" not in st.session_state:
        st.session_state.base_centroids = np.array(
            [[-4.0, 0.0], [0.0, 4.0], [4.0, 0.0]]
        )
    if "points" not in st.session_state:
        st.session_state.points = np.empty((0, 2))
    if "labels" not in st.session_state:
        st.session_state.labels = np.array([], dtype=int)
    if "centroids" not in st.session_state:
        st.session_state.centroids = st.session_state.base_centroids.copy()
    if "metrics" not in st.session_state:
        st.session_state.metrics = {
            "silhouette_score": None,
            "active_clusters": 0,
            "noise_ratio": 0.0,
        }
    if "logs" not in st.session_state:
        st.session_state.logs = []


def _reset_state() -> None:
    st.session_state.running = False
    st.session_state.batch_id = 0
    st.session_state.seed = 7
    st.session_state.rng = np.random.default_rng(st.session_state.seed)
    st.session_state.base_centroids = np.array(
        [[-4.0, 0.0], [0.0, 4.0], [4.0, 0.0]]
    )
    st.session_state.points = np.empty((0, 2))
    st.session_state.labels = np.array([], dtype=int)
    st.session_state.centroids = st.session_state.base_centroids.copy()
    st.session_state.metrics = {
        "silhouette_score": None,
        "active_clusters": 0,
        "noise_ratio": 0.0,
    }
    st.session_state.logs = []


def _generate_mock_batch(
    *,
    batch_size: int,
    drift_rate: float,
    step: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_centroids = st.session_state.base_centroids
    directions = np.array([[0.5, 0.1], [-0.2, 0.4], [0.3, -0.3]])
    centroids = base_centroids + directions * drift_rate * step
    noise_ratio = 0.05
    noise_count = int(batch_size * noise_ratio)
    cluster_count = centroids.shape[0]
    cluster_points = max(batch_size - noise_count, 0)
    per_cluster = cluster_points // cluster_count
    remainder = cluster_points - per_cluster * cluster_count

    points = []
    labels = []
    for idx, centroid in enumerate(centroids):
        count = per_cluster + (1 if idx < remainder else 0)
        if count == 0:
            continue
        blob = rng.normal(0.0, 0.6, size=(count, 2)) + centroid
        points.append(blob)
        labels.append(np.full(count, idx))

    if noise_count > 0:
        noise = rng.uniform(-8.0, 8.0, size=(noise_count, 2))
        points.append(noise)
        labels.append(np.full(noise_count, -1))

    if points:
        points_array = np.vstack(points)
        labels_array = np.concatenate(labels)
    else:
        points_array = np.empty((0, 2))
        labels_array = np.array([], dtype=int)

    shuffle: np.ndarray
    shuffle = rng.permutation(points_array.shape[0]) if points_array.size else np.array([])
    return points_array[shuffle], labels_array[shuffle], centroids


def _compute_metrics(points: np.ndarray, labels: np.ndarray) -> dict[str, float | int | None]:
    active_labels = labels[labels != -1]
    active_clusters = len(set(active_labels.tolist()))
    noise_ratio = float(np.mean(labels == -1)) if labels.size else 0.0
    silhouette = None
    if points.shape[0] > 1 and active_clusters > 1:
        try:
            from sklearn.metrics import silhouette_score  # type: ignore[import-untyped]

            silhouette = float(silhouette_score(points, labels))
        except Exception:
            silhouette = None
    return {
        "silhouette_score": silhouette,
        "active_clusters": active_clusters,
        "noise_ratio": noise_ratio,
    }


def _build_plot(
    points: np.ndarray, labels: np.ndarray, centroids: np.ndarray
) -> go.Figure:
    palette = plotly_colors.qualitative.Set2
    fig = go.Figure()
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
                    marker=dict(color="gray", size=6, opacity=0.4),
                )
            )
        else:
            color = palette[label % len(palette)]
            fig.add_trace(
                go.Scatter(
                    x=points[mask, 0],
                    y=points[mask, 1],
                    mode="markers",
                    name=f"Cluster {label}",
                    marker=dict(color=color, size=6),
                )
            )
    for idx, centroid in enumerate(centroids):
        fig.add_trace(
            go.Scatter(
                x=[centroid[0]],
                y=[centroid[1]],
                mode="markers+text",
                name=f"C{idx}",
                text=[f"C{idx}"],
                textposition="top center",
                marker=dict(color="black", size=14, symbol="x"),
            )
        )
    fig.update_layout(
        title="Current clusters (mock)",
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        legend=dict(orientation="v"),
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def _append_log_entry(
    *,
    batch_id: int,
    n_samples: int,
    active_clusters: int,
    noise_ratio: float,
    latency_ms: float,
) -> None:
    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "batch_id": batch_id,
        "n_samples": n_samples,
        "active_clusters": active_clusters,
        "noise_ratio": round(noise_ratio, 3),
        "latency_ms": round(latency_ms, 2),
    }
    st.session_state.logs.append(entry)
    st.session_state.logs = st.session_state.logs[-20:]


def _next_batch(batch_size: int, drift_rate: float) -> None:
    start = datetime.utcnow()
    points, labels, centroids = _generate_mock_batch(
        batch_size=batch_size,
        drift_rate=drift_rate,
        step=st.session_state.batch_id,
        rng=st.session_state.rng,
    )
    metrics = _compute_metrics(points, labels)
    latency_ms = (datetime.utcnow() - start).total_seconds() * 1000

    st.session_state.points = points
    st.session_state.labels = labels
    st.session_state.centroids = centroids
    st.session_state.metrics = metrics

    active_clusters = int(metrics["active_clusters"] or 0)
    noise_ratio = float(metrics["noise_ratio"] or 0.0)
    _append_log_entry(
        batch_id=st.session_state.batch_id,
        n_samples=int(points.shape[0]),
        active_clusters=active_clusters,
        noise_ratio=noise_ratio,
        latency_ms=latency_ms,
    )
    st.session_state.batch_id += 1


def main() -> None:
    st.set_page_config(page_title="Clustering Dashboard", layout="wide")
    _init_state()

    st.title("Clustering Dashboard")
    st.write(
        "Explore mock DenStream behavior, drift, and batch-level metrics "
        "with deterministic synthetic data."
    )

    with st.sidebar:
        st.subheader("Stream controls")
        start = st.button("Start Stream", use_container_width=True)
        pause = st.button("Pause", use_container_width=True)
        reset = st.button("Reset", use_container_width=True)
        next_batch = st.button("Next Batch", use_container_width=True)
        batch_size = st.slider("batch_size", 50, 2000, 500, step=50)
        drift_rate = st.slider("drift_rate", 0.0, 2.0, 0.2, step=0.05)
        refresh_interval = st.slider("refresh_interval_seconds", 1, 10, 3)
        mock_mode = st.checkbox("Mock mode", value=True)

    if start:
        st.session_state.running = True
    if pause:
        st.session_state.running = False
    if reset:
        _reset_state()
    if next_batch:
        _next_batch(batch_size, drift_rate)

    if not mock_mode:
        st.warning("Backend mode is not available yet. Mock mode only.")

    tabs = st.tabs(["Current State"])
    with tabs[0]:
        left, right = st.columns([3, 1])
        with left:
            fig = _build_plot(
                st.session_state.points,
                st.session_state.labels,
                st.session_state.centroids,
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Metrics")
            metrics = st.session_state.metrics
            st.metric("silhouette_score", metrics["silhouette_score"])
            st.metric("active_clusters", metrics["active_clusters"])
            st.metric("noise_ratio", metrics["noise_ratio"])

            st.subheader("Recent logs")
            if st.session_state.logs:
                df = pd.DataFrame(st.session_state.logs)
                st.dataframe(df, use_container_width=True, height=300)
            else:
                st.write("No batches processed yet.")

    st.caption(f"Refresh interval setting: {refresh_interval} seconds")


if __name__ == "__main__":
    main()
