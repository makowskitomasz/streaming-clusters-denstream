from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd  # type: ignore[import-untyped]
import streamlit as st
from frontend_metrics import compute_drift_magnitude
from frontend_state import UiActions, extract_nyc_ranges
from frontend_stream import build_plot_data, refresh_logs
from plotting import (
    build_centroid_snapshot,
    build_centroid_trajectories,
    build_cluster_scatter,
    build_logs_timeline,
)

if TYPE_CHECKING:
    from api_client import ApiClient


def render_tab_current_state() -> None:
    """Render the Current State tab with plot and metrics."""
    left, right = st.columns([3, 1])
    with left:
        if st.session_state.data_source == "nyc_taxi":
            bounds = st.session_state.nyc_bounds or {}
            extracted = extract_nyc_ranges(bounds)
            if extracted is None:
                x_range = None
                y_range = None
            else:
                x_range, y_range = extracted
        else:
            x_range = (-8.0, 8.0)
            y_range = (-8.0, 8.0)

        points_list, labels_list, centroid_map = build_plot_data()
        fig = build_cluster_scatter(
            points_list,
            labels_list,
            centroid_map,
            x_range=x_range,
            y_range=y_range,
            uirevision=f"cluster-scatter-{st.session_state.data_source}",
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Metrics & State")
        metrics = st.session_state.metrics
        latest = st.session_state.latest_metrics
        if latest is None:
            st.info("No metrics yet. Start the stream or click Next Batch.")

        silhouette = metrics["silhouette_score"]
        active_clusters = metrics["active_clusters"]
        noise_percent = (metrics["noise_ratio"] or 0.0) * 100
        drift_value = latest.drift_magnitude if latest else None
        if drift_value is None:
            drift_value = compute_drift_magnitude(st.session_state.centroid_history)
        active_points = int(st.session_state.points.shape[0]) if hasattr(st.session_state, "points") else 0

        st.metric(
            "silhouette_score",
            f"{silhouette:.3f}" if isinstance(silhouette, (int, float)) else "—",
        )
        st.metric("active_clusters", active_clusters)
        st.metric("active_points", active_points)
        st.metric("noise_percentage", f"{noise_percent:.1f}%")
        st.metric(
            "drift_magnitude",
            f"{drift_value:.3f}" if isinstance(drift_value, (int, float)) else "—",
        )

        if isinstance(drift_value, (int, float)):
            progress_value = max(0.0, min(drift_value / 5.0, 1.0))
            st.progress(progress_value)

        if st.session_state.metrics_history:
            rows = [
                {
                    "timestamp": item.timestamp,
                    "model_name": item.model_name,
                    "batch_id": item.batch_id,
                    "silhouette_score": item.silhouette_score,
                    "active_clusters": item.active_clusters,
                    "noise_ratio": item.noise_ratio,
                    "drift_magnitude": item.drift_magnitude,
                    "latency_ms": item.latency_ms,
                }
                for item in list(st.session_state.metrics_history)[-10:]
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", height=220)

        st.subheader("Recent logs")
        if st.session_state.logs:
            rows = [entry.as_row() for entry in st.session_state.logs]
            st.dataframe(pd.DataFrame(rows), width="stretch", height=220)
        else:
            st.write("No batches processed yet.")


def render_tab_history_view(*, show_labels: bool, show_last_n: bool, last_n: int) -> None:
    """Render the History View tab with trajectories or snapshots."""
    st.subheader("Centroid trajectories")
    history = st.session_state.centroid_history
    if not history:
        st.info("No centroid history yet. Run a few batches first.")
        return

    if st.session_state.data_source == "nyc_taxi":
        bounds = st.session_state.nyc_bounds or {}
        extracted = extract_nyc_ranges(bounds)
        if extracted is None:
            x_range = None
            y_range = None
        else:
            x_range, y_range = extracted
    else:
        x_range = (-8.0, 8.0)
        y_range = (-8.0, 8.0)

    view_mode = st.selectbox("history_view_mode", ["Trajectories", "Snapshot slider"])
    if view_mode == "Trajectories":
        show_timestamps = st.checkbox("Show timestamp on hover", value=True)
        only_last_n = last_n if show_last_n else None
        fig = build_centroid_trajectories(
            list(history),
            show_labels=show_labels,
            show_timestamps=show_timestamps,
            only_last_n=only_last_n,
            x_range=x_range,
            y_range=y_range,
        )
        st.plotly_chart(fig, width="stretch")
        latest = history[-1]
        table_rows = [
            {
                "cluster_id": cluster_id,
                "x": centroid[0],
                "y": centroid[1],
                "batch_id": latest.batch_id,
                "timestamp": latest.timestamp,
            }
            for cluster_id, centroid in latest.centroids.items()
        ]
        st.dataframe(pd.DataFrame(table_rows), width="stretch", height=220)
    else:
        idx = st.slider("snapshot_index", 0, len(history) - 1, len(history) - 1)
        snapshot = history[idx]
        st.caption(f"Snapshot timestamp: {snapshot.timestamp}")
        fig = build_centroid_snapshot(snapshot, show_labels=show_labels, x_range=x_range, y_range=y_range)
        st.plotly_chart(fig, width="stretch")
        table_rows = [
            {
                "cluster_id": cluster_id,
                "x": centroid[0],
                "y": centroid[1],
                "batch_id": snapshot.batch_id,
                "timestamp": snapshot.timestamp,
            }
            for cluster_id, centroid in snapshot.centroids.items()
        ]
        st.dataframe(pd.DataFrame(table_rows), width="stretch", height=220)


def render_tab_logs(client: ApiClient) -> UiActions:
    """Render the Logs tab and return log-related action state."""
    st.subheader("Logs & Timeline")

    log_limit = st.slider("log_limit", 50, 1000, int(st.session_state.get("log_limit", 200)), step=50)
    auto_refresh_logs = st.checkbox(
        "Auto-refresh logs when running",
        value=bool(st.session_state.get("auto_refresh_logs", True)),
    )

    st.session_state.log_limit = log_limit
    st.session_state.auto_refresh_logs = auto_refresh_logs

    refresh_clicked = st.button("Refresh logs")
    if refresh_clicked:
        refresh_logs(client, log_limit)

    if st.session_state.running and auto_refresh_logs:
        refresh_logs(client, log_limit)

    if st.session_state.logs_last_error:
        st.warning(st.session_state.logs_last_error)

    logs = st.session_state.recent_logs
    if logs:
        series = st.selectbox(
            "timeline_metric",
            ["latency_ms", "active_clusters", "noise_ratio", "silhouette_score"],
        )
        fig = build_logs_timeline(logs, series)
        st.plotly_chart(fig, width="stretch")

        rows = []
        for log in logs:
            noise_percent = f"{(log.noise_ratio or 0.0) * 100:.1f}%" if log.noise_ratio is not None else "—"
            rows.append(
                {
                    "timestamp": log.timestamp,
                    "batch_id": log.batch_id,
                    "active_clusters": log.active_clusters,
                    "latency_ms": log.latency_ms,
                    "noise_ratio": noise_percent,
                    "silhouette_score": log.silhouette_score,
                    "drift_magnitude": log.drift_magnitude,
                    "message": log.message,
                },
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", height=260)

        raw_messages = "\n".join(f"{log.timestamp} | {log.message}" for log in logs if log.message)
        st.text_area("raw_logs", raw_messages, height=180)
    else:
        st.info("No logs available.")

    # Return updated actions (only the bits logs tab controls)
    return UiActions(
        apply_params=False,
        start=False,
        pause=False,
        reset=False,
        next_batch=False,
        refresh_interval=(float(st.session_state.get("update_interval_seconds", 0.5)) if False else 0.5),
        max_history=int(st.session_state.get("max_history_points", 200)),
        ttl_seconds=float(st.session_state.get("ttl_seconds", 5.0)),
        show_last_n=bool(st.session_state.get("show_last_n", True)),
        last_n=int(st.session_state.get("history_window", 50)),
        show_labels=bool(st.session_state.get("show_labels", True)),
        log_limit=log_limit,
        auto_refresh_logs=auto_refresh_logs,
        refresh_logs_clicked=refresh_clicked,
    )
