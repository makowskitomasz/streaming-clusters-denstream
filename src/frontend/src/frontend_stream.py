from __future__ import annotations

from collections import deque
from datetime import UTC, datetime
from time import time

import numpy as np
import streamlit as st
from api_client import ApiClient, BackendError, StreamParams, StreamPoint
from frontend_metrics import append_local_metrics, apply_metrics, compute_metrics
from frontend_state import CENTROID_DIMS, UiLogEntry
from history import CentroidSnapshot, append_history, compute_centroids_from_points
from utils import _lonlat_to_m, measure_latency


def points_from_batch(
    points: list[StreamPoint],
    *,
    use_ingest_time: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Convert streamed points to arrays for coordinates, labels, and timestamps."""
    if not points:
        return None
    coords = np.asarray([[point.x, point.y] for point in points], dtype=float)
    labels = [-1 if point.cluster_id is None or point.noise else point.cluster_id for point in points]
    if use_ingest_time:
        now = time()
        timestamps = np.full((len(points),), now, dtype=float)
    else:
        timestamps = np.asarray(
            [point.timestamp if point.timestamp is not None else time() for point in points],
            dtype=float,
        )
    return coords, np.asarray(labels, dtype=int), timestamps


def mark_small_clusters_as_noise(
    labels: np.ndarray,
    *,
    min_cluster_size: int,
) -> np.ndarray:
    """Relabel clusters smaller than the threshold as noise (-1)."""
    if labels.size == 0:
        return labels
    valid = labels[labels != -1]
    if valid.size == 0:
        return labels
    unique, counts = np.unique(valid, return_counts=True)
    small = unique[counts < min_cluster_size]
    if small.size == 0:
        return labels
    updated = labels.copy()
    updated[np.isin(updated, small)] = -1
    return updated


def build_plot_data() -> tuple[
    list[tuple[float, float]],
    list[int],
    dict[int, tuple[float, float]],
]:
    """Prepare plot-ready points, labels, and centroid maps."""
    points = st.session_state.points
    labels = st.session_state.labels
    if points.size == 0 or labels.size == 0:
        return [], [], {}
    if len(points) != len(labels):
        st.error("Points and labels lengths do not match.")
        return [], [], {}
    points_list = [(float(x), float(y)) for x, y in points.tolist()]
    labels_list = [int(label) for label in labels.tolist()]
    centroids = st.session_state.centroids
    centroid_map: dict[int, tuple[float, float]] = {}
    if isinstance(centroids, dict):
        for label, centroid in centroids.items():
            if isinstance(label, int) and isinstance(centroid, (list, tuple)) and len(centroid) == CENTROID_DIMS:
                centroid_map[label] = (float(centroid[0]), float(centroid[1]))
    return points_list, labels_list, centroid_map


def record_centroid_history(
    *,
    batch_id: int,
    centroid_map: dict[int, tuple[float, float]],
    max_history: int,
) -> None:
    """Append a centroid snapshot to the history buffer."""
    if not centroid_map:
        return
    if st.session_state.centroid_history.maxlen != max_history:
        st.session_state.centroid_history = deque(
            st.session_state.centroid_history,
            maxlen=max_history,
        )
    snapshot = CentroidSnapshot(
        batch_id=batch_id,
        timestamp=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        centroids=centroid_map,
    )
    append_history(st.session_state.centroid_history, snapshot)


def refresh_logs(client: ApiClient, limit: int) -> None:
    """Fetch recent logs and update log-related session state."""
    with measure_latency() as timer:
        try:
            logs = client.get_recent_logs(limit=limit)
            st.session_state.recent_logs = logs
            st.session_state.logs_last_error = None
            st.session_state.logs_last_refresh_ts = datetime.now(tz=UTC).isoformat(timespec="seconds")
        except BackendError as exc:
            st.session_state.logs_last_error = str(exc)
    append_log_entry(
        action="logs_refresh",
        batch_id=st.session_state.batch_id,
        n_samples=int(st.session_state.points.shape[0]),
        active_clusters=int(st.session_state.metrics["active_clusters"] or 0),
        noise_ratio=float(st.session_state.metrics["noise_ratio"] or 0.0),
        latency_ms=timer.ms,
        status="success" if st.session_state.logs_last_error is None else "error",
    )


def append_log_entry(
    *,
    action: str,
    batch_id: int,
    n_samples: int,
    active_clusters: int,
    noise_ratio: float,
    latency_ms: float,
    status: str = "success",
) -> None:
    """Append a UI log entry to the in-memory log list."""
    entry = UiLogEntry(
        timestamp=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        action=action,
        status=status,
        batch_id=batch_id,
        n_samples=n_samples,
        active_clusters=active_clusters,
        noise_ratio=noise_ratio,
        latency_ms=latency_ms,
    )
    st.session_state.logs.append(entry)


def trim_points(max_points: int) -> None:
    """Trim stored points to the most recent max_points."""
    if max_points <= 0:
        return
    if st.session_state.points.shape[0] <= max_points:
        return
    st.session_state.points = st.session_state.points[-max_points:]
    st.session_state.labels = st.session_state.labels[-max_points:]
    st.session_state.point_timestamps = st.session_state.point_timestamps[-max_points:]


def apply_ttl(ttl_seconds: float) -> None:
    """Drop points older than the TTL window."""
    if ttl_seconds <= 0:
        return
    now = time()
    cutoff = now - ttl_seconds
    mask = st.session_state.point_timestamps >= cutoff
    st.session_state.points = st.session_state.points[mask]
    st.session_state.labels = st.session_state.labels[mask]
    st.session_state.point_timestamps = st.session_state.point_timestamps[mask]


def accumulate_points(
    points: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    ttl_seconds: float,
) -> None:
    """Accumulate points into session state and apply retention rules."""
    if points.size == 0:
        return
    if st.session_state.points.size == 0:
        st.session_state.points = points
        st.session_state.labels = labels
        st.session_state.point_timestamps = timestamps
    else:
        st.session_state.points = np.vstack((st.session_state.points, points))
        st.session_state.labels = np.concatenate((st.session_state.labels, labels))
        st.session_state.point_timestamps = np.concatenate(
            (st.session_state.point_timestamps, timestamps),
        )
    apply_ttl(ttl_seconds)
    trim_points(st.session_state.max_history_points)


def next_batch_backend(params: StreamParams, client: ApiClient) -> None:
    """Fetch a batch from the backend and update local state."""
    with measure_latency() as timer:
        if st.session_state.data_source == "nyc_taxi":
            response = client.next_nyc_taxi_second_cluster_points()
        elif st.session_state.point_mode:
            response = client.next_point(st.session_state.points_per_tick)
        else:
            response = client.next_batch(params)
    latency_ms = timer.ms
    parsed = points_from_batch(
        response.points,
        use_ingest_time=st.session_state.data_source == "nyc_taxi",
    )
    if parsed:
        points, labels, timestamps = parsed
        accumulate_points(points, labels, timestamps, st.session_state.ttl_seconds)
        st.session_state.centroids = {}
        if st.session_state.data_source == "nyc_taxi":
            st.session_state.labels = mark_small_clusters_as_noise(
                st.session_state.labels,
                min_cluster_size=2,
            )
        st.session_state.metrics = compute_metrics(
            st.session_state.points,
            st.session_state.labels,
        )
    raw_points = response.raw.get("points")
    if isinstance(raw_points, list) and raw_points:
        try:
            if st.session_state.data_source == "nyc_taxi":
                transformed = []
                for p in raw_points:
                    if not isinstance(p, dict):
                        continue
                    lon = p.get("x")
                    lat = p.get("y")
                    if not isinstance(lon, (int, float)) or not isinstance(lat, (int, float)):
                        continue
                    x_m, y_m = _lonlat_to_m(float(lon), float(lat))
                    q = dict(p)
                    q["x"] = x_m
                    q["y"] = y_m
                    transformed.append(q)
                client.update_denstream(transformed)
            else:
                client.update_denstream(raw_points)
        except BackendError as exc:
            st.warning(f"Clustering update failed: {exc}")
    centroid_map = compute_centroids_from_points(
        st.session_state.points,
        st.session_state.labels,
    )
    if centroid_map:
        st.session_state.centroids = centroid_map
    try:
        metrics = client.get_latest_metrics()
        apply_metrics(metrics)
        if not metrics.raw.get("latest"):
            append_local_metrics(
                batch_id=response.batch_id,
                metrics=st.session_state.metrics,
                latency_ms=latency_ms,
            )
    except BackendError as exc:
        st.warning(f"Metrics unavailable: {exc}")
        append_local_metrics(
            batch_id=response.batch_id,
            metrics=st.session_state.metrics,
            latency_ms=latency_ms,
        )
    if response.batch_id is not None:
        st.session_state.batch_id = response.batch_id
    else:
        st.session_state.batch_id += 1
    st.session_state.backend_status = "Connected"
    metrics_state = st.session_state.metrics
    record_centroid_history(
        batch_id=st.session_state.batch_id,
        centroid_map=centroid_map,
        max_history=st.session_state.max_history_points,
    )
    append_log_entry(
        action="live_next_batch",
        batch_id=st.session_state.batch_id,
        n_samples=int(st.session_state.points.shape[0]),
        active_clusters=int(metrics_state["active_clusters"] or 0),
        noise_ratio=float(metrics_state["noise_ratio"] or 0.0),
        latency_ms=latency_ms,
    )
    if parsed is None:
        st.session_state.metrics = compute_metrics(
            st.session_state.points,
            st.session_state.labels,
        )


def call_backend(
    action: str,
    params: StreamParams,
    client: ApiClient,
) -> tuple[bool, str]:
    """Invoke backend control actions and record telemetry."""
    with measure_latency() as timer:
        try:
            match action:
                case "start":
                    client.start_stream(params)
                case "reset":
                    client.reset_stream()
                case "pause":
                    client.pause_stream()
                case _:
                    return False, "Unsupported action"
        except BackendError as exc:
            st.session_state.backend_status = f"Error: {exc}"
            return False, str(exc)
    latency_ms = timer.ms
    st.session_state.backend_status = "Connected"
    append_log_entry(
        action=f"backend_{action}",
        batch_id=st.session_state.batch_id,
        n_samples=int(st.session_state.points.shape[0]),
        active_clusters=int(st.session_state.metrics["active_clusters"] or 0),
        noise_ratio=float(st.session_state.metrics["noise_ratio"] or 0.0),
        latency_ms=latency_ms,
    )
    return True, "OK"
