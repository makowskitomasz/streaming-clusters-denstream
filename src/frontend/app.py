from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter, time

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import streamlit as st

from frontend.api_client import (
    ApiClient,
    BackendError,
    MetricsLatestResponse,
    StreamParams,
    StreamPoint,
)
from frontend.plotting import build_cluster_scatter

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    def st_autorefresh(interval: int, key: str) -> None:
        last_key = f"{key}-last"
        now = time()
        last = st.session_state.get(last_key, 0.0)
        if now - last >= interval / 1000:
            st.session_state[last_key] = now
            _rerun()


def _rerun() -> None:
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
        return
    legacy = getattr(st, "experimental_rerun", None)
    if callable(legacy):  # pragma: no cover
        legacy()


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
    if "latest_metrics" not in st.session_state:
        st.session_state.latest_metrics = None
    if "metrics_history" not in st.session_state:
        st.session_state.metrics_history = []
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "backend_status" not in st.session_state:
        st.session_state.backend_status = "Disconnected"
    if "use_backend" not in st.session_state:
        st.session_state.use_backend = False


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
    st.session_state.latest_metrics = None
    st.session_state.metrics_history = []
    st.session_state.logs = []
    st.session_state.backend_status = "Disconnected"
    st.session_state.use_backend = False


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

    points: list[np.ndarray] = []
    labels: list[np.ndarray] = []
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


def _compute_metrics(
    points: np.ndarray, labels: np.ndarray
) -> dict[str, float | int | None]:
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


def _points_from_batch(
    points: list[StreamPoint],
) -> tuple[np.ndarray, np.ndarray] | None:
    if not points:
        return None
    coords = np.asarray([[point.x, point.y] for point in points], dtype=float)
    labels = []
    for point in points:
        if point.cluster_id is None or point.noise:
            labels.append(-1)
        else:
            labels.append(point.cluster_id)
    return coords, np.asarray(labels, dtype=int)


def _apply_metrics(metrics: MetricsLatestResponse) -> None:
    st.session_state.metrics = {
        "silhouette_score": metrics.silhouette_score,
        "active_clusters": metrics.active_clusters or 0,
        "noise_ratio": metrics.noise_ratio or 0.0,
    }
    st.session_state.latest_metrics = metrics
    record = {
        "timestamp": metrics.timestamp,
        "model_name": metrics.model_name,
        "batch_id": metrics.batch_id,
        "silhouette_score": metrics.silhouette_score,
        "active_clusters": metrics.active_clusters,
        "noise_ratio": metrics.noise_ratio,
        "drift_magnitude": metrics.drift_magnitude,
        "latency_ms": metrics.latency_ms,
    }
    st.session_state.metrics_history.append(record)
    st.session_state.metrics_history = st.session_state.metrics_history[-100:]


def _build_plot_data() -> tuple[list[tuple[float, float]], list[int], dict[int, tuple[float, float]]]:
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
    if isinstance(centroids, np.ndarray) and centroids.size:
        for idx, centroid in enumerate(centroids.tolist()):
            if isinstance(centroid, (list, tuple)) and len(centroid) == 2:
                centroid_map[idx] = (float(centroid[0]), float(centroid[1]))
    return points_list, labels_list, centroid_map


def _append_log_entry(
    *,
    action: str,
    batch_id: int,
    n_samples: int,
    active_clusters: int,
    noise_ratio: float,
    latency_ms: float,
    status: str = "success",
) -> None:
    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "action": action,
        "status": status,
        "batch_id": batch_id,
        "n_samples": n_samples,
        "active_clusters": active_clusters,
        "noise_ratio": round(noise_ratio, 3),
        "latency_ms": round(latency_ms, 2),
    }
    st.session_state.logs.append(entry)
    st.session_state.logs = st.session_state.logs[-20:]


def _next_batch(batch_size: int, drift_rate: float) -> None:
    start = perf_counter()
    points, labels, centroids = _generate_mock_batch(
        batch_size=batch_size,
        drift_rate=drift_rate,
        step=st.session_state.batch_id,
        rng=st.session_state.rng,
    )
    metrics = _compute_metrics(points, labels)
    latency_ms = (perf_counter() - start) * 1000

    st.session_state.points = points
    st.session_state.labels = labels
    st.session_state.centroids = centroids
    st.session_state.metrics = metrics

    active_clusters = int(metrics["active_clusters"] or 0)
    noise_ratio = float(metrics["noise_ratio"] or 0.0)
    _append_log_entry(
        action="mock_next_batch",
        batch_id=st.session_state.batch_id,
        n_samples=int(points.shape[0]),
        active_clusters=active_clusters,
        noise_ratio=noise_ratio,
        latency_ms=latency_ms,
    )
    st.session_state.batch_id += 1


def _next_batch_backend(params: StreamParams, client: ApiClient) -> None:
    start = perf_counter()
    response = client.next_batch(params)
    latency_ms = (perf_counter() - start) * 1000
    parsed = _points_from_batch(response.points)
    if parsed:
        points, labels = parsed
        st.session_state.points = points
        st.session_state.labels = labels
        st.session_state.centroids = np.array([])
        st.session_state.metrics = _compute_metrics(points, labels)
    state = client.get_current_state()
    if state.centroids:
        st.session_state.centroids = np.asarray(list(state.centroids.values()))
    try:
        metrics = client.get_latest_metrics()
        _apply_metrics(metrics)
    except BackendError as exc:
        st.warning(f"Metrics unavailable: {exc}")
    if response.batch_id is not None:
        st.session_state.batch_id = response.batch_id
    else:
        st.session_state.batch_id += 1
    st.session_state.backend_status = "Connected"
    metrics_state = st.session_state.metrics
    _append_log_entry(
        action="live_next_batch",
        batch_id=st.session_state.batch_id,
        n_samples=int(st.session_state.points.shape[0]),
        active_clusters=int(metrics_state["active_clusters"] or 0),
        noise_ratio=float(metrics_state["noise_ratio"] or 0.0),
        latency_ms=latency_ms,
    )


def _call_backend(
    action: str,
    params: StreamParams,
    client: ApiClient,
) -> tuple[bool, str]:
    start = perf_counter()
    try:
        if action == "start":
            client.start_stream(params)
        elif action == "reset":
            client.reset_stream()
        else:
            return False, "Unsupported action"
    except BackendError as exc:
        st.session_state.backend_status = f"Error: {exc}"
        return False, str(exc)
    latency_ms = (perf_counter() - start) * 1000
    st.session_state.backend_status = "Connected"
    _append_log_entry(
        action=f"backend_{action}",
        batch_id=st.session_state.batch_id,
        n_samples=int(st.session_state.points.shape[0]),
        active_clusters=int(st.session_state.metrics["active_clusters"] or 0),
        noise_ratio=float(st.session_state.metrics["noise_ratio"] or 0.0),
        latency_ms=latency_ms,
    )
    return True, "OK"


def main() -> None:
    st.set_page_config(page_title="Clustering Dashboard", layout="wide")
    _init_state()

    st.title("Clustering Dashboard")
    st.write(
        "Explore mock DenStream behavior, drift, and batch-level metrics "
        "with deterministic synthetic data."
    )

    client = ApiClient()

    with st.sidebar:
        st.subheader("Stream controls")
        params_form = st.form("stream-params")
        batch_size = params_form.slider("batch_size", 50, 2000, 500, step=50)
        drift_rate = params_form.slider("drift_rate", 0.0, 2.0, 0.2, step=0.05)
        refresh_interval = params_form.slider("update_interval_seconds", 1, 10, 3)
        apply_params = params_form.form_submit_button("Apply")
        use_backend = st.checkbox(
            "Use backend (Live mode)", value=st.session_state.use_backend
        )
        st.session_state.use_backend = use_backend
        mock_mode = st.checkbox(
            "Mock mode", value=not use_backend, disabled=use_backend
        )
        start = st.button("Start Stream", use_container_width=True)
        pause = st.button("Pause", use_container_width=True)
        reset = st.button("Reset", use_container_width=True)
        next_batch = st.button("Next Batch", use_container_width=True)
        status = "Running" if st.session_state.running else "Paused"
        st.caption(f"Status: {status}")
        if st.session_state.use_backend:
            st.caption(f"Backend: {st.session_state.backend_status}")

    params = StreamParams(
        batch_size=batch_size,
        drift_rate=drift_rate,
        update_interval_seconds=refresh_interval,
    )

    if apply_params and (batch_size > 1500 or drift_rate > 1.5):
        st.warning("High values may reduce responsiveness in mock mode.")

    if start:
        if st.session_state.use_backend:
            ok, message = _call_backend("start", params, client)
            if ok:
                st.session_state.running = True
                st.success("Stream started.")
            else:
                st.error(message)
        else:
            st.session_state.running = True
            st.success("Stream started in mock mode.")
    if pause:
        st.session_state.running = False
        st.info("Stream paused locally.")
    if reset:
        if st.session_state.use_backend:
            ok, message = _call_backend("reset", params, client)
            if ok:
                _reset_state()
                st.success("Stream reset.")
            else:
                st.error(message)
        else:
            _reset_state()
            st.success("Stream reset in mock mode.")
    if next_batch:
        if st.session_state.use_backend:
            try:
                _next_batch_backend(params, client)
                st.success("Fetched next batch from backend.")
            except BackendError as exc:
                st.error(str(exc))
        else:
            _next_batch(batch_size, drift_rate)

    if st.session_state.running:
        st_autorefresh(interval=refresh_interval * 1000, key="stream-refresh")
        if st.session_state.use_backend:
            try:
                _next_batch_backend(params, client)
            except BackendError as exc:
                st.error(str(exc))
                st.session_state.running = False
        else:
            _next_batch(batch_size, drift_rate)

    if st.session_state.use_backend and mock_mode:
        st.info("Mock mode is disabled while backend mode is active.")

    tabs = st.tabs(["Current State"])
    with tabs[0]:
        left, right = st.columns([3, 1])
        with left:
            points_list, labels_list, centroid_map = _build_plot_data()
            fig = build_cluster_scatter(points_list, labels_list, centroid_map)
            st.plotly_chart(fig, use_container_width=True)

    with right:
            st.subheader("Metrics & State")
            metrics = st.session_state.metrics
            latest = st.session_state.latest_metrics
            if st.session_state.use_backend and latest is None:
                st.info("No metrics yet. Start the stream or click Next Batch.")
            silhouette = metrics["silhouette_score"]
            active_clusters = metrics["active_clusters"]
            noise_percent = (metrics["noise_ratio"] or 0.0) * 100
            drift_value = latest.drift_magnitude if latest else None

            st.metric(
                "silhouette_score",
                f"{silhouette:.3f}" if isinstance(silhouette, (int, float)) else "—",
            )
            st.metric("active_clusters", active_clusters)
            st.metric("noise_percentage", f"{noise_percent:.1f}%")
            st.metric(
                "drift_magnitude",
                f"{drift_value:.3f}" if isinstance(drift_value, (int, float)) else "—",
            )
            if isinstance(drift_value, (int, float)):
                progress_value = max(0.0, min(drift_value / 5.0, 1.0))
                st.progress(progress_value)

            if st.session_state.metrics_history:
                df = pd.DataFrame(st.session_state.metrics_history[-10:])
                st.dataframe(df, use_container_width=True, height=220)

            st.subheader("Recent logs")
            if st.session_state.logs:
                df = pd.DataFrame(st.session_state.logs)
                st.dataframe(df, use_container_width=True, height=220)
            else:
                st.write("No batches processed yet.")

    st.caption(f"Refresh interval setting: {refresh_interval} seconds")


if __name__ == "__main__":
    main()
