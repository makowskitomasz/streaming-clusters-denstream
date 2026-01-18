from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from time import time

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import streamlit as st
from api_client import (
    ApiClient,
    BackendError,
    MetricsLatestResponse,
    StreamParams,
    StreamPoint,
)
from history import CentroidSnapshot, append_history, compute_centroids_from_points
from plotting import (
    build_centroid_snapshot,
    build_centroid_trajectories,
    build_cluster_scatter,
    build_logs_timeline,
)
from utils import measure_latency

try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore[import-untyped]

    AUTOREFRESH_AVAILABLE = True
except ImportError:  # pragma: no cover
    AUTOREFRESH_AVAILABLE = False

    def st_autorefresh(interval: int, key: str) -> None:
        last_key = f"{key}-last"
        now = time()
        last = st.session_state.get(last_key, 0.0)
        if now - last >= interval / 1000:
            st.session_state[last_key] = now
            _rerun()


try:
    from sklearn.metrics import silhouette_score as sklearn_silhouette_score  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    sklearn_silhouette_score = None

CENTROID_DIMS = 2
HIGH_BATCH_SIZE = 1500
HIGH_DRIFT_RATE = 1.5
MIN_HISTORY_FOR_DRIFT = 2
DEFAULT_NYC_FILE = "data/raw/nyc_taxi/nyc_taxi_jan01.csv"


def _rerun() -> None:
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
        return
    legacy = getattr(st, "experimental_rerun", None)
    if callable(legacy):  # pragma: no cover
        legacy()


@dataclass(frozen=True, slots=True)
class UiLogEntry:
    timestamp: str
    action: str
    status: str
    batch_id: int
    n_samples: int
    active_clusters: int
    noise_ratio: float
    latency_ms: float

    def as_row(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "action": self.action,
            "status": self.status,
            "batch_id": self.batch_id,
            "n_samples": self.n_samples,
            "active_clusters": self.active_clusters,
            "noise_ratio": round(self.noise_ratio, 3),
            "latency_ms": round(self.latency_ms, 2),
        }


def _init_state() -> None:
    if "running" not in st.session_state:
        st.session_state.running = False
    if "batch_id" not in st.session_state:
        st.session_state.batch_id = 0
    if "points" not in st.session_state:
        st.session_state.points = np.empty((0, CENTROID_DIMS))
    if "labels" not in st.session_state:
        st.session_state.labels = np.array([], dtype=int)
    if "point_timestamps" not in st.session_state:
        st.session_state.point_timestamps = np.array([], dtype=float)
    if "centroids" not in st.session_state:
        st.session_state.centroids = np.array([])
    if "metrics" not in st.session_state:
        st.session_state.metrics = {
            "silhouette_score": None,
            "active_clusters": 0,
            "noise_ratio": 0.0,
        }
    if "latest_metrics" not in st.session_state:
        st.session_state.latest_metrics = None
    if "metrics_history" not in st.session_state:
        st.session_state.metrics_history = deque(maxlen=100)
    if "centroid_history" not in st.session_state:
        st.session_state.centroid_history = deque(maxlen=200)
    if "max_history_points" not in st.session_state:
        st.session_state.max_history_points = 200
    if "recent_logs" not in st.session_state:
        st.session_state.recent_logs = []
    if "logs_last_error" not in st.session_state:
        st.session_state.logs_last_error = None
    if "logs_last_refresh_ts" not in st.session_state:
        st.session_state.logs_last_refresh_ts = None
    if "logs" not in st.session_state:
        st.session_state.logs = deque(maxlen=20)
    if "backend_status" not in st.session_state:
        st.session_state.backend_status = "Disconnected"
    if "point_mode" not in st.session_state:
        st.session_state.point_mode = False
    if "points_per_tick" not in st.session_state:
        st.session_state.points_per_tick = 25
    if "ttl_seconds" not in st.session_state:
        st.session_state.ttl_seconds = 5.0
    if "stream_state" not in st.session_state:
        st.session_state.stream_state = {}
    if "data_source" not in st.session_state:
        st.session_state.data_source = "synthetic"
    if "nyc_file_path" not in st.session_state:
        st.session_state.nyc_file_path = DEFAULT_NYC_FILE
    if "nyc_bounds" not in st.session_state:
        st.session_state.nyc_bounds = None


def _reset_state() -> None:
    st.session_state.running = False
    st.session_state.batch_id = 0
    st.session_state.points = np.empty((0, CENTROID_DIMS))
    st.session_state.labels = np.array([], dtype=int)
    st.session_state.point_timestamps = np.array([], dtype=float)
    st.session_state.centroids = np.array([])
    st.session_state.metrics = {
        "silhouette_score": None,
        "active_clusters": 0,
        "noise_ratio": 0.0,
    }
    st.session_state.latest_metrics = None
    st.session_state.metrics_history = deque(maxlen=100)
    st.session_state.centroid_history = deque(maxlen=200)
    st.session_state.max_history_points = 200
    st.session_state.recent_logs = []
    st.session_state.logs_last_error = None
    st.session_state.logs_last_refresh_ts = None
    st.session_state.logs = deque(maxlen=20)
    st.session_state.backend_status = "Disconnected"
    st.session_state.point_mode = False
    st.session_state.points_per_tick = 25
    st.session_state.ttl_seconds = 5.0
    st.session_state.stream_state = {}
    st.session_state.data_source = "synthetic"
    st.session_state.nyc_file_path = DEFAULT_NYC_FILE
    st.session_state.nyc_bounds = None


def _compute_metrics(
    points: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | int | None]:
    active_labels = labels[labels != -1]
    active_clusters = len(set(active_labels.tolist()))
    noise_ratio = float(np.mean(labels == -1)) if labels.size else 0.0
    silhouette = None
    if points.shape[0] > 1 and active_clusters > 1 and sklearn_silhouette_score is not None:
        try:
            silhouette = float(sklearn_silhouette_score(points, labels))
        except (ValueError, RuntimeError):
            silhouette = None
    return {
        "silhouette_score": silhouette,
        "active_clusters": active_clusters,
        "noise_ratio": noise_ratio,
    }


def _points_from_batch(
    points: list[StreamPoint],
    *,
    use_ingest_time: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
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


def _apply_metrics(metrics: MetricsLatestResponse) -> None:
    if not metrics.raw.get("latest"):
        return
    st.session_state.latest_metrics = metrics
    st.session_state.metrics_history.append(metrics)


def _compute_drift_magnitude(history: deque[CentroidSnapshot]) -> float | None:
    if len(history) < MIN_HISTORY_FOR_DRIFT:
        return None
    current = history[-1].centroids
    previous = history[-2].centroids
    common_ids = set(current).intersection(previous)
    if not common_ids:
        return None
    distances = []
    for cluster_id in common_ids:
        x_now, y_now = current[cluster_id]
        x_prev, y_prev = previous[cluster_id]
        distances.append(float(np.hypot(x_now - x_prev, y_now - y_prev)))
    if not distances:
        return None
    return float(sum(distances) / len(distances))


def _extract_nyc_ranges(
    bounds: dict[str, object],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    min_lon = _as_float(bounds.get("min_lon"))
    max_lon = _as_float(bounds.get("max_lon"))
    min_lat = _as_float(bounds.get("min_lat"))
    max_lat = _as_float(bounds.get("max_lat"))
    if None in (min_lon, max_lon, min_lat, max_lat):
        return None
    assert min_lon is not None
    assert max_lon is not None
    assert min_lat is not None
    assert max_lat is not None
    return (min_lon, max_lon), (min_lat, max_lat)


def _as_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _build_plot_data() -> tuple[
    list[tuple[float, float]],
    list[int],
    dict[int, tuple[float, float]],
]:
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
            if isinstance(centroid, (list, tuple)) and len(centroid) == CENTROID_DIMS:
                centroid_map[idx] = (float(centroid[0]), float(centroid[1]))
    return points_list, labels_list, centroid_map


def _record_centroid_history(
    *,
    batch_id: int,
    centroid_map: dict[int, tuple[float, float]],
    max_history: int,
) -> None:
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


def _refresh_logs(client: ApiClient, limit: int) -> None:
    with measure_latency() as timer:
        try:
            logs = client.get_recent_logs(limit=limit)
            st.session_state.recent_logs = logs
            st.session_state.logs_last_error = None
            st.session_state.logs_last_refresh_ts = datetime.now(tz=UTC).isoformat(timespec="seconds")
        except BackendError as exc:
            st.session_state.logs_last_error = str(exc)
    _append_log_entry(
        action="logs_refresh",
        batch_id=st.session_state.batch_id,
        n_samples=int(st.session_state.points.shape[0]),
        active_clusters=int(st.session_state.metrics["active_clusters"] or 0),
        noise_ratio=float(st.session_state.metrics["noise_ratio"] or 0.0),
        latency_ms=timer.ms,
        status="success" if st.session_state.logs_last_error is None else "error",
    )


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


def _append_local_metrics(
    *,
    batch_id: int | str | None,
    metrics: dict[str, float | int | None],
    latency_ms: float,
) -> None:
    snapshot = MetricsLatestResponse(
        silhouette_score=metrics.get("silhouette_score"),
        active_clusters=metrics.get("active_clusters"),
        noise_ratio=metrics.get("noise_ratio"),
        drift_magnitude=None,
        batch_id=batch_id,
        timestamp=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        latency_ms=latency_ms,
        model_name="local",
        raw={"latest": {"local": dict(metrics)}},
    )
    st.session_state.latest_metrics = snapshot
    st.session_state.metrics_history.append(snapshot)


def _trim_points(max_points: int) -> None:
    if max_points <= 0:
        return
    if st.session_state.points.shape[0] <= max_points:
        return
    start = st.session_state.points.shape[0] - max_points
    st.session_state.points = st.session_state.points[start:]
    st.session_state.labels = st.session_state.labels[start:]
    st.session_state.point_timestamps = st.session_state.point_timestamps[start:]


def _apply_ttl(ttl_seconds: float) -> None:
    if ttl_seconds <= 0:
        return
    if st.session_state.point_timestamps.size == 0:
        return
    now = time()
    mask = (now - st.session_state.point_timestamps) <= ttl_seconds
    if np.all(mask):
        return
    st.session_state.points = st.session_state.points[mask]
    st.session_state.labels = st.session_state.labels[mask]
    st.session_state.point_timestamps = st.session_state.point_timestamps[mask]


def _accumulate_points(
    points: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    ttl_seconds: float,
) -> None:
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
    _apply_ttl(ttl_seconds)
    _trim_points(st.session_state.max_history_points)


def _next_batch_backend(params: StreamParams, client: ApiClient) -> None:
    with measure_latency() as timer:
        if st.session_state.data_source == "nyc_taxi":
            response = client.next_nyc_taxi_second_cluster_points()
        elif st.session_state.point_mode:
            response = client.next_point(st.session_state.points_per_tick)
        else:
            response = client.next_batch(params)
    latency_ms = timer.ms
    parsed = _points_from_batch(
        response.points,
        use_ingest_time=st.session_state.data_source == "nyc_taxi",
    )
    if parsed:
        points, labels, timestamps = parsed
        _accumulate_points(points, labels, timestamps, st.session_state.ttl_seconds)
        st.session_state.centroids = np.array([])
        st.session_state.metrics = _compute_metrics(
            st.session_state.points,
            st.session_state.labels,
        )
    raw_points = response.raw.get("points")
    if isinstance(raw_points, list) and raw_points:
        try:
            client.update_denstream(raw_points)
        except BackendError as exc:
            st.warning(f"Clustering update failed: {exc}")
    centroid_map = compute_centroids_from_points(
        st.session_state.points,
        st.session_state.labels,
    )
    if centroid_map:
        st.session_state.centroids = np.asarray(list(centroid_map.values()))
    try:
        metrics = client.get_latest_metrics()
        _apply_metrics(metrics)
        if not metrics.raw.get("latest"):
            _append_local_metrics(
                batch_id=response.batch_id,
                metrics=st.session_state.metrics,
                latency_ms=latency_ms,
            )
    except BackendError as exc:
        st.warning(f"Metrics unavailable: {exc}")
        _append_local_metrics(
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
    _record_centroid_history(
        batch_id=st.session_state.batch_id,
        centroid_map=centroid_map,
        max_history=st.session_state.max_history_points,
    )
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
    """Render the Streamlit clustering dashboard."""
    st.set_page_config(page_title="Clustering Dashboard", layout="wide")
    _init_state()

    st.title("Clustering Dashboard")
    st.write(
        "Explore DenStream behavior, drift, and batch-level metrics with synthetic data.",
    )

    client = ApiClient()

    with st.sidebar:
        st.subheader("Stream controls")
        data_source = st.selectbox(
            "data_source",
            ["synthetic", "nyc_taxi"],
            index=0 if st.session_state.data_source == "synthetic" else 1,
        )
        st.session_state.data_source = data_source
        params_form = st.form("stream-params")
        batch_size = 150
        drift_rate = 0.2
        refresh_interval = params_form.slider("update_interval_seconds", 0.1, 5.0, 0.5, step=0.1)
        if st.session_state.data_source == "nyc_taxi":
            nyc_file_path = params_form.text_input(
                "nyc_file_path",
                value=st.session_state.nyc_file_path,
            )
            st.session_state.nyc_file_path = nyc_file_path
        else:
            batch_size = params_form.slider("points_per_cluster", 10, 1000, 150, step=10)
            drift_rate = params_form.slider("drift_rate", 0.0, 2.0, 0.2, step=0.05)
        apply_params = params_form.form_submit_button("Apply")
        ttl_seconds = st.slider("point_ttl_seconds (0 = off)", 0.0, 30.0, 5.0, step=0.5)
        st.session_state.ttl_seconds = ttl_seconds
        if st.session_state.data_source == "synthetic":
            point_mode = st.checkbox("Point mode (animate)", value=False)
            st.session_state.point_mode = point_mode
            points_per_tick = st.slider("points_per_tick", 1, 200, 25, step=1)
            st.session_state.points_per_tick = points_per_tick
            dynamic_enabled = st.checkbox("Dynamic clusters", value=False)
            dynamic_interval = st.slider("dynamic_interval", 5, 50, 15, step=5)
            dynamic_min = st.slider("dynamic_min_clusters", 1, 10, 2, step=1)
            dynamic_max = st.slider("dynamic_max_clusters", 2, 12, 6, step=1)
        max_history = st.slider("max_history_points", 50, 500, 200, step=10)
        show_last_n = st.checkbox("Show only last N steps", value=True)
        last_n = st.slider("history_window", 20, 200, 50, step=10)
        show_labels = st.checkbox("Show centroid labels", value=True)
        start = st.button("Start Stream", width="stretch")
        pause = st.button("Pause", width="stretch")
        reset = st.button("Reset", width="stretch")
        next_batch = st.button("Next Batch", width="stretch")
        status = "Running" if st.session_state.running else "Paused"
        st.caption(f"Status: {status}")
        st.caption(f"Backend: {st.session_state.backend_status}")
        if st.session_state.stream_state:
            current_clusters = st.session_state.stream_state.get("n_clusters", "—")
            dyn_enabled = st.session_state.stream_state.get("dynamic_enabled", "—")
            st.caption(f"Stream clusters: {current_clusters} | dynamic: {dyn_enabled}")
        if st.session_state.running and not AUTOREFRESH_AVAILABLE:
            st.warning("Auto-refresh unavailable. Install streamlit-autorefresh to animate.")

        with st.expander("DenStream parameters", expanded=False):
            den_form = st.form("denstream-params")
            if st.session_state.data_source == "nyc_taxi":
                default_epsilon = 0.01
                default_mu = 10.0
                default_beta = 0.3
            else:
                default_epsilon = 0.5
                default_mu = 2.5
                default_beta = 0.5
            decay_factor = den_form.number_input(
                "decay_factor",
                min_value=0.001,
                max_value=0.2,
                value=0.01,
                step=0.001,
                format="%.4f",
            )
            epsilon = den_form.number_input(
                "epsilon",
                min_value=0.001,
                max_value=1.0,
                value=default_epsilon,
                step=0.001,
                format="%.4f",
            )
            beta = den_form.number_input(
                "beta",
                min_value=0.1,
                max_value=0.9,
                value=default_beta,
                step=0.05,
                format="%.2f",
            )
            mu = den_form.number_input(
                "mu",
                min_value=1.0,
                max_value=20.0,
                value=default_mu,
                step=0.5,
                format="%.2f",
            )
            n_samples_init = den_form.number_input(
                "n_samples_init",
                min_value=10,
                max_value=1000,
                value=200,
                step=10,
            )
            stream_speed = den_form.number_input(
                "stream_speed",
                min_value=1,
                max_value=500,
                value=50,
                step=5,
            )
            apply_denstream = den_form.form_submit_button("Apply DenStream settings")
            if apply_denstream:
                try:
                    client.configure_denstream(
                        {
                            "decay_factor": decay_factor,
                            "epsilon": epsilon,
                            "beta": beta,
                            "mu": mu,
                            "n_samples_init": n_samples_init,
                            "stream_speed": stream_speed,
                        },
                    )
                    st.success("DenStream configuration updated.")
                except BackendError as exc:
                    st.error(str(exc))

    params = StreamParams(
        batch_size=batch_size if st.session_state.data_source == "synthetic" else 0,
        drift_rate=drift_rate if st.session_state.data_source == "synthetic" else 0.0,
        update_interval_seconds=refresh_interval,
    )
    st.session_state.max_history_points = max_history

    if apply_params:
        config_payload = {}
        if st.session_state.data_source == "synthetic":
            if dynamic_min > dynamic_max:
                st.error("dynamic_min_clusters must be <= dynamic_max_clusters.")
                return
            config_payload = {
                "points_per_cluster": batch_size,
                "drift": drift_rate,
                "dynamic_enabled": dynamic_enabled,
                "dynamic_min_clusters": dynamic_min,
                "dynamic_max_clusters": dynamic_max,
                "dynamic_interval": dynamic_interval,
            }
        try:
            if st.session_state.data_source == "nyc_taxi":
                client.configure_nyc_taxi({"file_path": nyc_file_path})
                st.session_state.nyc_bounds = client.get_nyc_taxi_bounds()
            else:
                client.configure_stream(config_payload)
                st.session_state.stream_state = client.get_stream_state()
            st.success("Stream parameters updated.")
        except BackendError as exc:
            st.error(str(exc))
    if st.session_state.data_source == "synthetic" and (batch_size > HIGH_BATCH_SIZE or drift_rate > HIGH_DRIFT_RATE):
        st.warning("High values may reduce responsiveness.")

    if start:
        if st.session_state.data_source == "nyc_taxi":
            try:
                client.configure_nyc_taxi({"file_path": nyc_file_path})
                st.session_state.nyc_bounds = client.get_nyc_taxi_bounds()
            except BackendError as exc:
                st.error(str(exc))
                return
            st.session_state.running = True
            st.success("NYC Taxi stream started.")
        else:
            if dynamic_min > dynamic_max:
                st.error("dynamic_min_clusters must be <= dynamic_max_clusters.")
                return
            config_payload = {
                "points_per_cluster": batch_size,
                "drift": drift_rate,
                "dynamic_enabled": dynamic_enabled,
                "dynamic_min_clusters": dynamic_min,
                "dynamic_max_clusters": dynamic_max,
                "dynamic_interval": dynamic_interval,
            }
            try:
                client.configure_stream(config_payload)
                st.session_state.stream_state = client.get_stream_state()
            except BackendError as exc:
                st.error(str(exc))
                return
            ok, message = _call_backend("start", params, client)
            if ok:
                st.session_state.running = True
                st.success("Stream started.")
            else:
                st.error(message)
    if pause:
        st.session_state.running = False
        if st.session_state.data_source == "synthetic":
            _call_backend("pause", params, client)
        st.info("Stream paused.")
    if reset:
        if st.session_state.data_source == "nyc_taxi":
            try:
                client.reset_nyc_taxi()
                _reset_state()
                st.success("NYC Taxi stream reset.")
            except BackendError as exc:
                st.error(str(exc))
        else:
            ok, message = _call_backend("reset", params, client)
            if ok:
                _reset_state()
                st.success("Stream reset.")
            else:
                st.error(message)
    if next_batch:
        try:
            _next_batch_backend(params, client)
            st.success("Fetched next batch from backend.")
        except BackendError as exc:
            st.error(str(exc))

    if st.session_state.running:
        st_autorefresh(interval=int(refresh_interval * 1000), key="stream-refresh")
        try:
            _next_batch_backend(params, client)
            if st.session_state.data_source == "synthetic":
                st.session_state.stream_state = client.get_stream_state()
        except BackendError as exc:
            st.error(str(exc))
            st.session_state.running = False

    tabs = st.tabs(["Current State", "History View", "Logs"])
    with tabs[0]:
        left, right = st.columns([3, 1])
        with left:
            if st.session_state.data_source == "nyc_taxi":
                bounds = st.session_state.nyc_bounds or {}
                extracted = _extract_nyc_ranges(bounds)
                if extracted is None:
                    x_range = None
                    y_range = None
                else:
                    x_range, y_range = extracted
            else:
                x_range = (-8.0, 8.0)
                y_range = (-8.0, 8.0)
            points_list, labels_list, centroid_map = _build_plot_data()
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
                drift_value = _compute_drift_magnitude(st.session_state.centroid_history)

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

    with tabs[1]:
        st.subheader("Centroid trajectories")
        history = st.session_state.centroid_history
        if not history:
            st.info("No centroid history yet. Run a few batches first.")
        else:
            if st.session_state.data_source == "nyc_taxi":
                bounds = st.session_state.nyc_bounds or {}
                extracted = _extract_nyc_ranges(bounds)
                if extracted is None:
                    x_range = None
                    y_range = None
                else:
                    x_range, y_range = extracted
            else:
                x_range = (-8.0, 8.0)
                y_range = (-8.0, 8.0)
            view_mode = st.selectbox(
                "history_view_mode",
                ["Trajectories", "Snapshot slider"],
            )
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
                fig = build_centroid_snapshot(
                    snapshot,
                    show_labels=show_labels,
                    x_range=x_range,
                    y_range=y_range,
                )
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

    with tabs[2]:
        st.subheader("Logs & Timeline")
        log_limit = st.slider("log_limit", 50, 1000, 200, step=50)
        auto_refresh_logs = st.checkbox(
            "Auto-refresh logs when running",
            value=True,
        )
        if st.button("Refresh logs"):
            _refresh_logs(client, log_limit)

        if st.session_state.running and auto_refresh_logs:
            _refresh_logs(client, log_limit)

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

    st.caption(f"Refresh interval setting: {refresh_interval} seconds")


if __name__ == "__main__":
    main()
