from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from time import time
from typing import TYPE_CHECKING

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
from utils import _lonlat_to_m, measure_latency

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


if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

try:
    from sklearn.metrics import silhouette_score as sklearn_silhouette_score  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    sklearn_silhouette_score = None

CENTROID_DIMS = 2
HIGH_BATCH_SIZE = 1500
HIGH_DRIFT_RATE = 1.5
MIN_HISTORY_FOR_DRIFT = 2
DEFAULT_NYC_FILE = "data/nyc_taxi/nyc_taxi_jan01.csv"
TAB_NAMES = ["Current State", "History View", "Logs"]


NYC_EPSILON_DEFAULT_M = 500.0
NYC_EPSILON_MIN_M = 50.0
NYC_EPSILON_MAX_M = 10000.0
NYC_EPSILON_STEP_M = 50.0

SYN_EPSILON_MIN = 0.001
SYN_EPSILON_MAX = 1.0
SYN_EPSILON_STEP = 0.001

NYC_MU_DEFAULT = 30.0
NYC_MU_MIN = 5.0
NYC_MU_MAX = 200.0
NYC_MU_STEP = 5.0

SYN_MU_DEFAULT = 2.5
SYN_MU_MIN = 1.0
SYN_MU_MAX = 20.0
SYN_MU_STEP = 0.5


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


@dataclass(frozen=True, slots=True)
class UiActions:
    apply_params: bool
    start: bool
    pause: bool
    reset: bool
    next_batch: bool

    refresh_interval: float
    max_history: int
    ttl_seconds: float

    show_last_n: bool
    last_n: int
    show_labels: bool

    log_limit: int
    auto_refresh_logs: bool
    refresh_logs_clicked: bool


@dataclass(frozen=True, slots=True)
class SidebarState:
    params: StreamParams
    actions: UiActions

    # values used by handlers
    data_source: str
    nyc_file_path: str

    # synthetic-only settings
    batch_size: int
    drift_rate: float
    dynamic_enabled: bool
    dynamic_interval: int
    dynamic_min: int
    dynamic_max: int


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
        st.session_state.centroids = {}
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
        st.session_state.point_mode = True
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
    st.session_state.centroids = {}
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


def _mark_small_clusters_as_noise(
    labels: np.ndarray,
    *,
    min_cluster_size: int,
) -> np.ndarray:
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
    if isinstance(centroids, dict):
        for label, centroid in centroids.items():
            if isinstance(label, int) and isinstance(centroid, (list, tuple)) and len(centroid) == CENTROID_DIMS:
                centroid_map[label] = (float(centroid[0]), float(centroid[1]))
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
        st.session_state.centroids = {}
        if st.session_state.data_source == "nyc_taxi":
            st.session_state.labels = _mark_small_clusters_as_noise(
                st.session_state.labels,
                min_cluster_size=2,
            )
    centroid_map: dict[int, tuple[float, float]] = {}
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
    st.session_state.metrics = _compute_metrics(
        st.session_state.points,
        st.session_state.labels,
    )
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


def _get_active_tab_index() -> int:
    qp = st.query_params
    raw = qp.get("tab", "0")
    if isinstance(raw, list):
        raw = raw[0] if raw else "0"
    try:
        idx = int(raw)
    except (TypeError, ValueError):
        idx = 0
    return max(0, min(idx, len(TAB_NAMES) - 1))


def _denstream_defaults(data_source: str) -> dict[str, float]:
    """UI defaults (not backend truth)."""
    if data_source == "nyc_taxi":
        return {
            "epsilon": NYC_EPSILON_DEFAULT_M,  # meters
            "beta": 0.30,
            "mu": NYC_MU_DEFAULT,
        }
    return {
        "epsilon": 0.50,  # synthetic coordinate units
        "beta": 0.50,
        "mu": SYN_MU_DEFAULT,
    }


def _epsilon_input(den_form: DeltaGenerator, data_source: str, default_epsilon: float) -> float:
    if data_source == "nyc_taxi":
        return float(
            den_form.number_input(
                "epsilon_meters",
                min_value=NYC_EPSILON_MIN_M,
                max_value=NYC_EPSILON_MAX_M,
                value=default_epsilon,
                step=NYC_EPSILON_STEP_M,
                format="%.0f",
                help="Neighborhood radius in meters (requires clustering in projected meters).",
            ),
        )
    return float(
        den_form.number_input(
            "epsilon",
            min_value=SYN_EPSILON_MIN,
            max_value=SYN_EPSILON_MAX,
            value=default_epsilon,
            step=SYN_EPSILON_STEP,
            format="%.4f",
            help="Neighborhood radius in synthetic coordinate units.",
        ),
    )


def _mu_input(den_form: DeltaGenerator, data_source: str, default_mu: float) -> float:
    if data_source == "nyc_taxi":
        return float(
            den_form.number_input(
                "mu",
                min_value=NYC_MU_MIN,
                max_value=NYC_MU_MAX,
                value=default_mu,
                step=NYC_MU_STEP,
                format="%.0f",
                help="Minimum microcluster weight to be considered a core cluster.",
            ),
        )
    return float(
        den_form.number_input(
            "mu",
            min_value=SYN_MU_MIN,
            max_value=SYN_MU_MAX,
            value=default_mu,
            step=SYN_MU_STEP,
            format="%.2f",
            help="Minimum microcluster weight to be considered a core cluster.",
        ),
    )


def _build_synthetic_stream_payload(state: SidebarState) -> dict[str, object]:
    # Centralize payload construction to avoid duplicated dicts in start/apply.
    return {
        "points_per_cluster": state.batch_size,
        "drift": state.drift_rate,
        "dynamic_enabled": state.dynamic_enabled,
        "dynamic_min_clusters": state.dynamic_min,
        "dynamic_max_clusters": state.dynamic_max,
        "dynamic_interval": state.dynamic_interval,
    }


def render_sidebar(client: ApiClient) -> SidebarState:
    """
    Renders the entire sidebar and returns:
    - StreamParams used by the backend calls
    - UiActions flags + display options
    - Key widget values needed for action handlers
    """
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
        nyc_file_path = st.session_state.nyc_file_path

        refresh_interval = params_form.slider(
            "update_interval_seconds",
            0.1,
            5.0,
            0.5,
            step=0.1,
        )

        if data_source == "nyc_taxi":
            nyc_file_path = params_form.text_input(
                "nyc_file_path",
                value=st.session_state.nyc_file_path,
            )
            st.session_state.nyc_file_path = nyc_file_path
        else:
            batch_size = params_form.slider("points_per_cluster", 10, 1000, 150, step=10)
            drift_rate = params_form.slider("drift_rate", 0.0, 2.0, 0.2, step=0.05)

        apply_params = params_form.form_submit_button("Apply")

        ttl_seconds = st.slider("point_ttl_seconds (0 = off)", 0.0, 50.0, 5.0, step=0.5)
        st.session_state.ttl_seconds = ttl_seconds

        dynamic_enabled = False
        dynamic_interval = 15
        dynamic_min = 2
        dynamic_max = 6

        if data_source == "synthetic":
            st.session_state.point_mode = True
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

            defaults = _denstream_defaults(data_source)

            decay_factor = float(
                den_form.number_input(
                    "decay_factor",
                    min_value=0.001,
                    max_value=0.2,
                    value=0.01,
                    step=0.001,
                    format="%.4f",
                ),
            )

            epsilon = _epsilon_input(den_form, data_source, defaults["epsilon"])

            beta = float(
                den_form.number_input(
                    "beta",
                    min_value=0.1,
                    max_value=0.9,
                    value=defaults["beta"],
                    step=0.05,
                    format="%.2f",
                ),
            )

            mu = _mu_input(den_form, data_source, defaults["mu"])

            n_samples_init = int(
                den_form.number_input(
                    "n_samples_init",
                    min_value=10,
                    max_value=1000,
                    value=200,
                    step=10,
                ),
            )

            stream_speed = int(
                den_form.number_input(
                    "stream_speed",
                    min_value=1,
                    max_value=500,
                    value=50,
                    step=5,
                ),
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
        batch_size=batch_size if data_source == "synthetic" else 0,
        drift_rate=drift_rate if data_source == "synthetic" else 0.0,
        update_interval_seconds=refresh_interval,
    )

    actions = UiActions(
        apply_params=apply_params,
        start=start,
        pause=pause,
        reset=reset,
        next_batch=next_batch,
        refresh_interval=refresh_interval,
        max_history=max_history,
        ttl_seconds=ttl_seconds,
        show_last_n=show_last_n,
        last_n=last_n,
        show_labels=show_labels,
        log_limit=int(st.session_state.get("log_limit", 200)),
        auto_refresh_logs=bool(st.session_state.get("auto_refresh_logs", True)),
        refresh_logs_clicked=False,
    )

    return SidebarState(
        params=params,
        actions=actions,
        data_source=data_source,
        nyc_file_path=nyc_file_path,
        batch_size=batch_size,
        drift_rate=drift_rate,
        dynamic_enabled=dynamic_enabled,
        dynamic_interval=dynamic_interval,
        dynamic_min=dynamic_min,
        dynamic_max=dynamic_max,
    )


def handle_apply_params(state: SidebarState, client: ApiClient) -> None:
    if not state.actions.apply_params:
        return

    if state.data_source == "synthetic":
        if state.dynamic_min > state.dynamic_max:
            st.error("dynamic_min_clusters must be <= dynamic_max_clusters.")
            return
        payload = _build_synthetic_stream_payload(state)
        try:
            client.configure_stream(payload)
            st.session_state.stream_state = client.get_stream_state()
            st.success("Stream parameters updated.")
        except BackendError as exc:
            st.error(str(exc))
        return

    # nyc_taxi
    try:
        client.configure_nyc_taxi({"file_path": state.nyc_file_path})
        st.session_state.nyc_bounds = client.get_nyc_taxi_bounds()
        st.success("Stream parameters updated.")
    except BackendError as exc:
        st.error(str(exc))


def handle_buttons(state: SidebarState, client: ApiClient) -> None:
    st.session_state.max_history_points = state.actions.max_history
    st.session_state.ttl_seconds = state.actions.ttl_seconds

    if state.data_source == "synthetic" and (state.batch_size > HIGH_BATCH_SIZE or state.drift_rate > HIGH_DRIFT_RATE):
        st.warning("High values may reduce responsiveness.")

    if state.actions.start:
        if state.data_source == "nyc_taxi":
            try:
                client.configure_nyc_taxi({"file_path": state.nyc_file_path})
                st.session_state.nyc_bounds = client.get_nyc_taxi_bounds()
            except BackendError as exc:
                st.error(str(exc))
                return
            st.session_state.running = True
            st.success("NYC Taxi stream started.")
            return

        # synthetic
        if state.dynamic_min > state.dynamic_max:
            st.error("dynamic_min_clusters must be <= dynamic_max_clusters.")
            return

        payload = _build_synthetic_stream_payload(state)
        try:
            client.configure_stream(payload)
            st.session_state.stream_state = client.get_stream_state()
        except BackendError as exc:
            st.error(str(exc))
            return

        ok, message = _call_backend("start", state.params, client)
        if ok:
            st.session_state.running = True
            st.success("Stream started.")
        else:
            st.error(message)

    if state.actions.pause:
        st.session_state.running = False
        if state.data_source == "synthetic":
            _call_backend("pause", state.params, client)
        st.info("Stream paused.")

    if state.actions.reset:
        if state.data_source == "nyc_taxi":
            try:
                client.reset_nyc_taxi()
                _reset_state()
                st.success("NYC Taxi stream reset.")
            except BackendError as exc:
                st.error(str(exc))
        else:
            ok, message = _call_backend("reset", state.params, client)
            if ok:
                _reset_state()
                st.success("Stream reset.")
            else:
                st.error(message)

    if state.actions.next_batch:
        try:
            _next_batch_backend(state.params, client)
            st.success("Fetched next batch from backend.")
        except BackendError as exc:
            st.error(str(exc))


def handle_autorun(state: SidebarState, client: ApiClient) -> None:
    if not st.session_state.running:
        return

    st_autorefresh(interval=int(state.actions.refresh_interval * 1000), key="stream-refresh")
    try:
        _next_batch_backend(state.params, client)
        if state.data_source == "synthetic":
            st.session_state.stream_state = client.get_stream_state()
    except BackendError as exc:
        st.error(str(exc))
        st.session_state.running = False


def render_tab_current_state() -> None:
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
    st.subheader("Centroid trajectories")
    history = st.session_state.centroid_history
    if not history:
        st.info("No centroid history yet. Run a few batches first.")
        return

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


def main() -> None:
    st.set_page_config(page_title="Clustering Dashboard", layout="wide")
    _init_state()

    st.title("Clustering Dashboard")
    st.write("Explore DenStream behavior, drift, and batch-level metrics with synthetic data.")

    client = ApiClient()

    sidebar_state = render_sidebar(client)

    st.session_state.max_history_points = sidebar_state.actions.max_history

    # Handle actions
    handle_apply_params(sidebar_state, client)
    handle_buttons(sidebar_state, client)
    handle_autorun(sidebar_state, client)

    # Render tabs
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = TAB_NAMES[_get_active_tab_index()]

    active_name = st.radio(
        "tabs",
        TAB_NAMES,
        key="active_tab",
        horizontal=True,
        label_visibility="collapsed",
    )

    active_idx = TAB_NAMES.index(active_name)
    current_tab = st.query_params.get("tab")
    if isinstance(current_tab, list):
        current_tab = current_tab[0] if current_tab else None
    if current_tab != str(active_idx):
        st.query_params["tab"] = str(active_idx)

    if active_name == "Current State":
        render_tab_current_state()
    elif active_name == "History View":
        render_tab_history_view(
            show_labels=sidebar_state.actions.show_labels,
            show_last_n=sidebar_state.actions.show_last_n,
            last_n=sidebar_state.actions.last_n,
        )
    else:
        render_tab_logs(client)

    st.caption(f"Refresh interval setting: {sidebar_state.actions.refresh_interval} seconds")


if __name__ == "__main__":
    main()
