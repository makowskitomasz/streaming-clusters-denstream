from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import streamlit as st

if TYPE_CHECKING:
    from api_client import StreamParams

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
        """Convert the log entry to a row-friendly dict."""
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


def init_state() -> None:
    """Initialize Streamlit session state defaults."""
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


def reset_state() -> None:
    """Reset Streamlit session state to defaults."""
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
    st.session_state.point_mode = True
    st.session_state.points_per_tick = 25
    st.session_state.ttl_seconds = 5.0
    st.session_state.stream_state = {}
    st.session_state.data_source = "synthetic"
    st.session_state.nyc_file_path = DEFAULT_NYC_FILE
    st.session_state.nyc_bounds = None


def extract_nyc_ranges(
    bounds: dict[str, object],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Extract lon/lat ranges from the NYC bounds payload."""
    min_lon = as_float(bounds.get("min_lon"))
    max_lon = as_float(bounds.get("max_lon"))
    min_lat = as_float(bounds.get("min_lat"))
    max_lat = as_float(bounds.get("max_lat"))
    if None in (min_lon, max_lon, min_lat, max_lat):
        return None
    assert min_lon is not None
    assert max_lon is not None
    assert min_lat is not None
    assert max_lat is not None
    return (min_lon, max_lon), (min_lat, max_lat)


def as_float(value: object) -> float | None:
    """Safely coerce a value to float when possible."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
