from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import streamlit as st

try:
    from sklearn.metrics import silhouette_score as sklearn_silhouette_score  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    sklearn_silhouette_score = None

from api_client import MetricsLatestResponse
from frontend_state import MIN_HISTORY_FOR_DRIFT

if TYPE_CHECKING:
    from collections import deque

    from history import CentroidSnapshot


def compute_metrics(
    points: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | int | None]:
    """Compute basic clustering metrics for the current window."""
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


def apply_metrics(metrics: MetricsLatestResponse) -> None:
    """Apply backend metrics to session state when available."""
    if not metrics.raw.get("latest"):
        return
    st.session_state.latest_metrics = metrics
    st.session_state.metrics_history.append(metrics)


def append_local_metrics(
    *,
    batch_id: int | str | None,
    metrics: dict[str, float | int | None],
    latency_ms: float,
) -> None:
    """Store locally computed metrics as a latest snapshot."""
    active_clusters = metrics.get("active_clusters")
    active_clusters = int(active_clusters) if isinstance(active_clusters, (int, float)) else None
    snapshot = MetricsLatestResponse(
        silhouette_score=metrics.get("silhouette_score"),
        active_clusters=active_clusters,
        noise_ratio=metrics.get("noise_ratio"),
        drift_magnitude=None,
        batch_id=batch_id,
        timestamp=None,
        latency_ms=latency_ms,
        model_name="local",
        raw={"latest": {"local": dict(metrics)}},
    )
    st.session_state.latest_metrics = snapshot
    st.session_state.metrics_history.append(snapshot)


def compute_drift_magnitude(history: deque[CentroidSnapshot]) -> float | None:
    """Compute average centroid shift between the last two snapshots."""
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
