from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CentroidSnapshot:
    """Snapshot of centroid positions at a given step."""

    batch_id: int | str | None
    timestamp: str
    centroids: dict[int, tuple[float, float]]


def append_history(
    history: list[CentroidSnapshot],
    snapshot: CentroidSnapshot,
    max_len: int,
) -> list[CentroidSnapshot]:
    """Append a snapshot and trim history to max_len."""
    history.append(snapshot)
    if max_len > 0 and len(history) > max_len:
        history = history[-max_len:]
    return history


def compute_centroids_from_points(
    points: np.ndarray, labels: np.ndarray
) -> dict[int, tuple[float, float]]:
    """Compute centroids from points/labels, excluding noise (-1)."""
    centroids: dict[int, tuple[float, float]] = {}
    if points.size == 0 or labels.size == 0:
        return centroids
    for label in sorted({int(label) for label in labels.tolist()}):
        if label == -1:
            continue
        mask = labels == label
        if not np.any(mask):
            continue
        center = points[mask].mean(axis=0)
        centroids[label] = (float(center[0]), float(center[1]))
    return centroids
