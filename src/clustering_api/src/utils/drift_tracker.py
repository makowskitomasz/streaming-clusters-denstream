from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_AGE_FOR_DRIFT = 2


@dataclass(frozen=True, slots=True)
class ClusterDrift:
    """Per-cluster drift metrics for a single update cycle."""

    cluster_id: int
    previous_centroid: np.ndarray
    current_centroid: np.ndarray
    displacement: np.ndarray
    distance: float
    direction: np.ndarray | None
    speed: float
    ema_distance: float | None
    ema_direction: np.ndarray | None
    age: int
    last_seen: float | int


@dataclass(frozen=True, slots=True)
class DriftUpdate:
    """Summary of drift metrics and lifecycle events for one update cycle."""

    timestamp: float | int
    dt: float
    per_cluster: dict[int, ClusterDrift]
    appeared: list[int]
    disappeared: list[int]
    stable: list[int]
    mean_drift_distance: float | None
    max_drift_distance: float | None
    mean_speed: float | None
    num_appeared: int
    num_disappeared: int


class DriftTracker:
    """Track centroid drift and cluster lifecycle across update cycles.

    Example:
        tracker = DriftTracker(ema_alpha=0.3)
        update = tracker.update({0: np.array([0.0, 0.0])})
        update = tracker.update({0: np.array([1.0, 0.0])}, timestamp=12.0)
        drift = update.per_cluster[0].distance
    """

    def __init__(
        self,
        *,
        ema_alpha: float | None = None,
        smooth_direction: bool = True,
    ) -> None:
        self._validate_alpha(ema_alpha)
        self._ema_alpha = ema_alpha
        self._smooth_direction = smooth_direction
        self._prev_centroids: dict[int, np.ndarray] = {}
        self._ages: dict[int, int] = {}
        self._last_seen: dict[int, float | int] = {}
        self._ema_distance: dict[int, float] = {}
        self._ema_direction: dict[int, np.ndarray] = {}
        self._mode: str | None = None
        self._last_timestamp: float | None = None
        self._step = 0

    def update(
        self,
        centroids: dict[int, np.ndarray],
        *,
        timestamp: float | None = None,
    ) -> DriftUpdate:
        """Compute drift metrics for the current centroid snapshot."""
        current = self._sanitize_centroids(centroids)
        mode = "timestamp" if timestamp is not None else "step"
        self._set_mode(mode)
        if mode == "timestamp":
            now = float(timestamp)
            dt = self._compute_dt(now)
        else:
            self._step += 1
            now = self._step
            dt = 1.0

        prev_ids = set(self._prev_centroids.keys())
        curr_ids = set(current.keys())
        appeared = sorted(curr_ids - prev_ids)
        disappeared = sorted(prev_ids - curr_ids)
        stable = sorted(curr_ids & prev_ids)

        per_cluster: dict[int, ClusterDrift] = {}
        distances: list[float] = []
        speeds: list[float] = []

        for cluster_id in stable:
            prev_centroid = self._prev_centroids[cluster_id]
            curr_centroid = current[cluster_id]
            displacement = curr_centroid - prev_centroid
            distance = float(np.linalg.norm(displacement))
            direction = self._normalize_direction(displacement, distance)
            speed = distance / dt if dt > 0 else 0.0
            ema_distance, ema_direction = self._update_ema(
                cluster_id,
                distance,
                direction,
            )
            age = self._ages.get(cluster_id, 0) + 1
            self._ages[cluster_id] = age
            self._last_seen[cluster_id] = now

            per_cluster[cluster_id] = ClusterDrift(
                cluster_id=cluster_id,
                previous_centroid=prev_centroid.copy(),
                current_centroid=curr_centroid.copy(),
                displacement=displacement,
                distance=distance,
                direction=direction,
                speed=speed,
                ema_distance=ema_distance,
                ema_direction=ema_direction,
                age=age,
                last_seen=now,
            )
            distances.append(distance)
            speeds.append(speed)

        for cluster_id in appeared:
            self._ages[cluster_id] = 1
            self._last_seen[cluster_id] = now

        for cluster_id in disappeared:
            self._ages.pop(cluster_id, None)
            self._last_seen.pop(cluster_id, None)
            self._ema_distance.pop(cluster_id, None)
            self._ema_direction.pop(cluster_id, None)

        self._prev_centroids = {cid: centroid.copy() for cid, centroid in current.items()}

        mean_drift = float(np.mean(distances)) if distances else None
        max_drift = float(np.max(distances)) if distances else None
        mean_speed = float(np.mean(speeds)) if speeds else None

        return DriftUpdate(
            timestamp=now,
            dt=dt,
            per_cluster=per_cluster,
            appeared=appeared,
            disappeared=disappeared,
            stable=stable,
            mean_drift_distance=mean_drift,
            max_drift_distance=max_drift,
            mean_speed=mean_speed,
            num_appeared=len(appeared),
            num_disappeared=len(disappeared),
        )

    def get_state(self) -> dict[str, dict[int, np.ndarray | float | int]]:
        """Return a read-only snapshot of internal state."""
        return {
            "centroids": {cid: centroid.copy() for cid, centroid in self._prev_centroids.items()},
            "ages": dict(self._ages),
            "last_seen": dict(self._last_seen),
            "ema_distance": dict(self._ema_distance),
            "ema_direction": {cid: direction.copy() for cid, direction in self._ema_direction.items()},
        }

    def reset(self) -> None:
        """Clear all stored state."""
        self._prev_centroids = {}
        self._ages = {}
        self._last_seen = {}
        self._ema_distance = {}
        self._ema_direction = {}
        self._mode = None
        self._last_timestamp = None
        self._step = 0

    def get_high_drift(
        self,
        *,
        distance_threshold: float | None = None,
        speed_threshold: float | None = None,
    ) -> list[int]:
        """Return cluster IDs exceeding the supplied drift thresholds."""
        if distance_threshold is None and speed_threshold is None:
            msg = "Provide at least one drift threshold"
            raise ValueError(msg)
        high_drift: list[int] = []
        for cluster_id in self._prev_centroids:
            age = self._ages.get(cluster_id, 0)
            if age < MIN_AGE_FOR_DRIFT:
                continue
            ema_distance = self._ema_distance.get(cluster_id)
            if distance_threshold is not None and ema_distance is not None and ema_distance > distance_threshold:
                high_drift.append(cluster_id)
                continue
            if (
                speed_threshold is not None
                and ema_distance is not None
                and ema_distance > 0
                and ema_distance > speed_threshold
            ):
                high_drift.append(cluster_id)
        return sorted(set(high_drift))

    def _sanitize_centroids(
        self,
        centroids: dict[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        sanitized: dict[int, np.ndarray] = {}
        for cluster_id, centroid in centroids.items():
            array = np.asarray(centroid, dtype=float).copy()
            if array.ndim != 1:
                msg = f"Centroid for cluster {cluster_id} must be 1D array"
                raise ValueError(msg)
            if not np.isfinite(array).all():
                msg = f"Centroid for cluster {cluster_id} contains NaN/inf"
                raise ValueError(msg)
            sanitized[int(cluster_id)] = array
        return sanitized

    def _set_mode(self, mode: str) -> None:
        if self._mode is None:
            self._mode = mode
            return
        if self._mode != mode:
            msg = "Cannot mix timestamp-based and step-based updates"
            raise ValueError(msg)

    def _compute_dt(self, now: float) -> float:
        if self._last_timestamp is None:
            self._last_timestamp = now
            return 1.0
        dt = now - self._last_timestamp
        if dt <= 0:
            msg = "timestamp must be increasing between updates"
            raise ValueError(msg)
        self._last_timestamp = now
        return dt

    def _normalize_direction(
        self,
        displacement: np.ndarray,
        distance: float,
    ) -> np.ndarray | None:
        if distance == 0.0:
            return None
        return displacement / distance

    def _update_ema(
        self,
        cluster_id: int,
        distance: float,
        direction: np.ndarray | None,
    ) -> tuple[float | None, np.ndarray | None]:
        if self._ema_alpha is None:
            return None, None
        previous = self._ema_distance.get(cluster_id, distance)
        ema_distance = self._ema_alpha * distance + (1 - self._ema_alpha) * previous
        self._ema_distance[cluster_id] = ema_distance
        ema_direction = None
        if self._smooth_direction:
            prev_dir = self._ema_direction.get(
                cluster_id,
                np.zeros_like(direction) if direction is not None else None,
            )
            if direction is None or prev_dir is None:
                ema_direction = None
            else:
                ema_dir = self._ema_alpha * direction + (1 - self._ema_alpha) * prev_dir
                norm = float(np.linalg.norm(ema_dir))
                ema_direction = ema_dir / norm if norm > 0 else None
                self._ema_direction[cluster_id] = ema_dir
        return ema_distance, ema_direction

    def _validate_alpha(self, alpha: float | None) -> None:
        if alpha is None:
            return
        if alpha <= 0 or alpha > 1:
            msg = f"ema_alpha must be in the range (0, 1], got {alpha}"
            raise ValueError(msg)
