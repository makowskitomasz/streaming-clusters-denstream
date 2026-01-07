from __future__ import annotations

import os
from dataclasses import dataclass
from http import HTTPStatus

import httpx

DIMENSIONS = 2


class BackendError(RuntimeError):
    """Raised when the backend is unreachable or returns an error response."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class StreamParams:
    """Parameters for stream control calls."""

    batch_size: int
    drift_rate: float
    update_interval_seconds: int

    def to_payload(self) -> dict[str, float | int]:
        return {
            "batch_size": self.batch_size,
            "drift_rate": self.drift_rate,
            "update_interval_seconds": self.update_interval_seconds,
        }


@dataclass(frozen=True, slots=True)
class StreamPoint:
    """Point returned by the stream endpoints."""

    x: float
    y: float
    cluster_id: int | None
    noise: bool


@dataclass(frozen=True, slots=True)
class NextBatchResponse:
    """Response from the stream batch endpoint."""

    batch_id: int | None
    points: list[StreamPoint]
    raw: dict[str, object]


@dataclass(frozen=True, slots=True)
class ClusterStateResponse:
    """Response describing current clustering state."""

    centroids: dict[int, tuple[float, float]]
    raw: dict[str, object]


@dataclass(frozen=True, slots=True)
class MetricsLatestResponse:
    """Latest metrics snapshot for a model."""

    silhouette_score: float | None
    active_clusters: int | None
    noise_ratio: float | None
    drift_magnitude: float | None
    batch_id: int | str | None
    timestamp: str | None
    latency_ms: float | None
    model_name: str | None
    raw: dict[str, object]

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> MetricsLatestResponse:
        latest = payload.get("latest")
        model_payload: dict[str, object] = {}
        model_name = None
        if isinstance(latest, dict) and latest:
            model_name, model_payload = next(iter(latest.items()))
            if not isinstance(model_payload, dict):
                model_payload = {}
        silhouette = model_payload.get("silhouette_score")
        noise_ratio = model_payload.get("noise_ratio")
        active_clusters = model_payload.get("number_of_clusters")
        if active_clusters is None:
            active_clusters = model_payload.get("active_clusters")
        drift_magnitude = model_payload.get("drift_magnitude")
        batch_id = model_payload.get("batch_id")
        timestamp = model_payload.get("timestamp")
        latency_ms = model_payload.get("latency_ms")
        return cls(
            silhouette_score=_as_float(silhouette),
            active_clusters=_as_int(active_clusters),
            noise_ratio=_as_float(noise_ratio),
            drift_magnitude=_as_float(drift_magnitude),
            batch_id=_as_id(batch_id),
            timestamp=_as_str(timestamp),
            latency_ms=_as_float(latency_ms),
            model_name=model_name,
            raw=payload,
        )


@dataclass(frozen=True, slots=True)
class LogRecord:
    """Normalized backend log entry."""

    timestamp: str | None
    batch_id: int | str | None
    model_name: str | None
    active_clusters: int | None
    noise_ratio: float | None
    silhouette_score: float | None
    drift_magnitude: float | None
    latency_ms: float | None
    message: str | None
    raw: dict[str, object]

    @classmethod
    def from_payload(cls, payload: object) -> LogRecord:
        if not isinstance(payload, dict):
            return cls(
                timestamp=None,
                batch_id=None,
                model_name=None,
                active_clusters=None,
                noise_ratio=None,
                silhouette_score=None,
                drift_magnitude=None,
                latency_ms=None,
                message=None,
                raw={},
            )
        timestamp = payload.get("timestamp")
        message = payload.get("message")
        batch_id = payload.get("batch_id")
        model_name = payload.get("model_name")
        active_clusters = payload.get("active_clusters")
        noise_ratio = payload.get("noise_ratio")
        silhouette_score = payload.get("silhouette_score")
        drift_magnitude = payload.get("drift_magnitude")
        latency_ms = payload.get("latency_ms")
        return cls(
            timestamp=_as_str(timestamp),
            batch_id=_as_id(batch_id),
            model_name=_as_str(model_name),
            active_clusters=_as_int(active_clusters),
            noise_ratio=_as_float(noise_ratio),
            silhouette_score=_as_float(silhouette_score),
            drift_magnitude=_as_float(drift_magnitude),
            latency_ms=_as_float(latency_ms),
            message=_as_str(message),
            raw=payload,
        )


class ApiClient:
    """Minimal HTTP client for the Streamlit frontend."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout_seconds: float = 8.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        fallback = os.getenv("BACKEND_URL", "http://localhost:8000")
        self._base_url: str = base_url if base_url is not None else fallback
        self._timeout = httpx.Timeout(timeout_seconds)
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=self._timeout,
            transport=transport,
        )

    def start_stream(self, params: StreamParams) -> dict[str, object]:
        try:
            return self._request("POST", "/v1/stream/start", json=params.to_payload())
        except BackendError as exc:
            if exc.status_code == HTTPStatus.NOT_FOUND:
                return {}
            raise

    def reset_stream(self) -> dict[str, object]:
        return self._request("POST", "/v1/stream/reset")

    def next_batch(self, params: StreamParams) -> NextBatchResponse:
        payload = params.to_payload()
        try:
            data = self._request("POST", "/v1/stream/next", json=payload)
        except BackendError:
            data = self._request("GET", "/v1/stream/generate-cluster-points")
        return self._parse_next_batch(data)

    def get_current_state(self) -> ClusterStateResponse:
        data = self._request("GET", "/v1/clustering/denstream/clusters")
        return self._parse_cluster_state(data)

    def get_latest_metrics(self) -> MetricsLatestResponse:
        data = self._request("GET", "/v1/metrics/latest")
        return MetricsLatestResponse.from_payload(data)

    def get_recent_logs(self, limit: int = 200) -> list[LogRecord]:
        data = self._request("GET", f"/v1/logs/recent?limit={limit}")
        raw_logs = data.get("logs", [])
        if not isinstance(raw_logs, list):
            return []
        return [LogRecord.from_payload(item) for item in raw_logs]

    def ping(self) -> bool:
        try:
            self._request("GET", "/v1/health")
        except BackendError:
            return False
        return True

    def _request(
        self,
        method: str,
        path: str,
        json: dict[str, float | int] | None = None,
    ) -> dict[str, object]:
        try:
            response = self._client.request(method, path, json=json)
        except httpx.HTTPError as exc:
            raise BackendError(str(exc)) from exc
        if response.status_code == HTTPStatus.NOT_FOUND:
            msg = f"Endpoint not found: {path}"
            raise BackendError(msg, response.status_code)
        if response.is_error:
            msg = f"Error response from backend: {response.status_code}"
            raise BackendError(msg, response.status_code)
        try:
            payload = response.json()
        except ValueError:
            return {}
        if isinstance(payload, dict):
            return payload
        return {}

    def _parse_next_batch(self, payload: dict[str, object]) -> NextBatchResponse:
        raw_points = payload.get("points", [])
        points = [_parse_stream_point(item) for item in raw_points if isinstance(item, dict)]
        points = [point for point in points if point is not None]
        batch_id = payload.get("batch_id")
        return NextBatchResponse(
            batch_id=_as_int(batch_id),
            points=points,
            raw=payload,
        )

    def _parse_cluster_state(
        self,
        payload: dict[str, object],
    ) -> ClusterStateResponse:
        centroids: dict[int, tuple[float, float]] = {}
        raw_active = payload.get("active_clusters", [])
        if isinstance(raw_active, list):
            for idx, item in enumerate(raw_active):
                if not isinstance(item, dict):
                    continue
                centroid = item.get("centroid")
                parsed = _parse_centroid(centroid)
                if parsed is not None:
                    centroids[idx] = parsed
        return ClusterStateResponse(centroids=centroids, raw=payload)


def _as_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _as_int(value: object) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def _as_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _as_id(value: object) -> int | str | None:
    if isinstance(value, (int, str)):
        return value
    return None


def _parse_centroid(value: object) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != DIMENSIONS:
        return None
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None


def _parse_stream_point(item: dict[str, object]) -> StreamPoint | None:
    try:
        x = float(item.get("x", 0.0))
        y = float(item.get("y", 0.0))
    except (TypeError, ValueError):
        return None
    cluster_id = _as_int(item.get("cluster_id"))
    noise = bool(item.get("noise", False))
    return StreamPoint(x=x, y=y, cluster_id=cluster_id, noise=noise)
