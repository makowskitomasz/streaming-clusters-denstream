from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


class BackendError(RuntimeError):
    """Raised when the backend is unreachable or returns an error response."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class StreamPoint:
    """Point returned by the stream endpoints."""

    x: float
    y: float
    cluster_id: int | None
    noise: bool


@dataclass(frozen=True)
class NextBatchResponse:
    """Response from the stream batch endpoint."""

    batch_id: int | None
    points: list[StreamPoint]
    raw: dict[str, object]


@dataclass(frozen=True)
class ClusterStateResponse:
    """Response describing current clustering state."""

    centroids: dict[int, tuple[float, float]]
    raw: dict[str, object]


@dataclass(frozen=True)
class MetricsLatestResponse:
    """Latest metrics snapshot for a model."""

    silhouette_score: float | None
    active_clusters: int
    noise_ratio: float
    model_name: str | None
    raw: dict[str, object]


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
            if exc.status_code == 404:
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
        return self._parse_metrics_latest(data)

    def ping(self) -> bool:
        try:
            self._request("GET", "/v1/health")
        except BackendError:
            return False
        return True

    def _request(
        self, method: str, path: str, json: dict[str, float | int] | None = None
    ) -> dict[str, object]:
        try:
            response = self._client.request(method, path, json=json)
        except httpx.HTTPError as exc:
            raise BackendError(str(exc)) from exc
        if response.status_code == 404:
            raise BackendError(f"Endpoint not found: {path}", response.status_code)
        if response.is_error:
            raise BackendError(
                f"Backend error {response.status_code}: {response.text}",
                response.status_code,
            )
        try:
            payload = response.json()
        except ValueError:
            return {}
        if isinstance(payload, dict):
            return payload
        return {}

    def _parse_next_batch(self, payload: dict[str, object]) -> NextBatchResponse:
        raw_points = payload.get("points", [])
        points: list[StreamPoint] = []
        if isinstance(raw_points, list):
            for item in raw_points:
                if not isinstance(item, dict):
                    continue
                try:
                    x = float(item.get("x", 0.0))
                    y = float(item.get("y", 0.0))
                except (TypeError, ValueError):
                    continue
                cluster_id = item.get("cluster_id")
                parsed_id = None
                if cluster_id is not None:
                    try:
                        parsed_id = int(cluster_id)
                    except (TypeError, ValueError):
                        parsed_id = None
                noise = bool(item.get("noise", False))
                points.append(
                    StreamPoint(
                        x=x,
                        y=y,
                        cluster_id=parsed_id,
                        noise=noise,
                    )
                )
        batch_id = payload.get("batch_id")
        return NextBatchResponse(
            batch_id=int(batch_id) if isinstance(batch_id, int) else None,
            points=points,
            raw=payload,
        )

    def _parse_cluster_state(
        self, payload: dict[str, object]
    ) -> ClusterStateResponse:
        centroids: dict[int, tuple[float, float]] = {}
        raw_active = payload.get("active_clusters", [])
        if isinstance(raw_active, list):
            for idx, item in enumerate(raw_active):
                if not isinstance(item, dict):
                    continue
                centroid = item.get("centroid")
                if (
                    isinstance(centroid, (list, tuple))
                    and len(centroid) == 2
                ):
                    try:
                        centroids[idx] = (float(centroid[0]), float(centroid[1]))
                    except (TypeError, ValueError):
                        continue
        return ClusterStateResponse(centroids=centroids, raw=payload)

    def _parse_metrics_latest(
        self, payload: dict[str, object]
    ) -> MetricsLatestResponse:
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
        return MetricsLatestResponse(
            silhouette_score=float(silhouette)
            if isinstance(silhouette, (int, float))
            else None,
            active_clusters=int(active_clusters)
            if isinstance(active_clusters, (int, float))
            else 0,
            noise_ratio=float(noise_ratio)
            if isinstance(noise_ratio, (int, float))
            else 0.0,
            model_name=model_name,
            raw=payload,
        )
