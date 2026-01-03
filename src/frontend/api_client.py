from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


class BackendUnavailableError(RuntimeError):
    """Raised when the backend is unreachable or returns an error response."""


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
        return self._request("POST", "/stream/start", json=params.to_payload())

    def pause_stream(self) -> dict[str, object]:
        return self._request("POST", "/stream/pause")

    def reset_stream(self) -> dict[str, object]:
        return self._request("POST", "/stream/reset")

    def next_batch(self, params: StreamParams) -> dict[str, object]:
        return self._request("POST", "/stream/next", json=params.to_payload())

    def _request(
        self, method: str, path: str, json: dict[str, float | int] | None = None
    ) -> dict[str, object]:
        try:
            response = self._client.request(method, path, json=json)
        except httpx.HTTPError as exc:
            raise BackendUnavailableError(str(exc)) from exc
        if response.status_code == 404:
            raise BackendUnavailableError(f"Endpoint not found: {path}")
        if response.is_error:
            raise BackendUnavailableError(
                f"Backend error {response.status_code}: {response.text}"
            )
        try:
            payload = response.json()
        except ValueError:
            return {}
        if isinstance(payload, dict):
            return payload
        return {}
