import json

import httpx
import pytest

from frontend.api_client import ApiClient, BackendUnavailableError, StreamParams


def test_start_stream_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("http://example.com/stream/start")
        payload = json.loads(request.content.decode("utf-8"))
        assert payload == {
            "batch_size": 500,
            "drift_rate": 0.2,
            "update_interval_seconds": 3,
        }
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    client = ApiClient(base_url="http://example.com", transport=transport)
    params = StreamParams(batch_size=500, drift_rate=0.2, update_interval_seconds=3)

    response = client.start_stream(params)

    assert response == {"ok": True}


def test_missing_endpoint_raises():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Not Found"})

    transport = httpx.MockTransport(handler)
    client = ApiClient(base_url="http://example.com", transport=transport)

    with pytest.raises(BackendUnavailableError):
        client.reset_stream()
