import httpx
import pytest
from api_client import ApiClient, BackendError, StreamParams


def test_start_stream_payload():
    # Arrange
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("http://example.com/v1/stream/start")
        assert request.content == b""
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    client = ApiClient(base_url="http://example.com", transport=transport)
    params = StreamParams(batch_size=500, drift_rate=0.2, update_interval_seconds=3)
    # Act
    response = client.start_stream(params)
    # Assert
    assert response == {"ok": True}


def test_missing_endpoint_raises():
    # Arrange
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Not Found"})

    transport = httpx.MockTransport(handler)
    client = ApiClient(base_url="http://example.com", transport=transport)
    # Act
    with pytest.raises(BackendError):
        # Assert
        client.reset_stream()


def test_next_batch():
    # Arrange
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "batch_id": 3,
                "points": [{"x": 1.0, "y": 2.0, "cluster_id": "1"}],
            },
        )

    transport = httpx.MockTransport(handler)
    client = ApiClient(base_url="http://example.com", transport=transport)
    params = StreamParams(batch_size=50, drift_rate=0.1, update_interval_seconds=2)
    # Act
    response = client.next_batch(params)
    # Assert
    assert response.batch_id == 3
    assert len(response.points) == 1
    assert calls[0].endswith("/v1/stream/next")


def test_get_latest_metrics_parsing():
    # Arrange
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "latest": {
                    "denstream": {
                        "silhouette_score": 0.4,
                        "number_of_clusters": 3,
                        "noise_ratio": 0.1,
                    },
                },
            },
        )

    transport = httpx.MockTransport(handler)
    client = ApiClient(base_url="http://example.com", transport=transport)
    # Act
    metrics = client.get_latest_metrics()
    # Assert
    assert metrics.model_name == "denstream"
    assert metrics.active_clusters == 3
    assert metrics.noise_ratio == 0.1


def test_get_recent_logs_parsing():
    # Arrange
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "logs": [
                    {"timestamp": "t1", "message": "m1", "latency_ms": 12.3},
                    {"timestamp": "t2", "message": "m2", "latency_ms": 10.0},
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    client = ApiClient(base_url="http://example.com", transport=transport)
    # Act
    logs = client.get_recent_logs(limit=2)
    # Assert
    assert len(logs) == 2
    assert logs[0].message == "m1"
