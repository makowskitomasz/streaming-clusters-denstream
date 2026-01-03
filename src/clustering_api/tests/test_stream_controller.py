from fastapi.testclient import TestClient

from clustering_api.src.app import create_app


def test_pause_endpoint_sets_paused_state():
    # Arrange
    client = TestClient(create_app())

    # Act
    pause_response = client.post("/v1/stream/pause")
    state_response = client.get("/v1/stream/state")

    # Assert
    assert pause_response.status_code == 200
    assert state_response.status_code == 200
    assert state_response.json()["paused"] is True


def test_reset_clears_paused_state():
    # Arrange
    client = TestClient(create_app())
    client.post("/v1/stream/pause")

    # Act
    reset_response = client.post("/v1/stream/reset")
    state_response = client.get("/v1/stream/state")

    # Assert
    assert reset_response.status_code == 200
    assert state_response.status_code == 200
    assert state_response.json()["paused"] is False
