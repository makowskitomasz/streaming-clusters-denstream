from api_client import MetricsLatestResponse


def test_metrics_from_payload_with_missing_fields() -> None:
    # Arrange
    payload = {"latest": {"denstream": {"noise_ratio": 0.2}}}
    # Act
    metrics = MetricsLatestResponse.from_payload(payload)
    # Assert
    assert metrics.model_name == "denstream"
    assert metrics.noise_ratio == 0.2
    assert metrics.active_clusters is None
    assert metrics.silhouette_score is None
