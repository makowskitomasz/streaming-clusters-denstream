from frontend.api_client import MetricsLatestResponse


def test_metrics_from_payload_with_missing_fields() -> None:
    payload = {"latest": {"denstream": {"noise_ratio": 0.2}}}
    metrics = MetricsLatestResponse.from_payload(payload)

    assert metrics.model_name == "denstream"
    assert metrics.noise_ratio == 0.2
    assert metrics.active_clusters is None
    assert metrics.silhouette_score is None
