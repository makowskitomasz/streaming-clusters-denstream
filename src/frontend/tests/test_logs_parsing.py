from api_client import LogRecord


def test_log_record_parses_missing_fields() -> None:
    # Arrange
    payload = {"timestamp": "2024-01-01T00:00:00", "message": "ok"}
    # Act
    record = LogRecord.from_payload(payload)
    # Assert
    assert record.timestamp == "2024-01-01T00:00:00"
    assert record.message == "ok"
    assert record.active_clusters is None
