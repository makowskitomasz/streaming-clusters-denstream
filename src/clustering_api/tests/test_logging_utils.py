from typing import List, Tuple

import pytest
from loguru import logger as loguru_logger

from clustering_api.src.utils import logging_utils


@pytest.fixture(autouse=True)
def restore_logger_state():
    """Ensure Loguru sinks do not leak between tests."""
    loguru_logger.remove()
    yield
    loguru_logger.remove()


def test_configure_logger_sets_up_colored_sink(monkeypatch):
    add_calls: List[Tuple[tuple, dict]] = []
    removed = {"value": False}

    def fake_remove():
        removed["value"] = True

    def fake_add(*args, **kwargs):
        add_calls.append((args, kwargs))
        return 1

    monkeypatch.setattr(logging_utils.logger, "remove", fake_remove)
    monkeypatch.setattr(logging_utils.logger, "add", fake_add)

    logging_utils.configure_logger(level="debug")

    assert removed["value"] is True, "Logger.remove should be called before reconfiguration"
    assert add_calls, "Logger.add should be invoked to set up stderr sink"
    args, kwargs = add_calls[0]
    assert "format" in kwargs and kwargs["format"] == logging_utils.LOG_FORMAT
    assert kwargs["level"] == "DEBUG"


def test_log_helpers_emit_expected_levels():
    records = []

    def sink(message):
        records.append(
            (
                message.record["level"].name,
                message.record["message"],
                dict(message.record["extra"]),
            )
        )

    logging_utils.logger.remove()
    logging_utils.logger.add(sink, level="INFO")

    logging_utils.log_info("hello", source="test-info")
    logging_utils.log_warning("careful", source="test-warn")
    logging_utils.log_error("boom", source="test-error")

    assert records == [
        ("INFO", "hello", {"source": "test-info"}),
        ("WARNING", "careful", {"source": "test-warn"}),
        ("ERROR", "boom", {"source": "test-error"}),
    ]
