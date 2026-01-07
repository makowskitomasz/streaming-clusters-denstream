from __future__ import annotations

import sys
from collections import deque
from collections.abc import Deque, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypedDict

from loguru import logger

LOG_FORMAT: str = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)


# If you want stricter than `object`, you can replace object with a JSON-like union.
LogExtra = Mapping[str, object]


class LogPayload(TypedDict, total=False):
    timestamp: str
    level: str
    message: str


class _LoguruRecord(Protocol):
    def __getitem__(self, key: str) -> object: ...
    def get(self, key: str, default: object = ...) -> object: ...


class _LoguruMessage(Protocol):
    record: _LoguruRecord


@dataclass
class LoggingState:
    configured: bool = False
    recent_logs: Deque[LogPayload] = field(default_factory=lambda: deque(maxlen=1000))


_STATE = LoggingState()


def configure_logger(level: str = "INFO") -> None:
    """Configure Loguru console logger with colored output."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        colorize=True,
        format=LOG_FORMAT,
        backtrace=True,
        diagnose=True,
    )


def init_logging(
    *,
    log_dir: str | Path = "logs",
    filename: str = "clustering.jsonl",
    level: str = "INFO",
    reset: bool = False,
) -> Path:
    """Configure structured logging to a rotating JSONL file."""
    if _STATE.configured and not reset:
        return Path(log_dir) / filename

    logger.remove()
    configure_logger(level=level)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_path = log_path / filename

    logger.add(
        str(file_path),
        level=level.upper(),
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        serialize=True,
        enqueue=True,
    )
    logger.add(_log_sink, level=level.upper())

    _STATE.configured = True
    return file_path


def log_info(message: str, **kwargs: object) -> None:
    logger.bind(**kwargs).info(message)


def log_warning(message: str, **kwargs: object) -> None:
    logger.bind(**kwargs).warning(message)


def log_error(message: str, **kwargs: object) -> None:
    logger.bind(**kwargs).error(message)


def get_recent_logs(limit: int = 200) -> list[LogPayload]:
    """Return recent logs, newest first."""
    if limit <= 0:
        return []
    logs = list(_STATE.recent_logs)
    return list(reversed(logs[-limit:]))


def _log_sink(message: _LoguruMessage) -> None:
    record = message.record

    t = record["time"]
    lvl = record["level"]
    msg = record["message"]

    # These runtime asserts both document expectations and help type narrowing.
    assert hasattr(t, "isoformat")
    assert hasattr(lvl, "name")
    assert isinstance(msg, str)

    payload: LogPayload = {
        "timestamp": t.isoformat(),  # type: ignore[call-arg]
        "level": lvl.name,           # type: ignore[attr-defined]
        "message": msg,
    }

    extra = record.get("extra", {})
    if isinstance(extra, Mapping):
        payload.update(extra)  # values are object; TypedDict is total=False

    _STATE.recent_logs.append(payload)
