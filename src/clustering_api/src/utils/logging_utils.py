import sys
from collections import deque
from pathlib import Path

from loguru import logger

LOG_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)

_LOGGING_CONFIGURED = False
_RECENT_LOGS: deque[dict[str, object]] = deque(maxlen=1000)


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
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED and not reset:
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
    _LOGGING_CONFIGURED = True
    return file_path


def log_info(message: str, **kwargs) -> None:
    logger.bind(**kwargs).info(message)


def log_warning(message: str, **kwargs) -> None:
    logger.bind(**kwargs).warning(message)


def log_error(message: str, **kwargs) -> None:
    logger.bind(**kwargs).error(message)


def get_recent_logs(limit: int = 200) -> list[dict[str, object]]:
    """Return recent logs, newest first."""
    if limit <= 0:
        return []
    logs = list(_RECENT_LOGS)
    return list(reversed(logs[-limit:]))


def _log_sink(message) -> None:
    record = message.record
    payload: dict[str, object] = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
    }
    extra = record.get("extra", {})
    if isinstance(extra, dict):
        payload.update(extra)
    _RECENT_LOGS.append(payload)
