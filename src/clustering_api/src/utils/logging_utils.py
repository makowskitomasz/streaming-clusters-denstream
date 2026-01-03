import sys
from pathlib import Path

from loguru import logger

LOG_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)

_LOGGING_CONFIGURED = False


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
    _LOGGING_CONFIGURED = True
    return file_path


def log_info(message: str, **kwargs) -> None:
    logger.bind(**kwargs).info(message)


def log_warning(message: str, **kwargs) -> None:
    logger.bind(**kwargs).warning(message)


def log_error(message: str, **kwargs) -> None:
    logger.bind(**kwargs).error(message)
