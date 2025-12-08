import sys

from loguru import logger
LOG_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)


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


def log_info(message: str, **kwargs) -> None:
    logger.bind(**kwargs).info(message)


def log_warning(message: str, **kwargs) -> None:
    logger.bind(**kwargs).warning(message)


def log_error(message: str, **kwargs) -> None:
    logger.bind(**kwargs).error(message)
