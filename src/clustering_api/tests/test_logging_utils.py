from pathlib import Path

from clustering_api.src.utils.logging_utils import init_logging


def test_init_logging_creates_log_dir(tmp_path: Path):
    # Arrange
    log_dir = tmp_path / "logs"

    # Act
    log_path = init_logging(log_dir=log_dir, reset=True)

    # Assert
    assert log_dir.exists()
    assert log_path.parent == log_dir
