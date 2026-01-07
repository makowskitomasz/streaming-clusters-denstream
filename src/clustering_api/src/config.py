from dataclasses import dataclass
from pathlib import Path

from pyaml_env import parse_config


class ClusteringConfig:
    @dataclass
    class App:
        server_port: int

    @dataclass
    class DenStream:
        decay_factor: float
        epsilon: float
        beta: float
        mu: float
        n_samples_init: int
        stream_speed: int

    def __init__(self, version: str, app: dict, denstream: dict) -> None:
        self.version = version
        self.app = ClusteringConfig.App(**app)
        self.denstream = ClusteringConfig.DenStream(**denstream)


current_dir = Path(__file__).resolve().parent
config_path = current_dir / ".." / "config.yaml"
config = ClusteringConfig(**parse_config(path=config_path))
