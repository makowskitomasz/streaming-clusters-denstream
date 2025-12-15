import os
from dataclasses import dataclass

from pyaml_env import parse_config


class ClusteringConfig:
    @dataclass
    class App:
        server_port: int

    def __init__(self, version, app):
        self.version = version
        self.app = ClusteringConfig.App(**app)

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '..', 'config.yaml')
config = ClusteringConfig(**parse_config(path=config_path))
