import toml
import os
from src.utils.models import ConfigModel

class Config:
    def __init__(self, root_config_path: str):
        self.whisper = ConfigModel(
            **toml.load(os.path.join(root_config_path, "config.toml"))
        )