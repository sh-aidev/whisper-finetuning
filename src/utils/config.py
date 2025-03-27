import toml
import os
from src.utils.models import ConfigModel

class Config:
    """
    Config class to load the config file
    """
    def __init__(self, root_config_path: str):
        """
        Load the config file
        Args:
            root_config_path (str): Path to the config file
        """
        self.whisper = ConfigModel(
            **toml.load(os.path.join(root_config_path, "config.toml"))
        )