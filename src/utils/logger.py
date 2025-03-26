import os

from src.utils.config import Config

from loguru import logger
import sys


# # init logger with custom config
class Logger:
    @staticmethod
    def create_sess(env):
        logger.remove()
        logger.add(
            "logs/server.log",
            format="{time} {level} {message}",
            rotation="10 MB",
            compression="zip",
            serialize=True,
        )
        env_dict = {"dev": "DEBUG", "prod": "INFO"}
        logger.add(sys.stderr, level=env_dict[env])
        return logger

class GetConfig:

    def __init__(self) -> None:
        self.config_root_dir = "configs"
        self.config = Config(self.config_root_dir)
        self.logger = Logger.create_sess(os.getenv("ENVIRONMENT", self.config.whisper.logger.environment))
        self.logger.info(
            f"Logger for environemnt: {str(os.getenv('ENVIRONMENT', self.config.whisper.logger.environment))}"
        )
    
    def run(self) -> Logger:
        return self.logger, self.config
        
logger, config = GetConfig().run()