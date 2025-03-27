import os

from src.utils.config import Config

from loguru import logger
import sys


# # init logger with custom config
class Logger:
    """
    Logger class to create a logger session
    """
    @staticmethod
    def create_sess(env):
        """
        Create a logger session
        Args:
            env (str): Environment
        Returns:
            Logger: Logger object
        """
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
    """
    GetConfig class to get the config and logger
    """
    def __init__(self) -> None:
        self.config_root_dir = "configs"
        self.config = Config(self.config_root_dir)
        self.logger = Logger.create_sess(os.getenv("ENVIRONMENT", self.config.whisper.logger.environment))
        self.logger.info(
            f"Logger for environemnt: {str(os.getenv('ENVIRONMENT', self.config.whisper.logger.environment))}"
        )
    
    def run(self) -> Logger:
        """
        Run the logger and config
        Returns:
            Logger: Logger object
        """
        return self.logger, self.config
        
logger, config = GetConfig().run()