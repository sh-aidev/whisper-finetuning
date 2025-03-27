from src.utils.logger import logger
from src.core.training import WhisperTrainer
class App:
    def __init__(self) -> None:
        self.trainer = WhisperTrainer()
        logger.info("App initialized...")

    def run(self):
        logger.info("App running...")
        self.trainer.train()
        logger.info("App completed...")