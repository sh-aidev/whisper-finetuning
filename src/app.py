from src.utils.logger import logger
from src.core.training import WhisperTrainer
class App:
    """
    Main Application class
    It initializes the WhisperTrainer and runs the training
    """
    def __init__(self) -> None:
        self.trainer = WhisperTrainer()
        logger.info("App initialized...")

    def run(self):
        logger.info("App running...")
        self.trainer.train()
        logger.info("App completed...")