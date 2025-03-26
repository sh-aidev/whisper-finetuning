from src.utils.logger import logger

class App:
    def __init__(self) -> None:
        logger.info("App initialized...")

    def run(self):
        logger.info("App running...")