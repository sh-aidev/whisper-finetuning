from transformers import WhisperForConditionalGeneration
from src.utils.config import Config

class WhisperModel:
    """
    WhisperModel class to load the model
    """
    @staticmethod
    def load_model(config: Config) -> WhisperForConditionalGeneration:
        """
        Load the model
        Args:
            config (Config): Config object
        Returns:
            WhisperForConditionalGeneration: Whisper model
        """
        model = WhisperForConditionalGeneration.from_pretrained(config.whisper.model.model_name) 
        model.generation_config.language = config.whisper.model.language.lower()
        model.generation_config.task = config.whisper.task_type

        model.generation_config.forced_decoder_ids = None

        return model