from transformers import WhisperForConditionalGeneration
from src.utils.config import Config

class WhisperModel:
    @staticmethod
    def load_model(config: Config) -> WhisperForConditionalGeneration:
        model = WhisperForConditionalGeneration.from_pretrained(config.whisper.model.model_name) 
        model.generation_config.language = config.whisper.model.language.lower()
        model.generation_config.task = config.whisper.task_type

        model.generation_config.forced_decoder_ids = None

        return model