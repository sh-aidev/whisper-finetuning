from src.utils.config import Config
from transformers import pipeline
from datasets import load_dataset, Audio
import evaluate
from src.utils.logger import logger

class WhisperEvaluation():
    def __init__(self, config: Config) -> None:
        self.config = config
        self.whisper_asr = pipeline(
            "automatic-speech-recognition",
            model=config.whisper.model.model_name,
            device = 0
        )
        logger.debug("Pipeline loaded...")
        # self.whisper_asr.model.config.suppress_tokens.remove(6)
        # self.whisper_asr.model.config.suppress_tokens.remove(12)

        self.metric = evaluate.load("wer")
        self.whisper_norm = self.whisper_asr.tokenizer._normalize
        dataset = load_dataset(config.whisper.data.data_name, config.whisper.data.language, revision="streaming", split="test", streaming=True)
        logger.debug(f"Dataset loaded: {config.whisper.data.data_name}")

        # only for debugging, restricts the number of rows to numeric value in brackets
        # dataset = dataset.take(config.whisper.data.no_of_test_samples)

        # resample to 16kHz
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

        # normalise references
        dataset = dataset.map(self.normalise)

        # remove any empty references
        self.dataset = dataset.filter(self.is_target_text_in_range, input_columns=["norm_text"])

        self.batch_size = 16
        logger.debug("Whisper Evaluation Initialized...")

    def evaluate(self):
        logger.info("Evaluation Starting...")
        predictions = []
        references = []

        # run streamed inference
        for out in self.whisper_asr(self.data(self.dataset), batch_size=self.batch_size, generate_kwargs={"language": self.config.whisper.model.language.lower()}):
            predictions.append(self.whisper_norm(out["text"]))
            references.append(out["reference"][0])
        # compute the WER
        wer = self.metric.compute(references=references, predictions=predictions)
        wer = round(100 * wer, 2)
        logger.info(f"WER: {wer}")
        logger.info("Evaluation Completed...")
        print([self.config.whisper.data.data_name, str(wer)])
        return [self.config.whisper.data.data_name, str(wer)]
    
    def is_target_text_in_range(self, ref):
        if ref.strip() == "ignore time segment in scoring":
            return False
        else:
            return ref.strip() != ""
    
    
    def data(self, dataset):
        for i, item in enumerate(dataset):
            yield {**item["audio"], "reference": item["norm_text"]}
    
    def normalise(self, batch):
        batch["norm_text"] = self.whisper_norm(self.get_text(batch))
        return batch
    
    def get_text(self, sample):
        if "text" in sample:
            return sample["text"]
        elif "sentence" in sample:
            return sample["sentence"]
        elif "normalized_text" in sample:
            return sample["normalized_text"]
        elif "transcript" in sample:
            return sample["transcript"]
        else:
            raise ValueError(f"Sample: {sample.keys()} has no transcript.")