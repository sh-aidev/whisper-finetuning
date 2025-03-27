from datasets import load_dataset, Audio, DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from src.utils.config import Config

class PreProcessData():
    def __init__(self, config: Config) -> None:
        self.config = config
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(config.whisper.model.model_name)
        self.tokenizer =  WhisperTokenizer.from_pretrained(config.whisper.model.model_name, language=config.whisper.model.language, task=config.whisper.task_type)
        self.processor = WhisperProcessor.from_pretrained(config.whisper.model.model_name, language=config.whisper.model.language, task=config.whisper.task_type)

    def process_data(self) -> Dict[str, (DatasetDict | Dataset | IterableDatasetDict | IterableDataset)]:
        split = ""
        if self.config.whisper.data.streaming:
            split = "train"
        else:
            split = "train+validation"
        

        train_data = load_dataset(self.config.whisper.data.data_name, self.config.whisper.data.language, split=split, streaming=self.config.whisper.data.streaming)
        train_data = train_data.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
        train_data = train_data.cast_column("audio", Audio(sampling_rate=16000))
        train_data = train_data.map(self.map_dataset)

        test_data = load_dataset(self.config.whisper.data.data_name, self.config.whisper.data.language, split="test", streaming=self.config.whisper.data.streaming)
        test_data = test_data.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
        if self.config.whisper.data.streaming:
            test_data = test_data.take(self.config.whisper.data.no_of_test_samples)
        test_data = test_data.cast_column("audio", Audio(sampling_rate=16000))
        test_data = test_data.map(self.map_dataset)
        
        return {
            "train": train_data,
            "test": test_data
        }

    def map_dataset(self, batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids 
        batch["labels"] = self.tokenizer(batch["sentence"]).input_ids
        return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch