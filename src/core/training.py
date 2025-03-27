import evaluate
from transformers import Seq2SeqTrainer
from src.utils.logger import logger, config
from src.core.data import PreProcessData, DataCollatorSpeechSeq2SeqWithPadding
from src.core.whisper import WhisperModel
from src.core.evaluate import WhisperEvaluation
from src.utils.utils import get_train_args, print_console_table, ForceSaveCallback

class WhisperTrainer():
    def __init__(self)-> None:
        audio_data_obj = PreProcessData(config)
        self.tokenizer = audio_data_obj.tokenizer
        processor = audio_data_obj.processor
        audio_data = audio_data_obj.process_data()
        model = WhisperModel().load_model(config)
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id,
        )
        self.metric = evaluate.load("wer")
        training_args = get_train_args(config)
        self.trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=audio_data["train"],
            eval_dataset=audio_data["test"],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=processor.feature_extractor,
            callbacks=[ForceSaveCallback()],
        )
        self.eval_obj = WhisperEvaluation(config)
        logger.info("Whisper Trainer Initialized")

    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    
    def train(self):
        # logger.info("Evaluation Before Training Starting...")
        # wer_before = self.eval_obj.evaluate()
        # logger.debug(f"WER Before Training: {wer_before}")
        # wer_before = ['mozilla-foundation/common_voice_17_0', '138.47']
        # print_console_table([wer_before], title="🚀 WER Matrix Before Training")
        # logger.info("Evaluation Before Training Completed...")
        logger.info("Training Starting...")
        self.trainer.train()
        logger.info("Training Completed...")
        # logger.info("Evaluation After Training Starting...")
        # # wer_after = self.eval_obj.evaluate()
        # wer_after = ['mozilla-foundation/common_voice_17_0', '130.47']
        # print_console_table([wer_after], title="🚀 WER Matrix After Training")

        # logger.info("Evaluation After Training Completed...")
        # wer_diff = float(wer_before[1]) - float(wer_after[1])
        # if wer_diff < 0:
        #     logger.info(f"WER Reduced by: {wer_diff}%")
        # else:
        #     logger.info(f"WER Improved by: {wer_diff}%")