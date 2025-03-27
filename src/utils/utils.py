from transformers import Seq2SeqTrainingArguments, TrainerCallback
from src.utils.config import Config
import os
from typing import List, Any
from rich.console import Console
from rich.table import Table

def get_train_args(config: Config)-> Seq2SeqTrainingArguments:
    output_dir_name = config.whisper.model.model_name.split("/")[-1] + "-" + config.whisper.data.language
    output_dir = os.path.join(config.whisper.paths.model_save_path, output_dir_name)
    return Seq2SeqTrainingArguments(
    output_dir=output_dir,  # change to a repo name of your choice
    per_device_train_batch_size=config.whisper.training.per_device_train_batch_size,
    gradient_accumulation_steps=config.whisper.training.gradient_accumulation_steps,  # increase by 2x for every 2x decrease in batch size
    learning_rate=config.whisper.training.learning_rate,
    warmup_steps=config.whisper.training.warmup_steps,
    max_steps=config.whisper.training.max_steps,
    gradient_checkpointing=config.whisper.training.gradient_checkpointing,
    fp16=config.whisper.training.fp16,
    evaluation_strategy=config.whisper.training.evaluation_strategy,
    per_device_eval_batch_size=config.whisper.training.per_device_eval_batch_size,
    predict_with_generate=config.whisper.training.predict_with_generate,
    generation_max_length=config.whisper.training.generation_max_length,
    save_steps=config.whisper.training.save_steps,
    eval_steps=config.whisper.training.eval_steps,
    logging_steps=config.whisper.training.logging_steps,
    report_to=[config.whisper.training.report_to],
    load_best_model_at_end=config.whisper.training.load_best_model_at_end,
    metric_for_best_model=config.whisper.training.metric_for_best_model,
    greater_is_better=config.whisper.training.greater_is_better,
    push_to_hub=config.whisper.training.push_to_hub,
)


class ForceSaveCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.save_steps == 0:
            control.should_save = True
        return control
