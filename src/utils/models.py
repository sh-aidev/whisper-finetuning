from pydantic import BaseModel


class DataModel(BaseModel):
    """
    Data model for the configuration file
    """
    data_name: str          # Name of the data
    language: str           # Language of the data
    streaming: bool         # Streaming data
    no_of_test_samples: int # Number of test samples

class Whisper(BaseModel):
    """
    Whisper model for the configuration file
    """
    model_name: str         # Name of the model
    language: str           # Language of the model

class Paths(BaseModel):
    """
    Paths model for the configuration file
    """
    model_save_path: str    # Path to save the model
    # data_path: str

class LoggerModel(BaseModel):
    """
    Logger model for the configuration file
    """
    environment: str        # Environment to run the model

class TrainingConfigModel(BaseModel):
    """
    Training configuration model for the configuration file
    """
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    max_steps: int
    gradient_checkpointing: bool
    fp16: bool
    evaluation_strategy: str
    per_device_eval_batch_size: int
    predict_with_generate: bool
    generation_max_length: int
    save_steps: int
    eval_steps: int
    logging_steps: int
    report_to: str
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool
    push_to_hub: bool

class ConfigModel(BaseModel):
    """
    Configuration model for the configuration file
    """
    task_name: str          # Name of the task
    task_type: str          # Type of the task
    data: DataModel         # Data model
    model: Whisper          # Whisper model for training
    paths: Paths            # Paths for the model
    logger: LoggerModel     # Logger model
    training: TrainingConfigModel