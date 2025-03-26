from pydantic import BaseModel

class LoggerModel(BaseModel):
    environment: str


class ConfigModel(BaseModel):
    task_name: str
    model_save_path: str
    logger: LoggerModel