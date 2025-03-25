from pydantic import BaseModel


class TrainingOutput(BaseModel):
    train_error: float
    valid_error: float
    checkpoint_path: str
