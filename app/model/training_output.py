from dataclasses import dataclass


@dataclass
class TrainingOutput:
    train_error: float
    valid_error: float
    checkpoint_path: str
