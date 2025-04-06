from pydantic import BaseModel


class TrainingOutput(BaseModel):
    checkpoint: str
