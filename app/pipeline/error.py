from dataclasses import dataclass

from fastapi import status


@dataclass
class NoCheckpointRetrieved(Exception):
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    message: str = "No checkpoint retrieved. Was your model trained?"
