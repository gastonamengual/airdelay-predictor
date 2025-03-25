from fastapi import Request
from fastapi.responses import JSONResponse

from app.pipeline import (
    NoCheckpointRetrieved,
)


def no_checkpoint_exception_handler(
    request: Request,  # noqa: ARG001
    exc: NoCheckpointRetrieved,
) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"message": exc.message})


ERROR_TO_HANDLER_MAPPING = [
    [NoCheckpointRetrieved, no_checkpoint_exception_handler],
]
