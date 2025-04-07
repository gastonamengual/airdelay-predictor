from .base import router as home_router
from .exception_handlers import ERROR_TO_HANDLER_MAPPING
from .model_predict import router as model_predict_router
from .model_predict import router as model_router
from .model_train import router as model_train_router

routers = [
    home_router,
    model_router,
    model_train_router,
    model_predict_router,
]

__all__ = ["ERROR_TO_HANDLER_MAPPING", "routers"]
