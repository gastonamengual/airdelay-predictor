from typing import Any, Protocol

from app.model import MLModel

from .config import ModelRegistryConfig


class ModelRegistry(Protocol):
    def load(self, config: ModelRegistryConfig) -> MLModel | None: ...

    def create_model(
        self, model: MLModel, config: ModelRegistryConfig
    ) -> ModelRegistryConfig: ...

    def load_model(self, model_uri: str) -> Any: ...

    def get_latest_version(self, model_name: str) -> str: ...
