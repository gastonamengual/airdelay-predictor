from dataclasses import dataclass

import mlflow
from mlflow import xgboost
from mlflow.models.model import ModelInfo

from app.clients import MLFlowClient
from app.model import MLModel

from .config import ModelRegistryConfig


@dataclass
class MLFlowModelRegistry:
    def model_uri(self, model_name: str, model_version: str) -> str:
        return f"models:/{model_name}/{model_version}"

    @property
    def client(self) -> mlflow.MlflowClient:
        return MLFlowClient().client

    def load(self, config: ModelRegistryConfig) -> MLModel | None:
        model_version = config.model_version or self.get_latest_version(
            config.model_name
        )
        if not model_version:
            raise ValueError

        model_uri = self.model_uri(
            model_name=config.model_name, model_version=model_version
        )
        return self.load_model(model_uri)

    def create_model(
        self, model: MLModel, config: ModelRegistryConfig
    ) -> ModelRegistryConfig:
        model_info: ModelInfo = xgboost.log_model(
            xgb_model=model,
            artifact_path=f"{config.model_name}",
            registered_model_name=config.model_name,
        )
        return config.model_copy(
            update={
                "model_version": model_info.__dict__["_registered_model_version"],
                "artifact_path": model_info.__dict__["artifact_path"],
                "model_uri": model_info.__dict__["model_uri"],
                "run_id": model_info.__dict__["run_id"],
            }
        )

    def load_model(self, model_uri: str) -> MLModel | None:
        return xgboost.load_model(model_uri)  # type: ignore

    def get_latest_version(self, model_name: str) -> str:
        versions = self.client.get_latest_versions(model_name)
        return versions[-1].version  # type: ignore
