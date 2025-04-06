from typing import Annotated, Any

from fastapi import APIRouter, Depends

import ray
from app.async_processing import async_train
from app.database import DatabaseConnector
from app.experiment_tracking import AbstractExperimentTracker, MLFlowExperimentTracker
from app.model import Flight, PredictionResult
from app.model_registry import MLFlowModelRegistry, ModelRegistryConfig
from app.pipeline import ModelPipeline, PreprocessingPipeline
from app.settings import Settings

router = APIRouter(prefix="/model", tags=["Model"])


@router.post("/train")
async def train(
    model_name: str,
    experiment_tracker: Annotated[
        AbstractExperimentTracker, Depends(MLFlowExperimentTracker)
    ],
    preprocessing_pipeline: Annotated[
        PreprocessingPipeline, Depends(PreprocessingPipeline)
    ],
    model_registry: Annotated[MLFlowModelRegistry, Depends(MLFlowModelRegistry)],
) -> Any:
    model_pipeline = ModelPipeline()
    job_id = async_train.remote(
        model_name,
        experiment_tracker,
        preprocessing_pipeline,
        model_pipeline,
        model_registry,
    )

    return {
        "message": f"Training job for model '{model_name}' started.",
        "job_id": job_id.hex(),
        "status_url": f"/train/status/{job_id.hex()}",
    }


@router.get("/train/status/{job_id}")
async def get_job_status(job_id: str) -> dict[str, str]:
    object_ref = ray.ObjectRef(bytes.fromhex(job_id))  # type: ignore
    if ray.wait([object_ref], timeout=0.1, num_returns=1)[0]:  # type: ignore
        result = ray.get(object_ref)  # type: ignore
        return {"status": "finished", "result": result}
    return {"status": "running"}


@router.get("/train/{id}/download")
async def download_training(job_id: str) -> ModelRegistryConfig:
    database_connector = DatabaseConnector(
        engine_url=Settings.MODELS_DB, table_name=Settings.MODELS_DB_TABLE
    )
    return database_connector.get_metadata_by_job_id(job_id)


@router.post("/predict")
async def predict(
    model_name: str,
    model_version: str,
    flight: Flight,
    model_registry: Annotated[MLFlowModelRegistry, Depends(MLFlowModelRegistry)],
    model_pipeline: Annotated[ModelPipeline, Depends(ModelPipeline)],
    preprocessing_pipeline: Annotated[
        PreprocessingPipeline, Depends(PreprocessingPipeline)
    ],
) -> PredictionResult:
    df = preprocessing_pipeline.run([flight], with_split=False)
    config = ModelRegistryConfig(
        model_name=model_name,
        model_version=model_version,
    )
    model = model_registry.load(config)
    return model_pipeline.predict(df, model)
