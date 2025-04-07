from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import JSONResponse

import ray
import ray.exceptions
from app.async_processing import async_train
from app.database import DatabaseConnector
from app.experiment_tracking import AbstractExperimentTracker, MLFlowExperimentTracker
from app.model_registry import MLFlowModelRegistry, ModelRegistryConfig
from app.pipeline import ModelPipeline, PreprocessingPipeline
from app.settings import Settings

router = APIRouter(prefix="/model/train")


@router.post("/")
async def train(
    model_name: str,
    experiment_tracker: Annotated[
        AbstractExperimentTracker, Depends(MLFlowExperimentTracker)
    ],
    preprocessing_pipeline: Annotated[
        PreprocessingPipeline, Depends(PreprocessingPipeline)
    ],
    model_registry: Annotated[MLFlowModelRegistry, Depends(MLFlowModelRegistry)],
) -> Response:
    model_pipeline = ModelPipeline()
    job_id = async_train.remote(
        model_name,
        experiment_tracker,
        preprocessing_pipeline,
        model_pipeline,
        model_registry,
    )

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "message": f"Training job for model '{model_name}' started.",
            "job_id": job_id,
            "status_url": f"/train/status/{job_id}",
        },
    )


@router.get("/status/{job_id}")
async def get_job_status(job_id: str) -> JSONResponse:
    try:
        object_ref = ray.ObjectRef(bytes.fromhex(job_id))  # type: ignore
        ready_ids, _remaining_ids = ray.wait(object_ref, num_returns=1, timeout=None)  # type: ignore

        if ready_ids:
            result = ray.get(object_ref)  # type: ignore
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"status": "finished", "result": result},
            )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "running"},
        )
    except ray.exceptions.RayError:
        raise HTTPException(  # noqa: B904
            status_code=404, detail=f"Job {job_id} not found or invalid."
        )


@router.get("/training_results/{id}")
async def training_results(job_id: str) -> ModelRegistryConfig:
    database_connector = DatabaseConnector(
        engine_url=Settings.MODELS_DB, table_name=Settings.MODELS_DB_TABLE
    )
    metadata = database_connector.get_metadata_by_job_id(job_id)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Metadata for job {job_id} not found in the database.",
        )

    return metadata
