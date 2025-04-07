from typing import Annotated

from fastapi import APIRouter, Depends

from app.model import Flight, PredictionResult
from app.model_registry import MLFlowModelRegistry, ModelRegistryConfig
from app.pipeline import ModelPipeline, PreprocessingPipeline

router = APIRouter(prefix="/model/predict")


@router.post("/")
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
