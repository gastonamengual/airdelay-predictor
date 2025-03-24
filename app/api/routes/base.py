from pathlib import Path

from fastapi import APIRouter
from ray.train import Result

from app.database.database_connector import DatabaseConnector
from app.model import Flight, PredictionResult
from app.model_pipeline import ModelPipeline

router = APIRouter()


@router.get("/")
async def root() -> dict[str, str]:
    return {"message": "access the /detect_objects endpoint"}


@router.post("/model/train")
async def train() -> Result:
    database_connector = DatabaseConnector()
    result = database_connector.get_data()
    pipeline = ModelPipeline()
    return pipeline.train(result)


@router.post("/model/predict")
async def predict(
    flight: Flight,
) -> PredictionResult:
    checkpoint_path = f"{Path.cwd()}/app/model_data/app/model_data/XGBoostTrainer_2025-03-24_17-01-53/XGBoostTrainer_4d9b0_00000_0_2025-03-24_17-01-53/checkpoint_000001"  # noqa: E501
    pipeline = ModelPipeline()
    return pipeline.predict(flight, checkpoint_path)
