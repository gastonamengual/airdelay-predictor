from fastapi import APIRouter

from app.database.database_connector import DatabaseConnector
from app.model import Flight, PredictionResult
from app.model.training_output import TrainingOutput
from app.model_pipeline import ModelPipeline

router = APIRouter()


@router.get("/")
async def root() -> dict[str, str]:
    return {"message": "access the /train or /predict endpoint"}


@router.post("/model/train")
async def train() -> TrainingOutput:
    database_connector = DatabaseConnector()
    data = database_connector.get_data()
    pipeline = ModelPipeline()
    return pipeline.train(data)


@router.post("/model/predict")
async def predict(
    flight: Flight,
) -> PredictionResult:
    pipeline = ModelPipeline()
    return pipeline.predict(flight)
