from fastapi import APIRouter, Depends

from app.database.database_connector import DatabaseConnector
from app.model import Flight, PredictionResult, TrainingOutput
from app.pipeline import ModelPipeline, PreprocessingPipeline

router = APIRouter(prefix="/model", tags=["Model"])


@router.post("/train")
async def train(
    model_pipeline: ModelPipeline = Depends(ModelPipeline),
    preprocessing_pipeline: PreprocessingPipeline = Depends(PreprocessingPipeline),
    database_connector: DatabaseConnector = Depends(DatabaseConnector),
) -> TrainingOutput:
    flihgts = database_connector.load()
    train_dataset, valid_dataset, _ = preprocessing_pipeline.run(flihgts)
    return model_pipeline.train(train_dataset, valid_dataset)


@router.post("/predict")
async def predict(
    flight: Flight,
    model_pipeline: ModelPipeline = Depends(ModelPipeline),
    preprocessing_pipeline: PreprocessingPipeline = Depends(PreprocessingPipeline),
) -> PredictionResult:
    df = preprocessing_pipeline.run([flight], with_split=False)
    return model_pipeline.predict(df)
