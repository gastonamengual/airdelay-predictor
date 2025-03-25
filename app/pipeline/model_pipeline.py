from dataclasses import dataclass

import polars as pl
import xgboost
from ray.data import Dataset
from ray.train import Checkpoint, CheckpointConfig, RunConfig, ScalingConfig
from ray.train.xgboost import XGBoostTrainer

from app.model import PredictionResult, TrainingOutput
from app.settings import Settings

from .error import NoCheckpointRetrieved


@dataclass
class ModelPipeline:
    def predict(self, df: pl.DataFrame) -> PredictionResult:
        checkpoint = Checkpoint.from_directory(Settings.MODEL_PATH)
        try:
            model = XGBoostTrainer.get_model(checkpoint)
        except Exception:  # noqa: BLE001
            raise NoCheckpointRetrieved  # noqa: B904

        dmatrix = xgboost.DMatrix(df)
        prediction = model.predict(dmatrix)
        return PredictionResult(prediction=float(prediction[0]))

    def get_trainer(
        self, train_dataset: Dataset, valid_dataset: Dataset
    ) -> XGBoostTrainer:
        return XGBoostTrainer(
            train_loop_per_worker=None,
            scaling_config=ScalingConfig(
                num_workers=2,
                use_gpu=False,
            ),
            label_column="delay",
            num_boost_round=20,
            params={
                "objective": "binary:logistic",
                "eval_metric": ["logloss", "error"],
            },
            datasets={"train": train_dataset, "valid": valid_dataset},
            run_config=RunConfig(
                checkpoint_config=CheckpointConfig(
                    checkpoint_frequency=10,
                    num_to_keep=1,
                )
            ),
        )

    def train(self, train_dataset: Dataset, valid_dataset: Dataset) -> TrainingOutput:
        trainer = self.get_trainer(train_dataset, valid_dataset)

        result = trainer.fit()
        return TrainingOutput(
            train_error=float(result.metrics["train-error"]),
            valid_error=float(result.metrics["valid-error"]),
            checkpoint_path=result.checkpoint.path,
        )
