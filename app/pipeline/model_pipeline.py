from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
import xgboost

from app.experiment_tracking import AbstractExperimentTracker, MLFlowExperimentTracker
from app.model import PredictionResult, TrainingOutput
from ray.data import Dataset
from ray.train import Checkpoint, CheckpointConfig, RunConfig, ScalingConfig
from ray.train.xgboost import XGBoostTrainer


@dataclass
class ModelPipeline:
    experiment_tracker: AbstractExperimentTracker = field(
        default_factory=MLFlowExperimentTracker
    )

    def predict(self, df: pl.DataFrame, model: Any) -> PredictionResult:
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
                storage_path=f"{Path.cwd()}/ray/checkpoints",
                checkpoint_config=CheckpointConfig(
                    checkpoint_frequency=10,
                    num_to_keep=1,
                ),
            ),
        )

    def train(
        self, train_dataset: Dataset, valid_dataset: Dataset, run_id: str
    ) -> TrainingOutput:
        trainer = self.get_trainer(train_dataset, valid_dataset)

        result = trainer.fit()
        self.experiment_tracker.log_metrics(
            run_id,
            {
                "train_error": float(result.metrics["train-error"]),
                "valid_error": float(result.metrics["valid-error"]),
            },
        )
        return TrainingOutput(checkpoint=result.checkpoint.path)

    def get_model_from_checkpoint(self, checkpoint: str) -> xgboost.XGBModel:
        checkpoint = Checkpoint.from_directory(checkpoint)
        try:
            return XGBoostTrainer.get_model(checkpoint)
        except Exception:  # noqa: BLE001
            msg = "No checkpoint retrieved"
            raise ValueError(msg)  # noqa: B904
