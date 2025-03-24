from dataclasses import dataclass
from typing import Any

import polars as pl
import ray
import xgboost
from ray.data import Dataset
from ray.data.preprocessors import StandardScaler
from ray.train import Checkpoint, CheckpointConfig, Result, RunConfig, ScalingConfig
from ray.train.xgboost import XGBoostTrainer

from app.model import Flight, PredictionResult


@dataclass
class ModelPipeline:
    def predict(self, flight: Flight, checkpoint_path: str) -> PredictionResult:
        checkpoint = Checkpoint.from_directory(checkpoint_path)
        try:
            model = XGBoostTrainer.get_model(checkpoint)
        except Exception:  # noqa: BLE001
            msg = "No checkpoint retrieved. Was your model trained?"
            raise ValueError(msg)  # noqa: B904

        df = pl.DataFrame([flight.model_dump()])
        df = self._preprocess(df)
        dmatrix = xgboost.DMatrix(df)
        prediction = model.predict(dmatrix)
        return PredictionResult(prediction=float(prediction[0]))

    def _preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            pl.col("departure_time")
            .str.to_datetime("%Y-%m-%d %H:%M:%S")
            .alias("departure_time")
        )
        df = df.with_columns([
            pl.col("departure_time").dt.hour().alias("hour"),
            pl.col("departure_time").dt.minute().alias("minute"),
            pl.col("departure_time").dt.weekday().alias("weekday"),
            pl.col("departure_time").dt.month().alias("month"),
            pl.col("departure_time").dt.year().alias("year"),
        ])
        df = df.drop("departure_time")
        categorical_columns = [
            "airline",
            "origin",
            "destination",
            "weather",
            "congestion_level",
            "day_of_week",
        ]

        df = df.with_columns([
            df[col].cast(pl.Categorical(ordering="lexical"))
            for col in categorical_columns
        ])
        return df.with_columns(df.select(pl.all().to_physical()))

    def _split(self, df: pl.DataFrame) -> tuple[Dataset, Dataset, Dataset]:
        dataset = ray.data.from_pandas(df.to_pandas())
        train_dataset, valid_dataset = dataset.train_test_split(test_size=0.3)
        test_dataset = valid_dataset.drop_columns(["delay"])

        columns_to_scale = ["hour", "minute", "weekday", "month", "year"]

        preprocessor = StandardScaler(columns=columns_to_scale)
        preprocessor.fit(train_dataset)

        train_dataset = preprocessor.transform(train_dataset)
        valid_dataset = preprocessor.transform(valid_dataset)
        return train_dataset, valid_dataset, test_dataset

    def train(self, result: list[tuple[Any, ...]]) -> Result:
        df = pl.DataFrame(result)
        preprocessed_df = self._preprocess(df)
        train_dataset, valid_dataset, _test_dataset = self._split(preprocessed_df)

        trainer = XGBoostTrainer(
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

        return trainer.fit()
