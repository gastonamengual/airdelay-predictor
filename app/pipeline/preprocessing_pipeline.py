from dataclasses import dataclass

import polars as pl
import ray
from ray.data import Dataset
from ray.data.preprocessors import StandardScaler

from app.model.flight import Flight


@dataclass
class PreprocessingPipeline:
    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
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

    def split(self, df: pl.DataFrame) -> tuple[Dataset, Dataset, Dataset]:
        dataset = ray.data.from_pandas(df.to_pandas())
        train_dataset, valid_dataset = dataset.train_test_split(test_size=0.3)
        test_dataset = valid_dataset.drop_columns(["delay"])

        columns_to_scale = ["hour", "minute", "weekday", "month", "year"]

        preprocessor = StandardScaler(columns=columns_to_scale)
        preprocessor.fit(train_dataset)

        train_dataset = preprocessor.transform(train_dataset)
        valid_dataset = preprocessor.transform(valid_dataset)
        return train_dataset, valid_dataset, test_dataset

    def run(
        self,
        flights: list[Flight],
        with_split: bool = True,
    ) -> pl.DataFrame | tuple[Dataset, Dataset, Dataset]:
        df = pl.DataFrame(flight.model_dump() for flight in flights)
        preprocessed_df = self.preprocess(df)
        if not with_split:
            return preprocessed_df
        return self.split(preprocessed_df)
