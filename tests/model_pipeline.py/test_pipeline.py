import polars as pl
import pytest

from app.model_pipeline.pipeline import ModelPipeline

# Sample input data
sample_data = [
    ("Delta", 2612, "MIA", "ORD", "2025-04-03 13:00:00", "Clear", "Low", "Sunday", "1"),
    (
        "American",
        100,
        "JFK",
        "LAX",
        "2025-04-04 15:30:00",
        "Rainy",
        "High",
        "Monday",
        "0",
    ),
]

# Column names based on the data
columns = [
    "airline",
    "flight_number",
    "origin",
    "destination",
    "departure_time",
    "weather",
    "congestion_level",
    "day_of_week",
    "delay",
]


@pytest.fixture
def input_df() -> pl.DataFrame:
    return pl.DataFrame(sample_data, schema=columns)


@pytest.fixture
def expected_df() -> pl.DataFrame:
    return pl.DataFrame([
        {
            "airline": 0,
            "flight_number": 2612,
            "origin": 0,
            "destination": 0,
            "weather": 0,
            "congestion_level": 0,
            "day_of_week": 0,
            "delay": "1",
            "hour": 13,
            "minute": 0,
            "weekday": 4,
            "month": 4,
            "year": 2025,
        },
        {
            "airline": 1,
            "flight_number": 100,
            "origin": 1,
            "destination": 1,
            "weather": 1,
            "congestion_level": 1,
            "day_of_week": 1,
            "delay": "0",
            "hour": 15,
            "minute": 30,
            "weekday": 5,
            "month": 4,
            "year": 2025,
        },
    ])


def test_preprocess(input_df: pl.DataFrame, expected_df: pl.DataFrame) -> None:
    pipeline = ModelPipeline()
    processed_df = pipeline._preprocess(input_df)  # noqa: SLF001

    assert processed_df.shape == expected_df.shape, "Shape mismatch"
    assert processed_df.columns == expected_df.columns, "Columns mismatch"
