from typing import Any

import polars as pl
import pytest

from app.pipeline import PreprocessingPipeline


@pytest.fixture
def sample_data() -> list[tuple[Any, ...]]:
    return [
        ("D", 2612, "MIA", "ORD", "2025-04-03 13:00:00", "Clear", "Low", "Sunday", "1"),
        ("A", 100, "JFK", "LAX", "2025-04-04 15:30:00", "Rainy", "High", "Monday", "0"),
    ]


@pytest.fixture
def expected_data() -> list[tuple[Any, ...]]:
    return [
        (0, 2612, 0, 0, 0, 0, 0, 1, 13, 0, 4, 4, 2025),
        (1, 100, 1, 1, 1, 1, 1, 0, 15, 30, 5, 4, 2025),
    ]


@pytest.fixture
def columns() -> list[str]:
    return [
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
def expected_columns() -> list[str]:
    return [
        "airline",
        "flight_number",
        "origin",
        "destination",
        "weather",
        "congestion_level",
        "day_of_week",
        "delay",
        "hour",
        "minute",
        "weekday",
        "month",
        "year",
    ]


@pytest.fixture
def input_df(sample_data: list[tuple[Any, ...]], columns: list[str]) -> pl.DataFrame:
    return pl.DataFrame(sample_data, schema=columns)


@pytest.fixture
def expected_df(
    expected_data: list[tuple[Any, ...]], expected_columns: list[str]
) -> pl.DataFrame:
    return pl.DataFrame(expected_data, expected_columns)


@pytest.fixture
def preprocessing_pipeline() -> PreprocessingPipeline:
    return PreprocessingPipeline()
