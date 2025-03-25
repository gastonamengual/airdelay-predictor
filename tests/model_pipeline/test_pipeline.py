import polars as pl

from app.pipeline import PreprocessingPipeline


def test_preprocess(
    input_df: pl.DataFrame,
    expected_df: pl.DataFrame,
    preprocessing_pipeline: PreprocessingPipeline,
) -> None:
    processed_df = preprocessing_pipeline.preprocess(input_df)

    assert processed_df.shape == expected_df.shape
    assert processed_df.columns == expected_df.columns
