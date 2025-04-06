from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings_(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        secrets_dir="secrets",
        extra="ignore",
    )

    MODEL_TRACKING_URI: str = Field(default="")

    MODEL_PATH: str = Field(
        default=f"{Path.cwd()}/app/model_data/XGBoostTrainer_2025-03-25_17-05-11/XGBoostTrainer_ee701_00000_0_2025-03-25_17-05-11/checkpoint_000001"
    )  # noqa: E501)

    FLIGHTS_DB: str = Field(default="")
    FLIGHTS_DB_TABLE: str = Field(default="flights")

    MODELS_DB: str = Field(default="")
    MODELS_DB_TABLE: str = Field(default="models")


Settings = Settings_()
