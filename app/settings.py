from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings_(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        secrets_dir="secrets",
        extra="ignore",
    )

    MODEL_PATH: str = f"{Path.cwd()}/app/model_data/app/model_data/XGBoostTrainer_2025-03-24_17-01-53/XGBoostTrainer_4d9b0_00000_0_2025-03-24_17-01-53/checkpoint_000001"  # noqa: E501


Settings = Settings_()
