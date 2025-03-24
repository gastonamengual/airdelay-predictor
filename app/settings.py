from dataclasses import dataclass
from pathlib import Path


@dataclass
class _Settings:
    model_path: str = f"{Path.cwd()}/app/model_data/app/model_data/XGBoostTrainer_2025-03-24_17-01-53/XGBoostTrainer_4d9b0_00000_0_2025-03-24_17-01-53/checkpoint_000001"  # noqa: E501


settings = _Settings()
