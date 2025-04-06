from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.model import Flight
from app.model_registry.config import ModelRegistryConfig

from .tables import AbstractTable, Base, FlightsTable, ModelsTable

TABLE_NAME_TO_BASE: dict[str, AbstractTable] = {
    "flights": FlightsTable,  # type: ignore
    "models": ModelsTable,  # type: ignore
}

TABLE_NAME_TO_OBJECT: dict[str, BaseModel] = {
    "flights": Flight,  # type: ignore
    "models": ModelRegistryConfig,  # type: ignore
}


@dataclass
class DatabaseConnector:
    engine_url: str
    table_name: str
    engine: Engine = field(init=False)

    def __post_init__(self) -> None:
        self.engine = create_engine(self.engine_url)

    @contextmanager
    def session(self) -> Generator[Session]:
        SessionLocal = sessionmaker(bind=self.engine)
        session = SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_table(self) -> None:
        Base.metadata.create_all(self.engine)

    def save(self, data: list[dict[Any, Any]]) -> None:
        with self.session() as session:
            session.bulk_insert_mappings(TABLE_NAME_TO_BASE[self.table_name], data)

    def load(self) -> list[Any]:
        with self.session() as session:
            return [
                TABLE_NAME_TO_OBJECT[self.table_name].model_validate(item.__dict__)
                for item in session.query(TABLE_NAME_TO_BASE[self.table_name]).all()
            ]

    def get_metadata_by_job_id(self, job_id: str) -> ModelRegistryConfig:
        with self.session() as session:
            metadata = (
                session.query(ModelsTable).filter(ModelsTable.job_id == job_id).first()
            )

            if metadata is None:
                msg = f"Metadata for job_id {job_id} not found."
                raise ValueError(msg)

            return ModelRegistryConfig.model_validate(metadata)
