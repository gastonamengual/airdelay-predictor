from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Column, Engine, Integer, String, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.model import Flight

Base = declarative_base()


class FlightsTable(Base):  # type: ignore
    __tablename__ = "flights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    airline = Column(String)
    flight_number = Column(Integer)
    origin = Column(String)
    destination = Column(String)
    departure_time = Column(String)
    weather = Column(String)
    congestion_level = Column(String)
    day_of_week = Column(String)
    delay = Column(Integer)


@dataclass
class DatabaseConnector:
    engine_url: str = "sqlite:///flights.db"
    table_name: str = "flights"
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
            session.bulk_insert_mappings(FlightsTable, data)

    def load(self) -> list[Flight]:
        with self.session() as session:
            return [
                Flight.model_validate(flight.__dict__)
                for flight in session.query(FlightsTable).all()
            ]
