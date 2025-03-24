from dataclasses import dataclass
from typing import Any

from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine
from sqlalchemy.orm import sessionmaker


@dataclass
class DatabaseConnector:
    engine_url: str = "sqlite:///flights.db"
    table_name: str = "flights"

    def save_data(self, data: list[dict[Any, Any]]) -> None:
        engine = create_engine(self.engine_url)

        Session = sessionmaker(bind=engine)
        session = Session()

        metadata = MetaData()

        flights_table = Table(
            self.table_name,
            metadata,
            Column("airline", String),
            Column("flight_number", Integer),
            Column("origin", String),
            Column("destination", String),
            Column("departure_time", String),
            Column("weather", String),
            Column("congestion_level", String),
            Column("day_of_week", String),
            Column("delay", Integer),
        )

        metadata.create_all(engine)

        with engine.connect() as conn:
            try:
                conn.execute(flights_table.insert(), data)
            except Exception:
                conn.rollback()
            else:
                conn.commit()

        session.close()

    def get_data(self) -> list[dict[Any, Any]]:
        engine = create_engine(self.engine_url, echo=True)
        Session = sessionmaker(bind=engine)
        session = Session()

        metadata = MetaData()
        flights_table = Table(self.table_name, metadata, autoload_with=engine)

        return session.execute(flights_table.select()).fetchall()
