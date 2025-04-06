from sqlalchemy import Column, Integer, String

from .abstract_table import AbstractTable


class FlightsTable(AbstractTable):
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
