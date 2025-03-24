from pydantic import BaseModel


class Flight(BaseModel):
    airline: str
    flight_number: int
    origin: str
    destination: str
    departure_time: str
    weather: str
    congestion_level: str
    day_of_week: str
