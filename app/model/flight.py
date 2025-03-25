from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints


class Flight(BaseModel):
    airline: str
    flight_number: Annotated[int, Field(strict=True, ge=1000, le=9999)]
    origin: Annotated[
        str,
        StringConstraints(strip_whitespace=True, to_upper=True, pattern=r"^[A-Z]{3}$"),
    ]

    destination: Annotated[
        str,
        StringConstraints(strip_whitespace=True, to_upper=True, pattern=r"^[A-Z]{3}$"),
    ]
    departure_time: str
    weather: Literal["Windy", "Foggy", "Stormy", "Rainy", "Clear"]
    congestion_level: Literal["Medium", "High", "Low"]
    day_of_week: Literal[
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ]
    delay: int | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "airline": "Delta",
                    "flight_number": 2612,
                    "origin": "MIA",
                    "destination": "ORD",
                    "departure_time": "2025-04-03 13:00:00",
                    "weather": "Clear",
                    "congestion_level": "Low",
                    "day_of_week": "Sunday",
                    "delay": None,
                }
            ]
        }
    }
