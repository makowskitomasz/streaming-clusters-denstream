from pydantic import BaseModel


class DataPoint(BaseModel):
    x: float
    y: float
    timestamp: float
    cluster_id: int
    source: str = "nyc_taxi"
