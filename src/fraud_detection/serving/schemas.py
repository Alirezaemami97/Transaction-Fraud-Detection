from typing import Any, Literal

from pydantic import BaseModel


class ScoreRequest(BaseModel):
    transaction: dict[str, Any]


class ScoreResponse(BaseModel):
    fraud_score: float
    decision: Literal["FRAUD", "LEGIT"]
    threshold: float


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_name: str
