"""Pydantic schemas for the simulator API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class SimConfigRequest(BaseModel):
    seed: int = 42
    n_events: int = Field(5000, ge=10, le=50_000)
    init_mid: int = Field(1000, gt=0)
    init_half_spread: int = Field(1, ge=1)
    init_levels: int = Field(10, ge=1, le=100)
    init_level_size: float = Field(5.0, gt=0)
    limit_rate: float = Field(8.0, gt=0)
    market_rate: float = Field(2.0, gt=0)
    cancel_rate: float = Field(6.0, ge=0)
    flow: str = Field("poisson", pattern="^(poisson|hawkes)$")
    hawkes_alpha: float = Field(0.4, ge=0)
    hawkes_beta: float = Field(1.0, gt=0)
    offset_lambda: float = Field(1.5, gt=0)
    size_mean: float = Field(1.0, gt=0)


class HawkesFitRequest(BaseModel):
    times: list[float] = Field(..., min_length=5, description="Event times (will be sorted)")
    horizon: float | None = Field(
        None, gt=0, description="Observation window T (defaults to the last event time)"
    )
