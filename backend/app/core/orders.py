"""Order and trade data types."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Order:
    id: int
    side: str       # "buy" | "sell"
    price: int      # integer ticks
    size: float
    ts: int


@dataclass(frozen=True)
class Trade:
    price: int
    size: float
    aggressor: str  # side of the incoming (taking) order
    ts: int
    maker_id: int
    taker_id: int
