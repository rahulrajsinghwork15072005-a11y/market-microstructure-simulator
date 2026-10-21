"""Inventory-aware market-making agent (Avellaneda-Stoikov)."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ASQuote:
    reservation_price: float
    bid: float
    ask: float
    half_spread: float


class AvellanedaStoikovMaker:
    """Computes inventory-skewed optimal quotes around a mid price.

    reservation price: ``r = s - q * gamma * sigma^2 * (T - t)``
    optimal spread:    ``gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)``
    """

    def __init__(self, gamma: float = 0.1, sigma: float = 1.0, k: float = 1.5, horizon: float = 1.0) -> None:
        if gamma <= 0 or sigma < 0 or k <= 0:
            raise ValueError("require gamma > 0, sigma >= 0, k > 0")
        self.gamma = gamma
        self.sigma = sigma
        self.k = k
        self.horizon = horizon

    def quote(self, mid: float, inventory: float, time_remaining: float | None = None) -> ASQuote:
        t = self.horizon if time_remaining is None else time_remaining
        variance = self.sigma**2
        reservation = mid - inventory * self.gamma * variance * t
        spread = self.gamma * variance * t + (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.k)
        half = spread / 2.0
        return ASQuote(
            reservation_price=reservation,
            bid=reservation - half,
            ask=reservation + half,
            half_spread=half,
        )
