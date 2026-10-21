"""Homogeneous Poisson arrival helpers."""
from __future__ import annotations

import numpy as np


def sample_interarrival(rate: float, rng: np.random.Generator) -> float:
    """Exponential inter-arrival time for a Poisson process of given ``rate``."""
    if rate <= 0:
        raise ValueError("rate must be positive")
    return float(rng.exponential(1.0 / rate))


def homogeneous_poisson(rate: float, horizon: float, rng: np.random.Generator) -> list[float]:
    """Event times of a homogeneous Poisson process on ``[0, horizon]``."""
    if rate <= 0:
        raise ValueError("rate must be positive")
    times: list[float] = []
    t = 0.0
    while True:
        t += float(rng.exponential(1.0 / rate))
        if t > horizon:
            break
        times.append(t)
    return times
