"""Price-impact models."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def kyle_lambda(price_changes: ArrayLike, signed_volumes: ArrayLike) -> float:
    """Estimate Kyle's lambda from ``dP = lambda * signed_volume + eps`` (OLS slope)."""
    dp = np.asarray(price_changes, dtype=float)
    v = np.asarray(signed_volumes, dtype=float)
    if dp.shape != v.shape or dp.ndim != 1 or dp.size < 2:
        raise ValueError("inputs must be equal-length 1-D arrays of length >= 2")
    design = np.column_stack([np.ones_like(v), v])
    coef, *_ = np.linalg.lstsq(design, dp, rcond=None)
    return float(coef[1])


def square_root_impact(y: float, sigma: float, quantity: float, volume: float) -> float:
    """Empirical square-root market impact ``Y * sigma * sqrt(Q / V)``."""
    if volume <= 0:
        raise ValueError("volume must be positive")
    return float(y * sigma * np.sqrt(max(quantity, 0.0) / volume))
