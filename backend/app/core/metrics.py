"""Microstructure metrics and stylized-fact diagnostics."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def log_returns(mids: ArrayLike) -> np.ndarray:
    m = np.asarray(mids, dtype=float)
    m = m[m > 0]
    if m.size < 2:
        return np.array([])
    return np.diff(np.log(m))


def excess_kurtosis(x: ArrayLike) -> float:
    arr = np.asarray(x, dtype=float)
    if arr.size < 4:
        return 0.0
    return float(stats.kurtosis(arr, fisher=True, bias=False))


def autocorr(x: ArrayLike, lag: int) -> float:
    arr = np.asarray(x, dtype=float)
    if lag <= 0 or arr.size <= lag + 1:
        return 0.0
    a = arr[:-lag]
    b = arr[lag:]
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def acf(x: ArrayLike, lags: list[int]) -> list[float]:
    return [autocorr(x, lag) for lag in lags]


def realized_volatility(mids: ArrayLike) -> float:
    """Standard deviation of log returns over the window."""
    r = log_returns(mids)
    return float(np.std(r)) if r.size else 0.0


def hurst_exponent(series: ArrayLike, max_lag: int = 20) -> float:
    """Hurst exponent via the lagged-difference variance scaling.

    H ~ 0.5 for a random walk, > 0.5 trending, < 0.5 mean-reverting.
    """
    x = np.asarray(series, dtype=float)
    n = x.size
    if n < 8:
        return 0.5
    upper = min(max_lag, n // 2)
    tau: list[float] = []
    valid_lags: list[int] = []
    for lag in range(2, upper):
        spread = float(np.std(x[lag:] - x[:-lag]))
        if spread > 0:
            tau.append(spread)
            valid_lags.append(lag)
    if len(tau) < 2:
        return 0.5
    slope = np.polyfit(np.log(valid_lags), np.log(tau), 1)[0]
    return float(slope)


def stylized_facts(mids: ArrayLike, max_lag: int = 10) -> dict:
    """Compute the canonical microstructure stylized facts from a mid series."""
    r = log_returns(mids)
    if r.size < 4:
        return {
            "excess_kurtosis": 0.0,
            "return_acf_lag1": 0.0,
            "abs_return_acf": [],
            "lags": [],
            "n_returns": int(r.size),
            "realized_volatility": realized_volatility(mids),
            "hurst": hurst_exponent(mids),
        }
    lags = list(range(1, max_lag + 1))
    return {
        "excess_kurtosis": excess_kurtosis(r),
        "return_acf_lag1": autocorr(r, 1),
        "abs_return_acf": acf(np.abs(r), lags),
        "lags": lags,
        "n_returns": int(r.size),
        "realized_volatility": realized_volatility(mids),
        "hurst": hurst_exponent(mids),
    }
