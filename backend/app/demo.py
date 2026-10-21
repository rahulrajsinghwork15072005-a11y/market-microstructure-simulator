"""Demo orchestration for the microstructure simulator."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .core.flow.hawkes import HawkesProcess, fit_hawkes
from .core.impact import kyle_lambda, square_root_impact
from .core.metrics import log_returns, stylized_facts
from .core.scenarios import SCENARIOS, get_scenario
from .core.simulator import MarketSimulator, SimResult


@lru_cache(maxsize=8)
def _run(scenario: str) -> SimResult:
    return MarketSimulator(get_scenario(scenario)).run()


def _downsample(xs: list[float], n: int = 800) -> list[float]:
    if len(xs) <= n:
        return list(xs)
    idx = np.linspace(0, len(xs) - 1, n).astype(int)
    return [xs[i] for i in idx]


def list_scenarios() -> list[str]:
    return list(SCENARIOS)


def demo_run(scenario: str = "balanced") -> dict:
    res = _run(scenario)
    return {
        "scenario": scenario,
        "n_events": len(res.mids),
        "n_trades": res.n_trades,
        "times": _downsample(res.times),
        "mids": _downsample(res.mids),
        "spreads": _downsample(res.spreads),
        "bid_depth": _downsample(res.bid_depth),
        "ask_depth": _downsample(res.ask_depth),
    }


def demo_stylized_facts(scenario: str = "balanced") -> dict:
    res = _run(scenario)
    return {"scenario": scenario, **stylized_facts(res.mids)}


def demo_impact(scenario: str = "balanced") -> dict:
    res = _run(scenario)
    mids = np.asarray(res.mids, dtype=float)
    dp = np.diff(mids)
    v = np.asarray(res.signed_mkt_vol[1:], dtype=float)
    lam = kyle_lambda(dp, v) if dp.size >= 2 else 0.0

    r = log_returns(mids)
    sigma = float(np.std(r)) if r.size else 0.0
    total_volume = float(np.sum(np.abs(res.signed_mkt_vol))) or 1.0
    q_grid = np.linspace(0.0, max(total_volume * 0.2, 1.0), 20)
    y = 0.5
    curve = [square_root_impact(y, sigma, float(q), total_volume) for q in q_grid]
    return {
        "scenario": scenario,
        "kyle_lambda": lam,
        "sigma": sigma,
        "total_volume": total_volume,
        "Y": y,
        "Q": q_grid.tolist(),
        "sqrt_impact": curve,
    }


def demo_book(scenario: str = "balanced") -> dict:
    res = _run(scenario)
    return {"scenario": scenario, "snapshot": res.snapshot}


@lru_cache(maxsize=16)
def demo_hawkes_calibration(
    mu: float = 1.0,
    alpha: float = 0.8,
    beta: float = 1.5,
    horizon: float = 1500.0,
    seed: int = 7,
) -> dict:
    """Simulate a Hawkes flow with known parameters, then MLE-recover them.

    A self-consistency demo: the fitted ``(mu, alpha, beta)`` and the branching
    ratio should match the values used to generate the events, which validates
    the calibration end-to-end.
    """
    proc = HawkesProcess(mu=mu, alpha=alpha, beta=beta)
    rng = np.random.default_rng(seed)
    times = proc.simulate(horizon, rng)
    fitted = fit_hawkes(times, horizon=horizon)
    true = {"mu": mu, "alpha": alpha, "beta": beta, "branching_ratio": proc.branching_ratio}
    return {
        "true": true,
        "fitted": fitted,
        "n_events": len(times),
        "horizon": horizon,
        "recovery_error": {key: abs(float(fitted[key]) - true[key]) for key in true},
    }
