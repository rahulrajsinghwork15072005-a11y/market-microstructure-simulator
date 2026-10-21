"""Preset simulation scenarios."""
from __future__ import annotations

from .simulator import SimConfig

SCENARIOS: dict[str, SimConfig] = {
    "balanced": SimConfig(seed=42, n_events=5000),
    "thin_liquidity": SimConfig(
        seed=7, n_events=5000, init_levels=4, init_level_size=2.0,
        limit_rate=5.0, market_rate=3.0, cancel_rate=5.0,
    ),
    "toxic_flow": SimConfig(
        seed=11, n_events=5000, flow="hawkes",
        hawkes_alpha=0.6, hawkes_beta=1.0, market_rate=2.0,
    ),
}


def get_scenario(name: str) -> SimConfig:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario '{name}'")
    return SCENARIOS[name]
