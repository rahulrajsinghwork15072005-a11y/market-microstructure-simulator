"""Tests for the multi-agent market simulation."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.multiagent import MultiAgentConfig, run_multi_agent
from app.main import app

client = TestClient(app)


def test_reproducible_with_same_seed():
    cfg = MultiAgentConfig(seed=11, n_steps=300)
    a = run_multi_agent(cfg)
    b = run_multi_agent(cfg)
    assert a.mids == b.mids
    assert a.agents == b.agents


def test_diverges_with_different_seed():
    a = run_multi_agent(MultiAgentConfig(seed=1, n_steps=300))
    b = run_multi_agent(MultiAgentConfig(seed=2, n_steps=300))
    assert a.mids != b.mids


def test_series_lengths_and_trades():
    res = run_multi_agent(MultiAgentConfig(seed=5, n_steps=250))
    assert len(res.mids) == len(res.spreads) == len(res.fair_values) == 250
    assert res.n_trades > 0
    assert {a["name"] for a in res.agents} == {"market_maker", "informed", "noise"}


def test_market_maker_provides_two_sided_liquidity():
    res = run_multi_agent(MultiAgentConfig(seed=3, n_steps=300))
    # The book stays two-sided (the MM keeps reposting), and the MM trades.
    assert res.snapshot["bids"] and res.snapshot["asks"]
    mm = next(a for a in res.agents if a["name"] == "market_maker")
    assert mm["n_trades"] > 0


def test_informed_trader_profits_under_strong_drift():
    # A persistent upward drift in the hidden fair value lets the informed
    # trader accumulate longs ahead of the price and profit.
    res = run_multi_agent(
        MultiAgentConfig(seed=9, n_steps=500, info_drift=2.0, info_vol=0.5)
    )
    informed = next(a for a in res.agents if a["name"] == "informed")
    assert informed["inventory"] > 0
    assert informed["pnl"] > 0
    # The price should have risen meaningfully above the starting mid.
    assert res.mids[-1] > res.mids[0]


def test_multi_agent_endpoint():
    resp = client.get("/api/demo/multi-agent?n_steps=200&seed=4")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_steps"] == 200
    assert len(body["mids"]) == 200
    assert len(body["agents"]) == 3
    assert body["n_trades"] > 0
