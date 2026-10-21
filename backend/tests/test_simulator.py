from app.core.simulator import MarketSimulator, SimConfig


def test_runs_and_produces_trades():
    res = MarketSimulator(SimConfig(seed=1, n_events=2000)).run()
    assert len(res.mids) == 2000
    assert res.n_trades > 0
    assert all(m > 0 for m in res.mids)


def test_reproducible_with_same_seed():
    a = MarketSimulator(SimConfig(seed=5, n_events=1000)).run()
    b = MarketSimulator(SimConfig(seed=5, n_events=1000)).run()
    assert a.mids == b.mids
    assert a.n_trades == b.n_trades


def test_different_seed_differs():
    a = MarketSimulator(SimConfig(seed=1, n_events=1000)).run()
    b = MarketSimulator(SimConfig(seed=2, n_events=1000)).run()
    assert a.mids != b.mids


def test_hawkes_flow_runs():
    res = MarketSimulator(SimConfig(seed=3, n_events=2000, flow="hawkes")).run()
    assert res.n_trades > 0


def test_snapshot_structure():
    res = MarketSimulator(SimConfig(seed=1, n_events=500)).run()
    assert "bids" in res.snapshot
    assert "asks" in res.snapshot
