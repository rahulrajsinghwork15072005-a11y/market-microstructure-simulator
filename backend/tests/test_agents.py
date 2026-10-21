import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.agents import AvellanedaStoikovMaker
from app.core.metrics import hurst_exponent, realized_volatility
from app.main import app

client = TestClient(app)


def test_as_quote_centered_without_inventory():
    q = AvellanedaStoikovMaker(gamma=0.1, sigma=1.0, k=1.5, horizon=1.0).quote(100.0, 0.0)
    assert q.reservation_price == pytest.approx(100.0)
    assert q.bid < q.reservation_price < q.ask
    assert q.half_spread > 0


def test_as_quote_inventory_skew():
    maker = AvellanedaStoikovMaker(gamma=0.2, sigma=2.0, k=1.5, horizon=1.0)
    assert maker.quote(100.0, inventory=5.0).reservation_price < 100.0   # long -> skew down
    assert maker.quote(100.0, inventory=-5.0).reservation_price > 100.0  # short -> skew up


def test_as_invalid_params():
    with pytest.raises(ValueError):
        AvellanedaStoikovMaker(gamma=0.0)


def test_hurst_random_walk_near_half():
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.normal(0, 1, 2000))
    assert 0.35 < hurst_exponent(walk) < 0.65


def test_realized_volatility_positive():
    rng = np.random.default_rng(0)
    mids = 1000 + np.cumsum(rng.normal(0, 1, 500))
    assert realized_volatility(mids) > 0


def test_agent_quote_endpoint():
    body = client.get("/api/agent/quote", params={"mid": 1000, "inventory": 2.0}).json()
    assert body["bid"] < body["ask"]


def test_stylized_facts_includes_new_metrics():
    body = client.get("/api/demo/stylized-facts").json()
    assert "hurst" in body
    assert "realized_volatility" in body
