"""Tests for the Hawkes maximum-likelihood calibration."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.flow.hawkes import HawkesProcess, fit_hawkes, hawkes_log_likelihood
from app.main import app

client = TestClient(app)


def test_mle_recovers_known_parameters():
    """Simulate with known params; MLE should recover them within tolerance."""
    true = HawkesProcess(mu=1.0, alpha=0.8, beta=1.5)
    rng = np.random.default_rng(7)
    times = true.simulate(2000.0, rng)
    fit = fit_hawkes(times, horizon=2000.0)
    assert fit["converged"]
    assert fit["mu"] == pytest.approx(1.0, rel=0.3)
    assert fit["beta"] == pytest.approx(1.5, rel=0.35)
    # Branching ratio (alpha/beta) is the well-identified quantity.
    assert fit["branching_ratio"] == pytest.approx(0.8 / 1.5, abs=0.12)


def test_log_likelihood_higher_at_truth_than_far():
    rng = np.random.default_rng(11)
    times = HawkesProcess(mu=1.0, alpha=0.6, beta=1.2).simulate(1800.0, rng)
    ll_true = hawkes_log_likelihood(times, 1.0, 0.6, 1.2, 1800.0)
    ll_far = hawkes_log_likelihood(times, 5.0, 0.1, 5.0, 1800.0)
    assert np.isfinite(ll_true)
    assert ll_true > ll_far


def test_log_likelihood_rejects_invalid_params():
    times = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert hawkes_log_likelihood(times, mu=-1.0, alpha=0.5, beta=1.0) == -np.inf
    assert hawkes_log_likelihood(times, mu=1.0, alpha=0.5, beta=-1.0) == -np.inf


def test_fit_requires_minimum_events():
    with pytest.raises(ValueError):
        fit_hawkes([1.0, 2.0, 3.0])


def test_branching_ratio_below_one_for_stationary():
    rng = np.random.default_rng(3)
    times = HawkesProcess(mu=1.0, alpha=0.5, beta=2.0).simulate(2000.0, rng)
    fit = fit_hawkes(times, horizon=2000.0)
    assert 0.0 <= fit["branching_ratio"] < 1.0


def test_api_demo_hawkes_calibration():
    resp = client.get("/api/demo/hawkes-calibration", params={"horizon": 1200, "seed": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["true"]) >= {"mu", "alpha", "beta", "branching_ratio"}
    assert "recovery_error" in body
    assert body["n_events"] > 50


def test_api_demo_hawkes_rejects_unstable():
    resp = client.get("/api/demo/hawkes-calibration", params={"alpha": 2.0, "beta": 1.0})
    assert resp.status_code == 422


def test_api_hawkes_fit_post():
    rng = np.random.default_rng(9)
    times = HawkesProcess(mu=1.0, alpha=0.7, beta=1.4).simulate(1500.0, rng)
    resp = client.post("/api/hawkes/fit", json={"times": times, "horizon": 1500.0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_events"] == len(times)
    assert "branching_ratio" in body
