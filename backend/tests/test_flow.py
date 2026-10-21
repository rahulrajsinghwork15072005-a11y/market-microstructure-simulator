import numpy as np
import pytest

from app.core.flow.hawkes import HawkesProcess, decay
from app.core.flow.poisson import homogeneous_poisson, sample_interarrival


def test_poisson_interarrival_mean():
    rng = np.random.default_rng(0)
    xs = [sample_interarrival(2.0, rng) for _ in range(20000)]
    assert np.mean(xs) == pytest.approx(0.5, abs=0.02)  # 1 / rate


def test_homogeneous_poisson_count():
    rng = np.random.default_rng(0)
    times = homogeneous_poisson(5.0, 100.0, rng)
    assert abs(len(times) - 500) < 70  # ~ rate * horizon


def test_hawkes_branching_ratio():
    assert HawkesProcess(mu=1.0, alpha=0.5, beta=1.0).branching_ratio == 0.5


def test_hawkes_self_excitation_raises_rate():
    rng = np.random.default_rng(1)
    h = HawkesProcess(mu=1.0, alpha=0.6, beta=1.0)  # stationary mean rate mu/(1-n)=2.5
    times = h.simulate(2000.0, rng)
    assert len(times) / 2000.0 > 1.0  # exceeds background mu


def test_decay_monotone():
    assert decay(1.0, 1.0, 0.0) == 1.0
    assert decay(1.0, 1.0, 1.0) < 1.0


def test_hawkes_invalid_params():
    with pytest.raises(ValueError):
        HawkesProcess(mu=0.0, alpha=0.0, beta=0.0)
