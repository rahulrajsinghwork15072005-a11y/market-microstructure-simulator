import numpy as np
import pytest

from app.core.impact import kyle_lambda, square_root_impact


def test_kyle_lambda_recovers_slope():
    rng = np.random.default_rng(0)
    v = rng.normal(0, 1, 5000)
    dp = 0.5 * v + rng.normal(0, 0.01, 5000)
    assert kyle_lambda(dp, v) == pytest.approx(0.5, abs=0.02)


def test_square_root_scaling():
    i1 = square_root_impact(1.0, 0.02, 1.0, 100.0)
    i4 = square_root_impact(1.0, 0.02, 4.0, 100.0)
    assert i4 == pytest.approx(2.0 * i1)  # sqrt(4) = 2


def test_square_root_volume_guard():
    with pytest.raises(ValueError):
        square_root_impact(1.0, 0.02, 1.0, 0.0)
