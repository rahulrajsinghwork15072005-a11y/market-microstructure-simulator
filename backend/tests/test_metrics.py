import numpy as np
import pytest

from app.core.metrics import autocorr, excess_kurtosis, log_returns, stylized_facts


def test_log_returns():
    np.testing.assert_allclose(log_returns([100, 100 * np.e]), [1.0])


def test_autocorr_of_linear_series_is_one():
    x = np.arange(100.0)
    assert autocorr(x, 1) == pytest.approx(1.0, abs=1e-6)


def test_excess_kurtosis_of_normal_near_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 20000)
    assert abs(excess_kurtosis(x)) < 0.2


def test_stylized_facts_structure():
    rng = np.random.default_rng(0)
    mids = 1000 + np.cumsum(rng.normal(0, 1, 500))
    sf = stylized_facts(mids)
    assert {"excess_kurtosis", "return_acf_lag1", "abs_return_acf", "lags"} <= set(sf)
    assert len(sf["abs_return_acf"]) == len(sf["lags"])
