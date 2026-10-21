# Market Microstructure Simulator — Stylized Facts

Real financial returns exhibit robust, model-independent regularities known as
**stylized facts**. A microstructure simulator is credible only if these emerge
from the order-flow dynamics rather than being imposed. This document lists the
facts the simulator targets and how each is measured.

All metrics below are computed in `app/core/metrics.py` and surfaced at
`/api/demo/stylized-facts`; the assertions are exercised by `backend/tests/`.

## 1. Heavy-tailed (fat-tailed) returns

Asset returns are not Gaussian; they have excess kurtosis (fat tails).

- **Metric:** `excess_kurtosis(log_returns)` (Fisher definition; 0 = Gaussian).
- **Expectation:** noticeably **> 0** under bursty/Hawkes flow.

## 2. Volatility clustering

Large moves cluster: returns are roughly uncorrelated, but **absolute** returns are
positively autocorrelated and decay slowly.

- **Metric:** `acf(|returns|, lags=1..10)` should be positive and decaying, while
  `autocorr(returns, lag=1)` sits near zero.
- **Driver:** the self-exciting **Hawkes** process (branching ratio α/β < 1).

## 3. Aggregational behaviour / random-walk mid

The mid price behaves close to a random walk over short horizons.

- **Metric:** `hurst_exponent(mids)` ≈ 0.5 (≈ random walk; >0.5 trending; <0.5
  mean-reverting).

## 4. Measurable price impact

Signed order flow moves the price; impact is concave in size.

- **Linear (Kyle):** `kyle_lambda()` — OLS slope of $\Delta P = \lambda \cdot V$.
- **Concave (square-root law):** `square_root_impact()` — impact $\propto \sigma
  \sqrt{Q/V}$.

## 5. Realized volatility

- **Metric:** `realized_volatility(mids)` — standard deviation of log returns,
  reported alongside the facts above.

## Scenarios

Three presets stress these facts differently (`app/core/scenarios.py`):

| Scenario | Intent |
|---|---|
| `balanced` | Baseline Poisson-ish flow |
| `thin_liquidity` | Fewer levels / larger relative sizes → wider spread, fatter tails |
| `toxic_flow` | Strong Hawkes excitation (α=0.6) → pronounced clustering |

## Roadmap

- ⬜ **L2 calibration:** calibrate arrival/cancel rates, spread/depth distributions,
  and the Hawkes branching ratio from a bundled sample of (synthetic-realistic)
  order-book data.
- ⬜ **Multi-agent mode:** explicit market-maker, informed, noise, and
  liquidity-taker agents.
- ⬜ **C++ engine binding:** optionally bind the workspace `limit-order-book` engine
  via pybind11 (today the Python engine is the only matching implementation — see
  the project README's Roadmap).
