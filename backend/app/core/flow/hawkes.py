"""Self-exciting (Hawkes) process with exponential kernel.

Intensity: ``lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta (t - t_i))``.
The branching ratio ``n = alpha / beta`` must be < 1 for stationarity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class HawkesProcess:
    mu: float
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if self.mu < 0 or self.alpha < 0 or self.beta <= 0:
            raise ValueError("require mu >= 0, alpha >= 0, beta > 0")

    @property
    def branching_ratio(self) -> float:
        return self.alpha / self.beta

    def simulate(self, horizon: float, rng: np.random.Generator) -> list[float]:
        """Ogata thinning simulation of event times on ``[0, horizon]``."""
        times: list[float] = []
        t = 0.0
        intensity_decayed = 0.0
        last_t = 0.0
        while t < horizon:
            upper = self.mu + intensity_decayed  # intensity just after last event
            if upper <= 0:
                break
            t += float(rng.exponential(1.0 / upper))
            if t >= horizon:
                break
            decayed = intensity_decayed * np.exp(-self.beta * (t - last_t))
            current = self.mu + decayed
            if rng.random() <= current / upper:
                times.append(t)
                intensity_decayed = decayed + self.alpha
                last_t = t
            else:
                intensity_decayed = decayed
                last_t = t
        return times


def decay(state: float, beta: float, dt: float) -> float:
    """Exponentially decay an excitation state forward by ``dt``."""
    return float(state * np.exp(-beta * dt))


def estimate_branching_ratio(times: list[float], beta: float) -> float:
    """Crude branching-ratio estimate from the fraction of clustered events.

    Uses the ratio of the realized rate to the implied background rate; intended
    as a sanity check on simulated clustering, not a full MLE.
    """
    if len(times) < 2:
        return 0.0
    span = times[-1] - times[0]
    if span <= 0:
        return 0.0
    realized_rate = len(times) / span
    # for a stationary Hawkes: realized_rate = mu / (1 - n)  => n = 1 - mu/rate
    # without mu we report the excess over the inter-event-implied baseline.
    baseline = (len(times) - 1) / span
    return float(max(0.0, 1.0 - baseline / realized_rate))


def hawkes_log_likelihood(
    times: list[float] | np.ndarray,
    mu: float,
    alpha: float,
    beta: float,
    horizon: float | None = None,
) -> float:
    """Exact log-likelihood of an exponential-kernel Hawkes process.

    Uses the Ozaki (1979) recursion so the excitation sum is evaluated in O(n):
    ``A_i = exp(-beta (t_i - t_{i-1})) (1 + A_{i-1})`` with ``A_0 = 0``. ``horizon``
    is the observation window ``T`` (defaults to the last event time).
    """
    t = np.asarray(times, dtype=float)
    n = int(t.size)
    if mu <= 0.0 or alpha < 0.0 or beta <= 0.0:
        return -np.inf
    if n == 0:
        return -mu * float(horizon or 0.0)
    horizon_t = float(horizon) if horizon is not None else float(t[-1])

    log_intensity_sum = float(np.log(mu))  # the first event sees only the baseline
    excitation = 0.0
    for i in range(1, n):
        excitation = float(np.exp(-beta * (t[i] - t[i - 1])) * (1.0 + excitation))
        intensity = mu + alpha * excitation
        if intensity <= 0.0:
            return -np.inf
        log_intensity_sum += float(np.log(intensity))

    compensator = mu * horizon_t + (alpha / beta) * float(
        np.sum(1.0 - np.exp(-beta * (horizon_t - t)))
    )
    return float(log_intensity_sum - compensator)


def fit_hawkes(
    times: list[float] | np.ndarray, horizon: float | None = None
) -> dict[str, float | int | bool]:
    """Maximum-likelihood fit of an exponential Hawkes process to event times.

    Returns the estimated ``mu``, ``alpha``, ``beta``, the branching ratio
    ``n = alpha / beta`` (a measure of endogeneity / reflexivity of the order
    flow), the maximised log-likelihood, and convergence info. The optimisation
    runs over log-parameters so the estimates stay strictly positive.
    """
    t = np.asarray(times, dtype=float)
    n = int(t.size)
    if n < 5:
        raise ValueError("need at least 5 events to fit a Hawkes process")
    if np.any(np.diff(t) < 0):
        t = np.sort(t)
    horizon_t = float(horizon) if horizon is not None else float(t[-1])
    if horizon_t <= 0:
        raise ValueError("horizon must be positive")

    rate = n / horizon_t
    mean_dt = horizon_t / n
    x0 = np.log([0.5 * rate, 0.5 / mean_dt, 1.0 / mean_dt])  # mu, alpha, beta

    def neg_ll(params: np.ndarray) -> float:
        mu, alpha, beta = (float(v) for v in np.exp(params))
        return -hawkes_log_likelihood(t, mu, alpha, beta, horizon_t)

    res = minimize(
        neg_ll,
        x0,
        method="Nelder-Mead",
        options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-7},
    )
    mu, alpha, beta = (float(v) for v in np.exp(res.x))
    return {
        "mu": mu,
        "alpha": alpha,
        "beta": beta,
        "branching_ratio": alpha / beta,
        "log_likelihood": float(-res.fun),
        "n_events": n,
        "horizon": horizon_t,
        "converged": bool(res.success),
    }
