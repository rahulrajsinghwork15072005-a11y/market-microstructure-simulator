"""Micro-benchmarks for the microstructure engine.

Times the discrete-event simulator (Poisson and Hawkes flow), the stylized-fact
metrics, and the Hawkes maximum-likelihood calibration. Runs entirely offline.

    python scripts/bench.py            # from backend/
    python scripts/bench.py --events 8000

Numbers are wall-clock and machine-dependent; treat them as relative signals.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make ``app`` importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app.core.flow.hawkes import HawkesProcess, fit_hawkes  # noqa: E402
from app.core.metrics import stylized_facts  # noqa: E402
from app.core.simulator import MarketSimulator, SimConfig  # noqa: E402


def _best(fn, repeats: int) -> float:
    best = float("inf")
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the microstructure engine.")
    parser.add_argument("--events", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)

    print(f"\nMicrostructure simulator benchmarks  (n_events={args.events})")
    print("-" * 60)

    for flow in ("poisson", "hawkes"):
        cfg = SimConfig(n_events=args.events, flow=flow, seed=1)
        secs = _best(lambda c=cfg: MarketSimulator(c).run(), args.repeats)
        print(f"  simulate ({flow:7})   {secs * 1e3:8.2f} ms   {args.events / secs:>12,.0f} events/s")

    res = MarketSimulator(SimConfig(n_events=args.events, seed=1)).run()
    sf = _best(lambda: stylized_facts(res.mids), 5)
    print(f"  stylized facts       {sf * 1e3:8.2f} ms")

    # Hawkes MLE calibration: simulate a known process and recover it.
    proc = HawkesProcess(mu=1.0, alpha=0.8, beta=1.5)
    times = proc.simulate(1500.0, np.random.default_rng(1))
    fit_secs = _best(lambda: fit_hawkes(times, horizon=1500.0), args.repeats)
    fitted = fit_hawkes(times, horizon=1500.0)
    print(
        f"  Hawkes MLE ({len(times)} ev)  {fit_secs * 1e3:8.2f} ms   "
        f"branching: true={proc.branching_ratio:.3f} fitted={fitted['branching_ratio']:.3f}"
    )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
