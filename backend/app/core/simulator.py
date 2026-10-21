"""Discrete-event market simulation kernel (zero-intelligence + agent flow).

Order flow is generated as competing Poisson streams of limit orders (liquidity
providers / makers), market orders (takers), and cancellations (noise). An
optional Hawkes self-excitation makes taker flow cluster, producing volatility
clustering. The flow maps onto the maker / taker / noise agent populations
described in PLAN.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .book import LimitOrderBook
from .flow import hawkes


@dataclass
class SimConfig:
    seed: int = 42
    n_events: int = 5000
    init_mid: int = 1000
    init_half_spread: int = 1
    init_levels: int = 10
    init_level_size: float = 5.0
    limit_rate: float = 8.0
    market_rate: float = 2.0
    cancel_rate: float = 6.0
    flow: str = "poisson"          # "poisson" | "hawkes"
    hawkes_alpha: float = 0.4
    hawkes_beta: float = 1.0
    offset_lambda: float = 1.5     # mean ticks for geometric limit placement
    size_mean: float = 1.0


@dataclass
class SimResult:
    times: list[float]
    mids: list[float]
    spreads: list[float]
    bid_depth: list[float]
    ask_depth: list[float]
    signed_mkt_vol: list[float]
    n_trades: int
    snapshot: dict = field(default_factory=dict)


class MarketSimulator:
    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.book = LimitOrderBook()
        self.last_mid = float(config.init_mid)

    def _seed_book(self) -> None:
        c = self.config
        for i in range(c.init_levels):
            self.book.limit("buy", c.init_mid - c.init_half_spread - i, c.init_level_size)
            self.book.limit("sell", c.init_mid + c.init_half_spread + i, c.init_level_size)

    def _rand_size(self) -> float:
        return float(max(self.rng.exponential(self.config.size_mean), 0.01))

    def _geom_offset(self) -> int:
        p = 1.0 / (1.0 + self.config.offset_lambda)
        return int(self.rng.geometric(p) - 1)

    def _place_limit(self, side: str) -> None:
        offset = self._geom_offset()
        size = self._rand_size()
        bb, ba = self.book.best_bid, self.book.best_ask
        if side == "buy":
            ref = (ba - 1) if ba is not None else int(round(self.last_mid))
            price = ref - offset
        else:
            ref = (bb + 1) if bb is not None else int(round(self.last_mid))
            price = ref + offset
        self.book.limit(side, int(price), size)

    def _place_market(self, side: str) -> float:
        return self.book.market(side, self._rand_size())

    def _cancel_random(self) -> None:
        ids = self.book.resting_ids()
        if ids:
            oid = ids[int(self.rng.integers(len(ids)))]
            self.book.cancel(oid)

    def run(self) -> SimResult:
        self._seed_book()
        c = self.config
        rng = self.rng

        t = 0.0
        excitation = 0.0
        times, mids, spreads, bid_depth, ask_depth, signed = [], [], [], [], [], []

        for _ in range(c.n_events):
            market_rate = c.market_rate + (excitation if c.flow == "hawkes" else 0.0)
            rates = np.array([c.limit_rate, market_rate, c.cancel_rate], dtype=float)
            total = float(rates.sum())
            dt = float(rng.exponential(1.0 / total))
            t += dt
            if c.flow == "hawkes":
                excitation = hawkes.decay(excitation, c.hawkes_beta, dt)

            u = rng.random() * total
            signed_vol = 0.0
            if u < rates[0]:
                side = "buy" if rng.random() < 0.5 else "sell"
                self._place_limit(side)
            elif u < rates[0] + rates[1]:
                side = "buy" if rng.random() < 0.5 else "sell"
                executed = self._place_market(side)
                signed_vol = executed if side == "buy" else -executed
                if c.flow == "hawkes":
                    excitation += c.hawkes_alpha
            else:
                self._cancel_random()

            m = self.book.mid
            if m is None:
                m = self.last_mid
            self.last_mid = m

            times.append(t)
            mids.append(m)
            spreads.append(float(self.book.spread or 0))
            bid_depth.append(self.book.depth("bid", 5))
            ask_depth.append(self.book.depth("ask", 5))
            signed.append(signed_vol)

        return SimResult(
            times=times,
            mids=mids,
            spreads=spreads,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            signed_mkt_vol=signed,
            n_trades=len(self.book.trades),
            snapshot=self.book.snapshot(10),
        )
