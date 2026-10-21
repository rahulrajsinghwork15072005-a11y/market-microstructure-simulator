"""Multi-agent market simulation.

Three archetypal agents trade in a shared limit order book:

* **MarketMaker** - posts two-sided Avellaneda-Stoikov quotes around the mid and
  refreshes them each step (a liquidity provider).
* **InformedTrader** - observes a hidden ``fair_value`` and takes liquidity when
  the book is mispriced relative to it (a liquidity taker with information).
* **NoiseTrader** - submits random passive/aggressive orders (uninformed flow).

Per-agent profit is attributed *exactly* from the book's trade tape via
maker/taker order ids, so realized + mark-to-market PnL is consistent with every
fill. The reference Python order book (`app.core.book.LimitOrderBook`) does the
matching, so this composes with the rest of the engine.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .agents import AvellanedaStoikovMaker
from .book import LimitOrderBook

__all__ = ["MultiAgentConfig", "MultiAgentResult", "AgentState", "run_multi_agent"]


@dataclass
class MultiAgentConfig:
    seed: int = 7
    n_steps: int = 600
    init_mid: int = 1000
    seed_levels: int = 5
    seed_size: float = 5.0
    info_vol: float = 0.5          # random-walk vol of the hidden fair value (ticks)
    info_drift: float = 0.0        # deterministic drift of fair value per step (ticks)
    informed_size: float = 2.0
    informed_edge: float = 1.0     # only trade when |fair - best| exceeds this
    noise_size: float = 1.0
    noise_market_prob: float = 0.4
    mm_gamma: float = 0.1
    mm_sigma: float = 1.0
    mm_k: float = 1.5
    mm_size: float = 3.0


@dataclass
class AgentState:
    name: str
    cash: float = 0.0
    inventory: float = 0.0
    n_trades: int = 0

    def pnl(self, mark: float) -> float:
        """Mark-to-market PnL: realized cash plus inventory at the mark price."""
        return self.cash + self.inventory * mark


@dataclass
class MultiAgentResult:
    times: list[int]
    mids: list[float]
    spreads: list[float]
    fair_values: list[float]
    agents: list[dict]
    n_trades: int
    snapshot: dict = field(default_factory=dict)


def run_multi_agent(config: MultiAgentConfig) -> MultiAgentResult:
    """Run the multi-agent simulation and return per-step series + agent stats."""
    rng = random.Random(config.seed)
    book = LimitOrderBook()
    mid0 = float(config.init_mid)

    # Map every agent-placed order id -> owning agent so trades can be attributed.
    owner: dict[int, AgentState] = {}
    mm = AgentState("market_maker")
    informed = AgentState("informed")
    noise = AgentState("noise")

    # Seed both sides with resting liquidity (external, unattributed).
    for lvl in range(1, config.seed_levels + 1):
        book.limit("buy", config.init_mid - lvl, config.seed_size)
        book.limit("sell", config.init_mid + lvl, config.seed_size)

    maker = AvellanedaStoikovMaker(
        gamma=config.mm_gamma, sigma=config.mm_sigma, k=config.mm_k, horizon=1.0
    )

    processed = len(book.trades)

    def attribute_new_trades() -> None:
        nonlocal processed
        for tr in book.trades[processed:]:
            taker = owner.get(tr.taker_id)
            mkr = owner.get(tr.maker_id)
            notional = tr.size * tr.price
            if tr.aggressor == "buy":
                if taker is not None:
                    taker.cash -= notional
                    taker.inventory += tr.size
                    taker.n_trades += 1
                if mkr is not None:
                    mkr.cash += notional
                    mkr.inventory -= tr.size
                    mkr.n_trades += 1
            else:  # aggressor sells
                if taker is not None:
                    taker.cash += notional
                    taker.inventory -= tr.size
                    taker.n_trades += 1
                if mkr is not None:
                    mkr.cash -= notional
                    mkr.inventory += tr.size
                    mkr.n_trades += 1
        processed = len(book.trades)

    def place(side: str, price: int, size: float, agent: AgentState) -> None:
        if size <= 0 or price <= 0:
            return
        oid, _ = book.limit(side, int(price), size)
        owner[oid] = agent
        attribute_new_trades()

    times: list[int] = []
    mids: list[float] = []
    spreads: list[float] = []
    fairs: list[float] = []

    fair = mid0
    last_mid = mid0
    last_spread = 2.0
    mm_quote_ids: list[int] = []

    for step in range(config.n_steps):
        fair += config.info_drift + rng.gauss(0.0, config.info_vol)
        cur_mid = book.mid if book.mid is not None else last_mid

        # --- Market maker: cancel stale quotes, repost around the mid ---
        for qid in mm_quote_ids:
            book.cancel(qid)
        mm_quote_ids = []
        quote = maker.quote(cur_mid, mm.inventory)
        bid_px = int(math.floor(quote.bid))
        ask_px = int(math.ceil(quote.ask))
        if ask_px <= bid_px:
            ask_px = bid_px + 1
        # Stay passive: never cross the resting book.
        if book.best_ask is not None and bid_px >= book.best_ask:
            bid_px = book.best_ask - 1
        if book.best_bid is not None and ask_px <= book.best_bid:
            ask_px = book.best_bid + 1
        if bid_px > 0:
            oid, _ = book.limit("buy", bid_px, config.mm_size)
            owner[oid] = mm
            mm_quote_ids.append(oid)
            attribute_new_trades()
        if ask_px > 0:
            oid, _ = book.limit("sell", ask_px, config.mm_size)
            owner[oid] = mm
            mm_quote_ids.append(oid)
            attribute_new_trades()

        # --- Noise trader ---
        side = "buy" if rng.random() < 0.5 else "sell"
        if rng.random() < config.noise_market_prob:
            if side == "buy" and book.best_ask is not None:
                place("buy", book.best_ask, config.noise_size, noise)
            elif side == "sell" and book.best_bid is not None:
                place("sell", book.best_bid, config.noise_size, noise)
        else:
            if side == "buy" and book.best_bid is not None:
                place("buy", book.best_bid, config.noise_size, noise)
            elif side == "sell" and book.best_ask is not None:
                place("sell", book.best_ask, config.noise_size, noise)

        # --- Informed trader: take liquidity when the book is mispriced ---
        if book.best_ask is not None and fair > book.best_ask + config.informed_edge:
            place("buy", book.best_ask, config.informed_size, informed)
        elif book.best_bid is not None and fair < book.best_bid - config.informed_edge:
            place("sell", book.best_bid, config.informed_size, informed)

        if book.mid is not None:
            last_mid = book.mid
        if book.spread is not None:
            last_spread = float(book.spread)

        times.append(step)
        mids.append(last_mid)
        spreads.append(last_spread)
        fairs.append(fair)

    final_mid = last_mid
    agents = [
        {
            "name": st.name,
            "inventory": round(st.inventory, 4),
            "pnl": round(st.pnl(final_mid), 4),
            "n_trades": st.n_trades,
        }
        for st in (mm, informed, noise)
    ]

    return MultiAgentResult(
        times=times,
        mids=mids,
        spreads=spreads,
        fair_values=fairs,
        agents=agents,
        n_trades=len(book.trades),
        snapshot=book.snapshot(10),
    )
