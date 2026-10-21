"""Limit order book with strict price-time (FIFO) priority matching."""
from __future__ import annotations

from collections import deque

from .orders import Order, Trade


class LimitOrderBook:
    """Integer-tick limit order book.

    ``bids``/``asks`` map a price tick to a FIFO queue of resting orders. Best
    bid is the highest populated bid tick; best ask the lowest populated ask.
    """

    def __init__(self) -> None:
        self.bids: dict[int, deque[Order]] = {}
        self.asks: dict[int, deque[Order]] = {}
        self._orders: dict[int, tuple[str, int]] = {}
        self._next_id = 1
        self.ts = 0
        self.trades: list[Trade] = []

    # ---- introspection -------------------------------------------------
    @property
    def best_bid(self) -> int | None:
        return max(self.bids) if self.bids else None

    @property
    def best_ask(self) -> int | None:
        return min(self.asks) if self.asks else None

    @property
    def mid(self) -> float | None:
        bb, ba = self.best_bid, self.best_ask
        return 0.5 * (bb + ba) if bb is not None and ba is not None else None

    @property
    def spread(self) -> int | None:
        bb, ba = self.best_bid, self.best_ask
        return (ba - bb) if bb is not None and ba is not None else None

    def _level_size(self, book: dict[int, deque[Order]], price: int | None) -> float:
        if price is None or price not in book:
            return 0.0
        return float(sum(o.size for o in book[price]))

    def best_bid_size(self) -> float:
        return self._level_size(self.bids, self.best_bid)

    def best_ask_size(self) -> float:
        return self._level_size(self.asks, self.best_ask)

    @property
    def microprice(self) -> float | None:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        qb, qa = self.best_bid_size(), self.best_ask_size()
        if qb + qa <= 0:
            return self.mid
        return (qa * bb + qb * ba) / (qb + qa)

    def depth(self, side: str, levels: int) -> float:
        book = self.bids if side == "bid" else self.asks
        prices = sorted(book, reverse=(side == "bid"))[:levels]
        return float(sum(self._level_size(book, p) for p in prices))

    def snapshot(self, levels: int = 10) -> dict[str, list[tuple[int, float]]]:
        bid_prices = sorted(self.bids, reverse=True)[:levels]
        ask_prices = sorted(self.asks)[:levels]
        return {
            "bids": [(p, self._level_size(self.bids, p)) for p in bid_prices],
            "asks": [(p, self._level_size(self.asks, p)) for p in ask_prices],
        }

    # ---- mutation ------------------------------------------------------
    def _match(self, taker_id: int, taker_side: str, remaining: float, level_price: int) -> float:
        book = self.asks if taker_side == "buy" else self.bids
        queue = book[level_price]
        while queue and remaining > 1e-12:
            maker = queue[0]
            take = min(maker.size, remaining)
            maker.size -= take
            remaining -= take
            self.trades.append(
                Trade(level_price, take, taker_side, self.ts, maker.id, taker_id)
            )
            if maker.size <= 1e-12:
                queue.popleft()
                self._orders.pop(maker.id, None)
        if not queue:
            del book[level_price]
        return remaining

    def limit(self, side: str, price: int, size: float) -> tuple[int, float]:
        """Submit a limit order. Marketable portions execute; the rest rests.

        Returns ``(order_id, executed_qty)``.
        """
        if size <= 0:
            raise ValueError("size must be positive")
        self.ts += 1
        oid = self._next_id
        self._next_id += 1
        remaining = float(size)

        if side == "buy":
            while remaining > 1e-12 and self.asks and price >= min(self.asks):
                remaining = self._match(oid, "buy", remaining, min(self.asks))
            if remaining > 1e-12:
                self.bids.setdefault(price, deque()).append(Order(oid, "buy", price, remaining, self.ts))
                self._orders[oid] = ("buy", price)
        elif side == "sell":
            while remaining > 1e-12 and self.bids and price <= max(self.bids):
                remaining = self._match(oid, "sell", remaining, max(self.bids))
            if remaining > 1e-12:
                self.asks.setdefault(price, deque()).append(Order(oid, "sell", price, remaining, self.ts))
                self._orders[oid] = ("sell", price)
        else:
            raise ValueError("side must be 'buy' or 'sell'")

        return oid, float(size) - remaining

    def market(self, side: str, size: float) -> float:
        """Submit a market order. Returns executed quantity."""
        if size <= 0:
            raise ValueError("size must be positive")
        self.ts += 1
        taker_id = self._next_id
        self._next_id += 1
        remaining = float(size)

        if side == "buy":
            while remaining > 1e-12 and self.asks:
                remaining = self._match(taker_id, "buy", remaining, min(self.asks))
        elif side == "sell":
            while remaining > 1e-12 and self.bids:
                remaining = self._match(taker_id, "sell", remaining, max(self.bids))
        else:
            raise ValueError("side must be 'buy' or 'sell'")

        return float(size) - remaining

    def cancel(self, order_id: int) -> bool:
        info = self._orders.get(order_id)
        if info is None:
            return False
        side, price = info
        book = self.bids if side == "buy" else self.asks
        queue = book.get(price)
        if queue is not None:
            for i, order in enumerate(queue):
                if order.id == order_id:
                    del queue[i]
                    break
            if not queue:
                del book[price]
        self._orders.pop(order_id, None)
        return True

    def resting_ids(self) -> list[int]:
        return list(self._orders)
