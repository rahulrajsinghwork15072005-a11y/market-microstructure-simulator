from app.core.book import LimitOrderBook


def test_resting_and_best():
    b = LimitOrderBook()
    b.limit("buy", 100, 5)
    b.limit("sell", 102, 5)
    assert b.best_bid == 100
    assert b.best_ask == 102
    assert b.mid == 101
    assert b.spread == 2


def test_price_time_priority_fifo():
    b = LimitOrderBook()
    id1, _ = b.limit("buy", 100, 5)
    b.limit("buy", 100, 5)  # same price, later in queue
    b.limit("sell", 100, 3)  # marketable, must hit id1 first
    assert b.trades[0].maker_id == id1
    assert b.trades[0].size == 3
    assert b.best_bid_size() == 7  # 2 left on id1 + 5 on id2


def test_crossing_limit_executes():
    b = LimitOrderBook()
    b.limit("sell", 101, 5)
    oid, executed = b.limit("buy", 101, 3)
    assert executed == 3
    assert b.best_ask == 101  # 2 remain
    assert b.best_bid is None


def test_market_order_sweeps_levels():
    b = LimitOrderBook()
    b.limit("sell", 101, 2)
    b.limit("sell", 102, 2)
    executed = b.market("buy", 3)
    assert executed == 3
    assert b.best_ask == 102


def test_market_order_partial_when_thin():
    b = LimitOrderBook()
    b.limit("sell", 101, 2)
    executed = b.market("buy", 5)
    assert executed == 2
    assert b.best_ask is None


def test_cancel():
    b = LimitOrderBook()
    oid, _ = b.limit("buy", 100, 5)
    assert b.cancel(oid) is True
    assert b.best_bid is None
    assert b.cancel(9999) is False


def test_size_conservation_on_match():
    b = LimitOrderBook()
    b.limit("buy", 100, 5)
    b.limit("sell", 100, 3)
    assert sum(t.size for t in b.trades) == 3
    assert b.best_bid == 100
    assert b.best_bid_size() == 2


def test_microprice_size_weighted():
    b = LimitOrderBook()
    b.limit("buy", 100, 1)
    b.limit("sell", 102, 9)
    mp = b.microprice
    assert 100 <= mp <= 102
    assert mp < b.mid  # heavy ask pulls microprice toward the bid
