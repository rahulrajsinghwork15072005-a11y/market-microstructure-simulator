"""Calibrate simulator parameters from a real ticker's recent volatility.

Falls back to sensible synthetic defaults when offline / yfinance is missing.
"""
from __future__ import annotations

import numpy as np

try:  # optional dependency
    import yfinance as _yf

    _HAS_YF = True
except Exception:  # pragma: no cover - import guard
    _HAS_YF = False


def calibrate(ticker: str, allow_network: bool = True) -> dict:
    if allow_network and _HAS_YF:
        try:  # pragma: no cover - network path
            frame = _yf.Ticker(ticker).history(period="1mo", interval="1d")
            closes = frame["Close"].dropna().to_numpy(dtype=float)
            if len(closes) > 5:
                daily_vol = float(np.std(np.diff(np.log(closes))))
                last = float(closes[-1])
                return {
                    "ticker": ticker,
                    "source": "yfinance",
                    "last_price": last,
                    "daily_vol": daily_vol,
                    "suggested_init_mid": max(int(round(last)), 10),
                    "suggested_market_rate": 2.0,
                }
        except Exception:  # pragma: no cover - network/parse failures
            pass
    return {
        "ticker": ticker,
        "source": "synthetic",
        "last_price": 100.0,
        "daily_vol": 0.012,
        "suggested_init_mid": 1000,
        "suggested_market_rate": 2.0,
    }
