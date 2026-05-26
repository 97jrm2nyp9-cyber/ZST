"""Thin yfinance wrapper. Returns a dict of metrics per ticker or None."""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional

import yfinance as yf


@dataclass
class Metrics:
    price: Optional[float]
    market_cap_b: Optional[float]
    avg_dollar_volume_30d: Optional[float]
    trailing_pe: Optional[float]
    ev_ebitda: Optional[float]


def _clean(x):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def fetch_metrics(ticker: str) -> Metrics:
    t = yf.Ticker(ticker)
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}

    price = _clean(info.get("currentPrice") or info.get("regularMarketPrice"))
    market_cap = _clean(info.get("marketCap"))
    market_cap_b = market_cap / 1e9 if market_cap else None
    trailing_pe = _clean(info.get("trailingPE"))
    ev_ebitda = _clean(info.get("enterpriseToEbitda"))

    avg_dollar_volume = None
    try:
        hist = t.history(period="2mo", auto_adjust=False)
        if not hist.empty:
            last30 = hist.tail(30)
            dv = (last30["Close"] * last30["Volume"]).mean()
            avg_dollar_volume = _clean(dv)
    except Exception:
        pass

    return Metrics(
        price=price,
        market_cap_b=market_cap_b,
        avg_dollar_volume_30d=avg_dollar_volume,
        trailing_pe=trailing_pe,
        ev_ebitda=ev_ebitda,
    )


def metrics_to_dict(m: Metrics) -> dict:
    return asdict(m)
