"""
Yahoo Finance data fetcher — drop-in alternative when Finviz is inaccessible.

Produces a DataFrame with the same column names as the Finviz fetcher so the
scorer works unchanged.  Uses:
  • S&P 500 tickers from Wikipedia as the stock universe
  • yfinance batch price download for technical indicators (SMA, RSI, momentum)
  • yfinance Ticker.info for fundamentals and analyst data
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Universe helpers
# ---------------------------------------------------------------------------

_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_NASDAQ100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"

# Hardcoded fallback: 200 large/mid-cap US tickers (S&P 500 top constituents)
_FALLBACK_TICKERS = [
    # Mega-caps
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "AVGO", "JPM",
    "LLY", "V", "UNH", "XOM", "MA", "COST", "HD", "WMT", "NFLX", "ABBV",
    "JNJ", "BAC", "CRM", "ORCL", "MRK", "CVX", "TMO", "KO", "CSCO", "ACN",
    "ABT", "LIN", "MCD", "PEP", "IBM", "GE", "TXN", "ADBE", "AMD", "QCOM",
    "PM", "DHR", "WFC", "GS", "BX", "ISRG", "INTU", "CAT", "NOW", "SPGI",
    "BKNG", "UNP", "AXP", "AMGN", "PLD", "RTX", "NEE", "HON", "SYK", "LOW",
    "MS", "T", "VRTX", "ELV", "BSX", "MDT", "BLK", "C", "DE", "UBER",
    "ADP", "PGR", "CB", "MMC", "PANW", "ETN", "CI", "SO", "MO", "DUK",
    "ZTS", "AON", "TJX", "ICE", "SCHW", "REGN", "CME", "CL", "ITW", "USB",
    "WM", "PNC", "EMR", "GM", "F", "TGT", "FDX", "NSC", "CARR", "COF",
    "NOC", "HUM", "GD", "LMT", "SLB", "ECL", "HCA", "EW", "OKE", "CTAS",
    "APD", "FTNT", "MCO", "WELL", "D", "PSX", "MPC", "VLO", "EOG", "COP",
    "PSA", "AMT", "EQIX", "O", "SPG", "PH", "AFL", "KLAC", "LRCX", "AMAT",
    "MCHP", "MU", "SNPS", "CDNS", "NXPI", "ON", "STZ", "KMB", "GIS", "HSY",
    "SHW", "PPG", "NEM", "FCX", "DOW", "LYB", "ALB", "VMC", "MLM", "NUE",
    "DHI", "LEN", "PHM", "NVR", "TOL", "FIS", "FI", "GPN", "PYPL", "SQ",
    "COIN", "RBLX", "SNAP", "PINS", "LYFT", "ABNB", "DASH", "DKNG", "HOOD",
    "PLTR", "SNOW", "DDOG", "CRWD", "ZS", "NET", "OKTA", "MDB", "GTLB",
    "TEAM", "HUBS", "VEEV", "WDAY", "NOW", "BILL", "TTD", "ROKU",
]


def get_sp500_tickers() -> list[str]:
    """Scrape S&P 500 constituent tickers from Wikipedia."""
    try:
        tables = pd.read_html(_SP500_URL)
        df = tables[0]
        tickers = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        return [t.strip() for t in tickers if t.strip()]
    except Exception as exc:
        print(f"  Warning: could not fetch S&P 500 tickers: {exc}", file=sys.stderr)
        return []


def get_nasdaq100_tickers() -> list[str]:
    """Scrape NASDAQ 100 tickers from Wikipedia."""
    try:
        tables = pd.read_html(_NASDAQ100_URL)
        for tbl in tables:
            if "Ticker" in tbl.columns:
                return tbl["Ticker"].astype(str).tolist()
            if "Symbol" in tbl.columns:
                return tbl["Symbol"].astype(str).tolist()
    except Exception as exc:
        print(f"  Warning: could not fetch NASDAQ-100 tickers: {exc}", file=sys.stderr)
    return []


# ---------------------------------------------------------------------------
# Technical indicator computation
# ---------------------------------------------------------------------------

def _rsi(prices: pd.Series, period: int = 14) -> float:
    """Compute RSI(period) using exponential smoothing."""
    delta = prices.diff().dropna()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - 100.0 / (1.0 + rs), 2)


def _sma_pct(prices: pd.Series, window: int) -> float:
    """Return % above/below SMA(window) as a decimal (e.g. 0.032 = +3.2%)."""
    if len(prices) < window:
        return np.nan
    sma = prices.rolling(window).mean().iloc[-1]
    last = prices.iloc[-1]
    if sma == 0:
        return np.nan
    return (last - sma) / sma


def _perf(prices: pd.Series, trading_days: int) -> float:
    """Return % price change over last *trading_days* as a decimal."""
    if len(prices) < trading_days + 1:
        return np.nan
    start = prices.iloc[-(trading_days + 1)]
    end = prices.iloc[-1]
    if start == 0:
        return np.nan
    return (end - start) / start


# ---------------------------------------------------------------------------
# Per-ticker info fetch
# ---------------------------------------------------------------------------

def _fetch_info(ticker: str) -> dict:
    """Fetch fundamental / analyst data for one ticker via yfinance."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        return {"ticker": ticker, "info": info}
    except Exception as exc:
        return {"ticker": ticker, "info": {}, "error": str(exc)}


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def fetch_all(max_stocks: int = 100, filters: str = "geo_usa",
              delay: float = 1.5) -> pd.DataFrame:
    """
    Fetch data via Yahoo Finance for up to *max_stocks* US equities.

    *filters* is accepted for API compatibility but only "geo_usa" and index
    sub-strings (sp500, ndx/nasdaq100) are acted upon.
    """
    # ── 1. Build ticker universe ──────────────────────────────────────────
    print("\nFetching ticker universe…", file=sys.stderr)

    tickers: list[str] = []
    filters_lower = filters.lower()

    if "ndx" in filters_lower or "nasdaq100" in filters_lower:
        tickers = get_nasdaq100_tickers()
        label = "NASDAQ-100"
    else:
        tickers = get_sp500_tickers()
        label = "S&P 500"
        if not tickers:
            tickers = get_nasdaq100_tickers()
            label = "NASDAQ-100 (fallback)"

    if not tickers:
        print("  Wikipedia unreachable — using built-in fallback ticker list.",
              file=sys.stderr)
        tickers = _FALLBACK_TICKERS
        label = "Fallback large-cap list"

    tickers = tickers[:max_stocks]
    print(f"  Universe: {len(tickers)} tickers ({label})", file=sys.stderr)

    # ── 2. Batch price download (1 year of daily closes) ─────────────────
    print(f"\nDownloading price history for {len(tickers)} tickers…", file=sys.stderr)
    raw_prices = yf.download(
        tickers,
        period="1y",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw_prices.columns, pd.MultiIndex):
        closes = raw_prices["Close"]
    else:
        closes = raw_prices[["Close"]]
        closes.columns = tickers[:1]

    print(f"  Price data shape: {closes.shape}", file=sys.stderr)

    # ── 3. Parallel fetch of Ticker.info ──────────────────────────────────
    print(f"\nFetching fundamentals for {len(tickers)} tickers (parallel)…",
          file=sys.stderr)

    info_map: dict[str, dict] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_fetch_info, t): t for t in tickers}
        for fut in as_completed(futs):
            result = fut.result()
            info_map[result["ticker"]] = result.get("info", {})
            done += 1
            if done % 25 == 0:
                print(f"  …{done}/{len(tickers)} fetched", file=sys.stderr)

    print(f"  Fundamentals fetched for {len(info_map)} tickers.", file=sys.stderr)

    # ── 4. Build rows ─────────────────────────────────────────────────────
    rows = []
    for ticker in tickers:
        info = info_map.get(ticker, {})
        price_series = closes.get(ticker, pd.Series(dtype=float)).dropna()

        if price_series.empty:
            continue

        current_price = price_series.iloc[-1]

        # Technical
        sma20  = _sma_pct(price_series, 20)
        sma50  = _sma_pct(price_series, 50)
        sma200 = _sma_pct(price_series, 200)
        rsi    = _rsi(price_series)
        p1m    = _perf(price_series, 21)
        p3m    = _perf(price_series, 63)
        p6m    = _perf(price_series, 126)
        p1y    = _perf(price_series, 252)

        # 52-week high/low proximity
        w52h_price = info.get("fiftyTwoWeekHigh", np.nan)
        w52l_price = info.get("fiftyTwoWeekLow", np.nan)
        w52h_pct = ((current_price - w52h_price) / w52h_price
                    if not np.isnan(w52h_price) and w52h_price > 0 else np.nan)
        w52l_pct = ((current_price - w52l_price) / w52l_price
                    if not np.isnan(w52l_price) and w52l_price > 0 else np.nan)

        def _pct_str(v) -> str:
            return f"{v * 100:.2f}%" if (v is not None and not np.isnan(v)) else "-"

        # Analyst / fundamentals from info
        recom         = info.get("recommendationMean", np.nan)
        target_price  = info.get("targetMeanPrice", np.nan)
        eps_next_y    = info.get("earningsGrowth", np.nan)
        eps_next_5y   = info.get("earningsQuarterlyGrowth", np.nan)
        # EPS Q/Q proxy: trailing vs. forward earnings growth differential
        eps_qq        = info.get("earningsGrowth", np.nan)
        insider_trans = info.get("heldPercentInsiders", np.nan)
        inst_trans    = info.get("heldPercentInstitutions", np.nan)
        float_short   = info.get("shortPercentOfFloat", np.nan)
        short_ratio   = info.get("shortRatio", np.nan)

        # yfinance insider/inst figures are totals, not 3M deltas — centre at typical levels
        insider_delta = (insider_trans - 0.10) if not np.isnan(insider_trans) else np.nan
        inst_delta    = (inst_trans    - 0.60) if not np.isnan(inst_trans)    else np.nan

        # Market cap: convert to "XB" string format for quality filter
        mktcap_raw = info.get("marketCap", np.nan)
        mktcap_str = (f"{mktcap_raw / 1e9:.2f}B"
                      if not np.isnan(mktcap_raw) else "-")

        row = {
            "Ticker":       ticker,
            "Company":      info.get("longName", ticker),
            "Sector":       info.get("sector", "-"),
            "Industry":     info.get("industry", "-"),
            "Market Cap":   mktcap_str,
            # Analyst signals
            "Recom":        recom,
            "Target Price": target_price if not np.isnan(target_price) else "-",
            "Price":        round(current_price, 2),
            # Technical
            "SMA20":        _pct_str(sma20),
            "SMA50":        _pct_str(sma50),
            "SMA200":       _pct_str(sma200),
            "52W High":     _pct_str(w52h_pct),
            "52W Low":      _pct_str(w52l_pct),
            "RSI":          round(rsi, 1),
            # Momentum
            "Perf Month":   _pct_str(p1m),
            "Perf Quart":   _pct_str(p3m),
            "Perf Half":    _pct_str(p6m),
            "Perf Year":    _pct_str(p1y),
            "Perf YTD":     "-",
            # Fundamentals
            "EPS Q/Q":      _pct_str(eps_qq)      if not np.isnan(eps_qq)      else "-",
            "EPS next Y":   _pct_str(eps_next_y)  if not np.isnan(eps_next_y)  else "-",
            "EPS next 5Y":  _pct_str(eps_next_5y) if not np.isnan(eps_next_5y) else "-",
            # Smart money
            "Insider Trans": _pct_str(insider_delta) if not np.isnan(insider_delta) else "-",
            "Inst Trans":    _pct_str(inst_delta)    if not np.isnan(inst_delta)    else "-",
            "Float Short":   _pct_str(float_short)   if not np.isnan(float_short)   else "-",
            "Short Ratio":   str(round(float(short_ratio), 1)) if not np.isnan(short_ratio) else "-",
            # Extra
            "Beta":          info.get("beta", np.nan),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\n  Built DataFrame: {len(df)} rows, {len(df.columns)} columns.", file=sys.stderr)
    return df
