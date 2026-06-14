#!/usr/bin/env python3
"""
Stock Recommendation Engine — US Equities
==========================================
Scores and ranks US stocks from highest conviction buy to highest conviction sell.

Primary data source: finviz.com (direct HTML scraping — works on local machines).
Fallback:           Yahoo Finance / yfinance (works everywhere, including cloud).

The scoring model is identical for both sources:

  Component                      Max pts   Signal
  ────────────────────────────────────────────────
  Analyst consensus (Recom 1−5)     30     Finviz / yfinance analyst mean
  Target-price upside               20     (Target − Price) / Price
  SMA trend: 20 / 50 / 200-day      20     % above/below each moving average
  Price momentum: 1M / 3M / 6M      15     Trailing return capped at ±30%
  Earnings growth: next Y / 5Y      10     Forward EPS growth estimate
  Smart money: Insider + Inst         5     Ownership flow / delta
  ────────────────────────────────────────────────
  TOTAL                            100     Range: −100 (Strong Sell) to +100 (Strong Buy)

Usage examples
──────────────
  # Auto-detect best available source (finviz → yfinance fallback):
  python run_stock_recommender.py

  # Force Yahoo Finance (works in cloud / behind corporate firewalls):
  python run_stock_recommender.py --source yfinance

  # Force Finviz (local machine only — cloud IPs are blocked by finviz):
  python run_stock_recommender.py --source finviz

  # Larger universe, custom output file:
  python run_stock_recommender.py --stocks 500 --top 30 --output recs.csv

  # S&P 500 only (yfinance) or large-caps (finviz filter):
  python run_stock_recommender.py --source yfinance --filters sp500
  python run_stock_recommender.py --source finviz   --filters geo_usa,cap_largeover

  # NASDAQ-100:
  python run_stock_recommender.py --source yfinance --filters ndx

Finviz filter strings (--filters, comma-separated):
  geo_usa              US equities (default)
  cap_largeover        Market cap > $10 B
  cap_midover          Market cap > $2 B
  idx_sp500            S&P 500 index
  idx_ndx              NASDAQ 100 index
  sec_technology       Technology sector only

Yahoo Finance universe note:
  The yfinance source pulls the S&P 500 constituent list from Wikipedia
  (or NASDAQ-100 if you pass --filters ndx/nasdaq100).  Use --stocks N to
  limit the universe size (default 100).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_recommender.scorer import score
from stock_recommender.report import print_report, save_csv


# ---------------------------------------------------------------------------
# Connectivity probe
# ---------------------------------------------------------------------------

def _finviz_reachable() -> bool:
    """Return True if finviz.com returns a 200 (not blocked)."""
    import requests
    try:
        r = requests.get(
            "https://finviz.com/screener.ashx",
            params={"v": "111", "f": "geo_usa", "r": "1"},
            timeout=8,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
                )
            },
        )
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="US equity conviction ranker (Finviz / Yahoo Finance)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--source", choices=["auto", "finviz", "yfinance", "demo"], default="auto",
        help=(
            "Data source. 'auto' tries finviz → yfinance → demo in order. "
            "'demo' uses synthetic data (works offline). (default: auto)"
        ),
    )
    p.add_argument(
        "--stocks", "-s", type=int, default=100, metavar="N",
        help="Maximum stocks to score (default: 100)",
    )
    p.add_argument(
        "--top", "-t", type=int, default=25, metavar="N",
        help="Top N buys and sells shown in console (default: 25)",
    )
    p.add_argument(
        "--output", "-o", default="", metavar="PATH",
        help="CSV output path (default: stock_recommendations_YYYYMMDD.csv)",
    )
    p.add_argument(
        "--filters", "-f", default="geo_usa", metavar="STR",
        help="Finviz filter string or 'sp500'/'ndx' for yfinance universe (default: geo_usa)",
    )
    p.add_argument(
        "--delay", "-d", type=float, default=1.5, metavar="SECS",
        help="Seconds between page requests for finviz (default: 1.5, min: 1.0)",
    )
    p.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI colour in console output",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    delay = max(args.delay, 1.0)

    # ── Decide data source ────────────────────────────────────────────────
    source = args.source
    if source == "auto":
        print("\nProbing finviz.com connectivity…", file=sys.stderr)
        if _finviz_reachable():
            source = "finviz"
            print("  finviz.com reachable → using Finviz.", file=sys.stderr)
        else:
            print(
                "  finviz.com blocked (cloud/datacenter IPs are blocked by Finviz).\n"
                "  Trying Yahoo Finance…",
                file=sys.stderr,
            )
            source = "yfinance"

    if source == "yfinance":
        # Quick connectivity check for Yahoo Finance
        try:
            import requests as _req
            r = _req.get("https://finance.yahoo.com", timeout=5,
                         headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 403:
                raise ConnectionError("Yahoo Finance also blocked")
        except Exception:
            print(
                "  Yahoo Finance also not reachable.\n"
                "  Falling back to demo mode (synthetic data).\n"
                "  Run on a local machine for live data.",
                file=sys.stderr,
            )
            source = "demo"

    # ── Fetch ─────────────────────────────────────────────────────────────
    print(
        f"\nSource  : {source}\n"
        f"Universe: up to {args.stocks} stocks\n"
        f"Filters : {args.filters}",
        file=sys.stderr,
    )

    if source == "finviz":
        from stock_recommender.fetcher import fetch_all
    elif source == "yfinance":
        from stock_recommender.yf_fetcher import fetch_all
    else:
        from stock_recommender.demo_fetcher import fetch_all
        print(
            "\n  NOTE: Running in DEMO mode with synthetic data.\n"
            "  For live rankings, run on a local machine with internet access.\n"
            "  Use --source finviz (local) or --source yfinance (most environments).",
            file=sys.stderr,
        )

    raw = fetch_all(max_stocks=args.stocks, filters=args.filters, delay=delay)
    print(f"\n  Fetched {len(raw)} stocks.", file=sys.stderr)

    if raw.empty:
        print("Error: no data fetched — cannot produce recommendations.", file=sys.stderr)
        sys.exit(1)

    # ── Score ─────────────────────────────────────────────────────────────
    ranked = score(raw)
    print(f"  Scored  {len(ranked)} stocks.\n", file=sys.stderr)

    # ── Report ────────────────────────────────────────────────────────────
    use_color = not args.no_color and sys.stdout.isatty()
    print_report(ranked, top_n=args.top, use_color=use_color)

    # ── Save ──────────────────────────────────────────────────────────────
    from datetime import datetime
    out_path = args.output or f"stock_recommendations_{datetime.now().strftime('%Y%m%d')}.csv"
    save_csv(ranked, out_path)


if __name__ == "__main__":
    main()
