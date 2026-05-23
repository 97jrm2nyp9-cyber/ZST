"""
Console and CSV report formatting for the recommendation engine.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd


_LABEL_COLORS = {
    "Strong Buy":  "\033[92m",   # bright green
    "Buy":         "\033[32m",   # green
    "Mild Buy":    "\033[36m",   # cyan
    "Neutral":     "\033[37m",   # white
    "Mild Sell":   "\033[33m",   # yellow
    "Sell":        "\033[31m",   # red
    "Strong Sell": "\033[91m",   # bright red
}
_RESET = "\033[0m"


def _color(label: str, text: str, use_color: bool) -> str:
    if not use_color:
        return text
    return f"{_LABEL_COLORS.get(label, '')}{text}{_RESET}"


def _fmt(val, fmt=".1f") -> str:
    """Format a value for display, returning '-' for NaN/None."""
    try:
        if val is None or (isinstance(val, float) and val != val):
            return "-"
        return format(float(val), fmt)
    except (ValueError, TypeError):
        return str(val) if val else "-"


def print_report(ranked: pd.DataFrame, top_n: int = 25, use_color: bool = True) -> None:
    """Print a formatted ranking table to stdout."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(ranked)

    print("=" * 108)
    print("  STOCK RECOMMENDATION ENGINE  –  Finviz US Equity Rankings")
    print(f"  Generated : {now}")
    print(f"  Universe  : {total} stocks scored")
    print("=" * 108)
    print()
    print("  Conviction score: −100 (Strong Sell) → +100 (Strong Buy)")
    print()
    print("  Score weights: Analyst Consensus 30% | Target Upside 20% | SMA Trend 20%")
    print("                 Momentum 15% (1M×3+3M×3+6M×4+12M×5) | Earnings Growth 10% | Smart Money 5%")
    print()

    def _section(subset: pd.DataFrame, title: str) -> None:
        print(f"{'─' * 108}")
        print(f"  {title}")
        print(f"{'─' * 108}")
        hdr = (
            f"  {'#':>3}  {'Ticker':<8}  {'Company':<26}  {'Sector':<18}  "
            f"{'Score':>6}  {'Label':<12}  "
            f"{'Recom':>5}  {'Upside':>7}  {'RSI':>5}  "
            f"{'1M':>6}  {'3M':>6}  {'6M':>6}  {'12M':>7}  "
            f"{'SMA50':>6}  {'SMA200':>7}"
        )
        print(hdr)
        print(f"  {'─'*3}  {'─'*8}  {'─'*26}  {'─'*18}  {'─'*6}  {'─'*12}  "
              f"{'─'*5}  {'─'*7}  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*7}")

        for i, (_, row) in enumerate(subset.iterrows(), 1):
            ticker  = str(row.get("Ticker",  "-"))
            company = str(row.get("Company", "-"))[:26]
            sector  = str(row.get("Sector",  "-"))[:18]
            score   = row.get("conviction_score", 0)
            label   = str(row.get("conviction_label", "-"))
            recom   = _fmt(row.get("Recom"))
            rsi     = _fmt(row.get("RSI"))
            sma50   = str(row.get("SMA50",  "-"))
            sma200  = str(row.get("SMA200", "-"))
            p1m     = str(row.get("Perf Month",  "-"))
            p3m     = str(row.get("Perf Quart",  "-"))
            p6m     = str(row.get("Perf Half",   "-"))
            p12m    = str(row.get("Perf Year",   "-"))

            # Target upside
            try:
                tp = float(str(row.get("Target Price") or "").replace(",", ""))
                px = float(str(row.get("Price") or "").replace(",", ""))
                upside_str = f"{(tp - px) / px * 100:+.1f}%"
            except Exception:
                upside_str = "-"

            score_str = _color(label, f"{score:+.1f}", use_color)
            label_str = _color(label, f"{label:<12}", use_color)

            print(
                f"  {i:>3}  {ticker:<8}  {company:<26}  {sector:<18}  "
                f"{score_str:>6}  {label_str}  "
                f"{recom:>5}  {upside_str:>7}  {rsi:>5}  "
                f"{p1m:>6}  {p3m:>6}  {p6m:>6}  {p12m:>7}  "
                f"{sma50:>6}  {sma200:>7}"
            )
        print()

    # Top buys
    buys = ranked[ranked["conviction_score"] > 0].head(top_n)
    _section(buys, f"TOP {len(buys)} CONVICTION BUYS  (highest score first)")

    # Top sells
    sells = ranked[ranked["conviction_score"] < 0].tail(top_n).iloc[::-1]
    _section(sells, f"TOP {len(sells)} CONVICTION SELLS  (lowest score first)")

    # Distribution
    print("─" * 108)
    print("  SCORE DISTRIBUTION")
    print("─" * 108)
    labels_order = ["Strong Buy", "Buy", "Mild Buy", "Neutral", "Mild Sell", "Sell", "Strong Sell"]
    counts = ranked["conviction_label"].value_counts()
    for lbl in labels_order:
        n = counts.get(lbl, 0)
        bar = "█" * int(n / max(counts.values) * 40) if counts.values.max() > 0 else ""
        print(f"  {lbl:<14}  {n:>4}  {bar}")
    print()
    print("=" * 108)


def save_csv(ranked: pd.DataFrame, path: str) -> None:
    """Write the full ranked list to CSV."""
    ranked.to_csv(path, index=False)
    print(f"\n  Full rankings saved → {path}")
