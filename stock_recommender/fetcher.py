"""
Finviz screener data fetcher.

Fetches four complementary views and merges them by Ticker:
  v=151  Performance  – Recom, Perf W/M/Q/H/Y/YTD, Price
  v=161  Technical    – SMA20/50/200, RSI, Beta, 52W H/L
  v=131  Financial    – EPS this/next Y, EPS next 5Y, Target Price
  v=141  Ownership    – Insider Trans, Inst Trans, Float Short
"""

import time
import sys

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

SCREENER_URL = "https://finviz.com/screener.ashx"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
}

# Columns that appear in multiple views – keep only from the first (primary) view
_SHARED_COLS = {"No.", "Company", "Sector", "Industry", "Country",
                "Market Cap", "Avg Volume", "Rel Volume", "Price",
                "Change", "Volume", "Recom", "Optionable", "Shortable"}


# ---------------------------------------------------------------------------
# Low-level page scraper
# ---------------------------------------------------------------------------

def _scrape_page(view_id: int, offset: int, filters: str) -> tuple[list[str] | None, list[list[str]]]:
    """Return (col_names, data_rows) for one page. col_names may be None on failure."""
    params = {"v": view_id, "f": filters, "r": offset}
    try:
        resp = requests.get(SCREENER_URL, params=params, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  [v={view_id} r={offset}] HTTP error: {exc}", file=sys.stderr)
        return None, []

    soup = BeautifulSoup(resp.text, "html.parser")

    # Finviz has used different table identifiers across versions – try all
    table = (
        soup.find("table", {"class": "screener_table"})
        or soup.find("table", id="screener-content")
        or soup.find("table", {"class": "table-light"})
    )
    if table is None:
        # Fall back: largest table by number of rows
        candidates = soup.find_all("table")
        table = max(candidates, key=lambda t: len(t.find_all("tr")), default=None)

    if table is None:
        return None, []

    rows = table.find_all("tr")
    if not rows:
        return None, []

    # Header row
    header_cells = rows[0].find_all(["td", "th"])
    col_names = [c.get_text(strip=True) for c in header_cells]

    # Data rows
    data_rows: list[list[str]] = []
    for row in rows[1:]:
        cells = row.find_all("td")
        if len(cells) < 3:
            continue
        data_rows.append([c.get_text(strip=True) for c in cells])

    return col_names, data_rows


# ---------------------------------------------------------------------------
# Multi-page view fetcher
# ---------------------------------------------------------------------------

def fetch_view(view_id: int, filters: str = "geo_usa", max_stocks: int = 100,
               delay: float = 1.5) -> pd.DataFrame:
    """
    Fetch one finviz screener view up to *max_stocks* rows.
    Returns a DataFrame (may be empty on failure).
    """
    col_names: list[str] | None = None
    all_rows: list[list[str]] = []
    fetched = 0
    page = 1

    while fetched < max_stocks:
        offset = fetched + 1
        cols, rows = _scrape_page(view_id, offset, filters)

        if cols is None or not rows:
            print(f"  [v={view_id}] No data on page {page} – stopping.", file=sys.stderr)
            break

        if col_names is None:
            col_names = cols

        # Align row width to header
        n = len(col_names)
        for r in rows:
            padded = r[:n] if len(r) >= n else r + [""] * (n - len(r))
            all_rows.append(padded)

        n_new = len(rows)
        fetched += n_new
        print(f"  [v={view_id}] page {page}: +{n_new} ({fetched} total)", file=sys.stderr)

        if n_new < 20:
            break  # last page

        page += 1
        time.sleep(delay)

    if not all_rows or col_names is None:
        return pd.DataFrame()

    return pd.DataFrame(all_rows, columns=col_names)


# ---------------------------------------------------------------------------
# Column-name normalisation
# ---------------------------------------------------------------------------

_COL_ALIASES: dict[str, str] = {
    "Perf Quarter":   "Perf Quart",
    "Perf Half Y":    "Perf Half",
    "Volatility W":   "Volatility W",
    "Volatility M":   "Volatility M",
    "52W High":       "52W High",
    "52W Low":        "52W Low",
    "Change from Open": "Change from Open",
}


def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=_COL_ALIASES)
    # Drop row-number column if present
    df = df.drop(columns=["No."], errors="ignore")
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_all(max_stocks: int = 100, filters: str = "geo_usa",
              delay: float = 1.5) -> pd.DataFrame:
    """
    Fetch and merge all four finviz views.
    Returns a merged DataFrame indexed by Ticker.
    """
    view_specs = [
        (151, "Performance"),
        (161, "Technical"),
        (131, "Financial"),
        (141, "Ownership"),
    ]

    merged: pd.DataFrame | None = None

    for view_id, label in view_specs:
        print(f"\nFetching {label} view (v={view_id})...", file=sys.stderr)
        df = fetch_view(view_id, filters=filters, max_stocks=max_stocks, delay=delay)

        if df.empty:
            print(f"  Warning: {label} view returned no data.", file=sys.stderr)
            continue

        df = _normalise_cols(df)

        if "Ticker" not in df.columns:
            print(f"  Warning: {label} view has no Ticker column "
                  f"(cols={list(df.columns)[:8]}...). Skipping.", file=sys.stderr)
            continue

        df = df.set_index("Ticker")
        df = df.loc[:, ~df.columns.duplicated()]

        if merged is None:
            merged = df
        else:
            # Drop columns already present so we don't create _x / _y duplicates
            new_cols = [c for c in df.columns if c not in merged.columns]
            merged = merged.join(df[new_cols], how="outer")

    if merged is None:
        print("Error: No data returned from any Finviz view.", file=sys.stderr)
        sys.exit(1)

    return merged.reset_index()
