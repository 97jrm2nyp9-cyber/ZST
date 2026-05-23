"""
Conviction scoring model.

Each stock receives a total score in [-100, +100]:

  Component              Max pts   Signal used
  ─────────────────────────────────────────────
  Analyst consensus         30     Finviz Recom (1=Strong Buy … 5=Strong Sell)
  Target-price upside       20     (Target Price − Price) / Price
  SMA trend (20/50/200)     20     % above/below each SMA (5+7+8 pts)
  Price momentum            15     Perf Month + Perf Quart + Perf Half (5 ea)
  Earnings growth           10     EPS next Y + EPS next 5Y (5 ea)
  Smart money                5     Insider Trans + Inst Trans (2.5 ea)
  ─────────────────────────────────────────────
  TOTAL                    100

Positive score → conviction buy; negative → conviction sell.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _to_float(s) -> float:
    """'12.34' or '-12.34' → float; '-' / empty → NaN."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    s = str(s).strip()
    if not s or s == "-":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _pct_to_float(s) -> float:
    """'5.23%' → 0.0523; '-3.14%' → -0.0314; '-' / empty → NaN."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    s = str(s).strip().rstrip("%")
    if not s or s == "-":
        return np.nan
    try:
        return float(s) / 100.0
    except ValueError:
        return np.nan


def _series_float(df: pd.DataFrame, col: str) -> pd.Series:
    """Extract a column as float, returning NaN for missing column."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df[col].map(_to_float)


def _series_pct(df: pd.DataFrame, col: str) -> pd.Series:
    """Extract a percentage column as float (0.xx), returning NaN for missing."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df[col].map(_pct_to_float)


# ---------------------------------------------------------------------------
# Individual scoring components
# ---------------------------------------------------------------------------

def _score_analyst(df: pd.DataFrame) -> pd.Series:
    """
    Finviz Recom: 1.0 = Strong Buy, 3.0 = Hold, 5.0 = Strong Sell.
    Maps linearly → [-30, +30].
    """
    recom = _series_float(df, "Recom")
    score = (3.0 - recom) / 2.0 * 30.0
    return score.clip(-30, 30).fillna(0.0)


def _score_upside(df: pd.DataFrame) -> pd.Series:
    """
    Analyst target-price upside capped at ±50% → [-20, +20].
    """
    target = _series_float(df, "Target Price")
    price = _series_float(df, "Price")
    upside = (target - price) / price.replace(0, np.nan)
    score = (upside / 0.50).clip(-1, 1) * 20.0
    return score.fillna(0.0)


def _score_sma(df: pd.DataFrame) -> pd.Series:
    """
    SMA20 (5 pts), SMA50 (7 pts), SMA200 (8 pts).
    Finviz shows each as % above/below SMA (e.g. '+3.21%' or '-1.50%').
    """
    def _component(col: str, pts: float) -> pd.Series:
        vals = _series_pct(df, col)
        # Non-zero value: +pts if above, −pts if below, 0 if NaN
        return vals.map(lambda v: pts if (not np.isnan(v) and v > 0)
                        else (-pts if (not np.isnan(v) and v < 0) else 0.0))

    return _component("SMA20", 5.0) + _component("SMA50", 7.0) + _component("SMA200", 8.0)


def _score_momentum(df: pd.DataFrame) -> pd.Series:
    """
    1-month (5 pts), 3-month (5 pts), 6-month (5 pts) performance.
    Each capped at ±30% → linear mapping.
    """
    def _perf(col: str, pts: float, cap: float = 0.30) -> pd.Series:
        vals = _series_pct(df, col)
        return ((vals / cap).clip(-1, 1) * pts).fillna(0.0)

    return _perf("Perf Month", 5.0) + _perf("Perf Quart", 5.0) + _perf("Perf Half", 5.0)


def _score_fundamentals(df: pd.DataFrame) -> pd.Series:
    """
    EPS next year (5 pts) + EPS next 5Y (5 pts).
    Capped at ±25% annual growth → linear mapping.
    """
    def _eps(col: str, pts: float, cap: float = 0.25) -> pd.Series:
        vals = _series_pct(df, col)
        return ((vals / cap).clip(-1, 1) * pts).fillna(0.0)

    return _eps("EPS next Y", 5.0) + _eps("EPS next 5Y", 5.0)


def _score_smart_money(df: pd.DataFrame) -> pd.Series:
    """
    Insider Trans (2.5 pts) + Inst Trans (2.5 pts).
    3-month change in insider/institutional ownership, capped at ±10%.
    """
    def _flow(col: str, pts: float, cap: float = 0.10) -> pd.Series:
        vals = _series_pct(df, col)
        return ((vals / cap).clip(-1, 1) * pts).fillna(0.0)

    return _flow("Insider Trans", 2.5) + _flow("Inst Trans", 2.5)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

_LABELS = [
    (60,  "Strong Buy"),
    (30,  "Buy"),
    (10,  "Mild Buy"),
    (-10, "Neutral"),
    (-30, "Mild Sell"),
    (-60, "Sell"),
    (-101, "Strong Sell"),
]


def _label(score: float) -> str:
    for threshold, name in _LABELS:
        if score >= threshold:
            return name
    return "Strong Sell"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute conviction scores for each row in *df*.

    Returns a DataFrame with columns:
      Ticker, conviction_score, conviction_label,
      analyst_score, upside_score, sma_score,
      momentum_score, fundamental_score, smart_money_score,
      Recom, Target Price, Price, RSI,
      Perf Month, Perf Quart, Perf Half,
      SMA20, SMA50, SMA200,
      EPS next Y, EPS next 5Y,
      Insider Trans, Inst Trans,
      Sector, Market Cap
    sorted by conviction_score descending.
    """
    out = pd.DataFrame()
    out["Ticker"] = df.get("Ticker", pd.Series(range(len(df)))).values

    out["analyst_score"]     = _score_analyst(df).values
    out["upside_score"]      = _score_upside(df).values
    out["sma_score"]         = _score_sma(df).values
    out["momentum_score"]    = _score_momentum(df).values
    out["fundamental_score"] = _score_fundamentals(df).values
    out["smart_money_score"] = _score_smart_money(df).values

    out["conviction_score"] = (
        out["analyst_score"]
        + out["upside_score"]
        + out["sma_score"]
        + out["momentum_score"]
        + out["fundamental_score"]
        + out["smart_money_score"]
    ).clip(-100, 100).round(1)

    out["conviction_label"] = out["conviction_score"].map(_label)

    # Attach useful raw fields for display
    _passthrough = [
        "Company", "Sector", "Industry", "Market Cap",
        "Recom", "Target Price", "Price",
        "RSI",
        "Perf Month", "Perf Quart", "Perf Half", "Perf Year", "Perf YTD",
        "SMA20", "SMA50", "SMA200",
        "EPS next Y", "EPS next 5Y",
        "Insider Trans", "Inst Trans",
        "Beta", "Float Short",
    ]
    for col in _passthrough:
        if col in df.columns:
            out[col] = df[col].values

    return out.sort_values("conviction_score", ascending=False).reset_index(drop=True)
