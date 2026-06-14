"""
Enhanced conviction scoring model — v2.

Key improvements over v1:

  1. Sector-relative z-scoring on all momentum/technical signals.
     Eliminates sector hot-streak bias: a tech stock now competes against
     other tech stocks, not against utilities.

  2. Continuous SMA signal — weighted % deviation replaces binary above/below.
     A stock 15% above its 200-day MA is meaningfully different from 0.1%.

  3. Analyst consensus weight: 30 → 10 pts.
     Analysts are consensus-following and systematically late.

  4. Earnings revision momentum: 0 → 20 pts (new, highest weight).
     EPS Q/Q (recent beat vs year-ago) + EPS next Y (forward direction).
     Much higher predictive power than static analyst ratings.

  5. RSI scored (5 pts, non-linear) — oversold bounce / overbought fade.

  6. Cross-sectional relative strength (10 pts, global z-score) — how the
     stock ranks vs. the full universe on 3M + 6M combined.

  7. 52-week high proximity (5 pts) — breakout/continuation signal.

  8. Short interest added to smart money (bearish pressure).

  9. Quality pre-filter — micro-caps removed before z-scoring.

Score range: −100 (Strong Sell) → +100 (Strong Buy)

  Component                          Max pts   Method
  ──────────────────────────────────────────────────────────────────
  Earnings revision momentum            20     EPS Q/Q + EPS next Y, sector z-score
  SMA trend (20/50/200, continuous)     15     Weighted % deviation, sector z-score
  Momentum 1M/3M/6M/12M                15     Composite perf, sector z-score
  Cross-sectional relative strength    10     3M+6M perf rank, global z-score
  Target-price upside                  10     (Target − Price) / Price, absolute
  Analyst consensus                    10     Recom 1-5 inverted, sector z-score
  RSI signal                            5     Piecewise: oversold+, overbought−
  52-week high proximity                5     % from 52W high, sector z-score
  Fundamentals (EPS next 5Y)            5     Sector z-score
  Smart money                           5     Insider + Inst − Short interest
  ──────────────────────────────────────────────────────────────────
  TOTAL                               100
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

Z_CAP = 2.5  # z-score outlier cap (±2.5σ)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _to_float(s) -> float:
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
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    s = str(s).strip().rstrip("%")
    if not s or s == "-":
        return np.nan
    try:
        return float(s) / 100.0
    except ValueError:
        return np.nan


def _col_float(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df[col].map(_to_float)


def _col_pct(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df[col].map(_pct_to_float)


def _parse_mktcap(s) -> float:
    """'1.23B' → 1.23, '456M' → 0.456, '2.1T' → 2100  (all in billions)."""
    if pd.isna(s):
        return np.nan
    s = str(s).upper().strip().replace(",", "")
    if s in ("-", "", "NAN"):
        return np.nan
    try:
        if s.endswith("T"):
            return float(s[:-1]) * 1_000
        if s.endswith("B"):
            return float(s[:-1])
        if s.endswith("M"):
            return float(s[:-1]) / 1_000
        if s.endswith("K"):
            return float(s[:-1]) / 1_000_000
        return float(s) / 1e9
    except (ValueError, TypeError):
        return np.nan


# ---------------------------------------------------------------------------
# Z-scoring infrastructure
# ---------------------------------------------------------------------------

def _global_zscore(series: pd.Series) -> pd.Series:
    """Z-score across the full universe, capped at ±Z_CAP."""
    filled = series.fillna(series.median() if not series.isna().all() else 0.0)
    mu, std = filled.mean(), filled.std()
    if std < 1e-9:
        return pd.Series(0.0, index=series.index)
    return ((filled - mu) / std).clip(-Z_CAP, Z_CAP)


def _sector_zscore(series: pd.Series, sectors: pd.Series,
                   min_group: int = 4) -> pd.Series:
    """
    Z-score within each sector.  Falls back to global z-score for sectors
    with fewer than *min_group* members (avoids noisy estimates).
    """
    global_z = _global_zscore(series)
    filled = series.fillna(series.median() if not series.isna().all() else 0.0)
    result = pd.Series(0.0, index=series.index)
    sec = sectors.fillna("Unknown")

    for sector in sec.unique():
        idx = sec[sec == sector].index
        if len(idx) < min_group:
            result.loc[idx] = global_z.loc[idx]
        else:
            g = filled.loc[idx]
            mu, std = g.mean(), g.std()
            if std < 1e-9:
                result.loc[idx] = 0.0
            else:
                result.loc[idx] = ((g - mu) / std).clip(-Z_CAP, Z_CAP)

    return result


def _z_to_pts(z: pd.Series, max_pts: float) -> pd.Series:
    """Scale z-scores in [−Z_CAP, +Z_CAP] to [−max_pts, +max_pts]."""
    return (z / Z_CAP * max_pts).fillna(0.0)


# ---------------------------------------------------------------------------
# Quality pre-filter
# ---------------------------------------------------------------------------

def apply_quality_filter(df: pd.DataFrame, min_mktcap_b: float = 0.3) -> pd.DataFrame:
    """
    Remove stocks below *min_mktcap_b* billion market cap.
    Stocks with missing market cap data are kept (benefit of the doubt).
    """
    if "Market Cap" not in df.columns:
        return df
    mktcap = df["Market Cap"].map(_parse_mktcap)
    mask = mktcap.isna() | (mktcap >= min_mktcap_b)
    removed = (~mask).sum()
    if removed:
        print(f"  Quality filter: removed {removed} stocks below ${min_mktcap_b}B market cap.",
              file=sys.stderr)
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Signal scoring functions  (all take df + sectors pd.Series)
# ---------------------------------------------------------------------------

def _score_earnings_revision(df: pd.DataFrame, sectors: pd.Series) -> pd.Series:
    """
    20 pts — best single predictor of forward returns.
    Proxy: EPS Q/Q (60%) + EPS next Y (40%), both sector z-scored.
    EPS Q/Q captures recent earnings beats; EPS next Y captures forward revision direction.
    """
    eps_qq = _col_pct(df, "EPS Q/Q").fillna(0.0)
    eps_ny = _col_pct(df, "EPS next Y").fillna(0.0)
    composite = 0.60 * eps_qq + 0.40 * eps_ny
    return _z_to_pts(_sector_zscore(composite, sectors), 20.0)


def _score_analyst(df: pd.DataFrame, sectors: pd.Series) -> pd.Series:
    """
    10 pts — analyst mean recommendation, sector z-scored.
    Inverted so that 1.0 (Strong Buy) → highest signal value.
    """
    recom = _col_float(df, "Recom")
    inverted = (6.0 - recom).fillna(3.0)  # neutral fill at midpoint
    return _z_to_pts(_sector_zscore(inverted, sectors), 10.0)


def _score_upside(df: pd.DataFrame) -> pd.Series:
    """
    10 pts — analyst consensus target upside, absolute (not sector-relative).
    Capped at ±50% to limit staleness impact.
    """
    target = _col_float(df, "Target Price")
    price = _col_float(df, "Price")
    upside = (target - price) / price.replace(0, np.nan)
    return ((upside / 0.50).clip(-1, 1) * 10.0).fillna(0.0)


def _score_sma_continuous(df: pd.DataFrame, sectors: pd.Series) -> pd.Series:
    """
    15 pts — continuous weighted % deviation from SMAs, sector z-scored.
    Longer SMAs get higher weight (more persistent signal).
    """
    sma20  = _col_pct(df, "SMA20").fillna(0.0)
    sma50  = _col_pct(df, "SMA50").fillna(0.0)
    sma200 = _col_pct(df, "SMA200").fillna(0.0)
    composite = 0.25 * sma20 + 0.35 * sma50 + 0.40 * sma200
    return _z_to_pts(_sector_zscore(composite, sectors), 15.0)


def _score_momentum(df: pd.DataFrame, sectors: pd.Series) -> pd.Series:
    """
    15 pts — composite momentum 1M/3M/6M/12M, sector z-scored.
    Increasing weight on longer windows (12M momentum is strongest predictor).
    """
    p1m  = _col_pct(df, "Perf Month").fillna(0.0)
    p3m  = _col_pct(df, "Perf Quart").fillna(0.0)
    p6m  = _col_pct(df, "Perf Half").fillna(0.0)
    p12m = _col_pct(df, "Perf Year").fillna(0.0)
    composite = 0.10 * p1m + 0.15 * p3m + 0.25 * p6m + 0.50 * p12m
    return _z_to_pts(_sector_zscore(composite, sectors), 15.0)


def _score_relative_strength(df: pd.DataFrame) -> pd.Series:
    """
    10 pts — cross-sectional relative strength, GLOBAL z-score (not sector).
    Ranks 3M + 6M combined performance vs. the full universe.
    Captures stocks outperforming regardless of sector.
    """
    p3m = _col_pct(df, "Perf Quart").fillna(0.0)
    p6m = _col_pct(df, "Perf Half").fillna(0.0)
    composite = 0.40 * p3m + 0.60 * p6m
    return _z_to_pts(_global_zscore(composite), 10.0)


def _score_rsi(df: pd.DataFrame) -> pd.Series:
    """
    5 pts — non-linear RSI signal. Absolute (sector z-scoring would remove
    the meaningful information that the whole sector is overbought).
    Oversold (<30): bullish mean-reversion.
    Overbought (>70): bearish mean-reversion.
    """
    def _signal(v: float) -> float:
        if np.isnan(v):
            return 0.0
        if v <= 30:
            return 1.0
        if v <= 50:
            return (50 - v) / 20.0   # linear +1→0
        if v <= 70:
            return -(v - 50) / 40.0  # linear 0→-0.5
        return -1.0

    rsi = _col_float(df, "RSI")
    return (rsi.map(_signal) * 5.0).fillna(0.0)


def _score_52w_high(df: pd.DataFrame, sectors: pd.Series) -> pd.Series:
    """
    5 pts — proximity to 52-week high, sector z-scored.
    Finviz '52W High' = % the stock is BELOW its 52W high (e.g. '-5%').
    Higher value (closer to 0) = nearer to yearly high = breakout signal.
    """
    w52h = _col_pct(df, "52W High")
    filled = w52h.fillna(w52h.median() if not w52h.isna().all() else -0.20)
    return _z_to_pts(_sector_zscore(filled, sectors), 5.0)


def _score_fundamentals(df: pd.DataFrame, sectors: pd.Series) -> pd.Series:
    """
    5 pts — long-term earnings growth (EPS next 5Y), sector z-scored.
    """
    eps5y = _col_pct(df, "EPS next 5Y").fillna(0.0)
    return _z_to_pts(_sector_zscore(eps5y, sectors), 5.0)


def _score_smart_money(df: pd.DataFrame, sectors: pd.Series) -> pd.Series:
    """
    5 pts — insider buying + institutional accumulation − short interest.
    High float short = bearish overhang, penalised.
    """
    insider = _col_pct(df, "Insider Trans").fillna(0.0)
    inst    = _col_pct(df, "Inst Trans").fillna(0.0)
    short   = _col_pct(df, "Float Short").fillna(
        _col_pct(df, "Float Short").median() if "Float Short" in df.columns else 0.10
    )
    composite = 0.40 * insider + 0.40 * inst - 0.20 * short
    return _z_to_pts(_sector_zscore(composite, sectors), 5.0)


# ---------------------------------------------------------------------------
# Conviction labels
# ---------------------------------------------------------------------------

_LABELS = [
    (60,   "Strong Buy"),
    (30,   "Buy"),
    (10,   "Mild Buy"),
    (-10,  "Neutral"),
    (-30,  "Mild Sell"),
    (-60,  "Sell"),
    (-101, "Strong Sell"),
]


def _label(s: float) -> str:
    for threshold, name in _LABELS:
        if s >= threshold:
            return name
    return "Strong Sell"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply quality filter, compute all signals, return ranked DataFrame.
    """
    # Quality filter first (before z-scoring so small sectors don't skew)
    df = apply_quality_filter(df)

    if df.empty:
        print("Warning: DataFrame is empty after quality filter.", file=sys.stderr)
        return df

    sectors = df.get("Sector", pd.Series("Unknown", index=df.index))
    if isinstance(sectors, pd.Series) and "Sector" in df.columns:
        sectors = df["Sector"].fillna("Unknown")
    else:
        sectors = pd.Series("Unknown", index=df.index)

    out = pd.DataFrame(index=df.index)
    out["Ticker"] = df.get("Ticker", pd.Series(range(len(df)), index=df.index)).values

    # ── Compute each signal ────────────────────────────────────────────────
    out["earnings_revision_score"] = _score_earnings_revision(df, sectors).values
    out["analyst_score"]           = _score_analyst(df, sectors).values
    out["upside_score"]            = _score_upside(df).values
    out["sma_score"]               = _score_sma_continuous(df, sectors).values
    out["momentum_score"]          = _score_momentum(df, sectors).values
    out["rel_strength_score"]      = _score_relative_strength(df).values
    out["rsi_score"]               = _score_rsi(df).values
    out["w52h_score"]              = _score_52w_high(df, sectors).values
    out["fundamental_score"]       = _score_fundamentals(df, sectors).values
    out["smart_money_score"]       = _score_smart_money(df, sectors).values

    _signal_cols = [
        "earnings_revision_score", "analyst_score", "upside_score",
        "sma_score", "momentum_score", "rel_strength_score",
        "rsi_score", "w52h_score", "fundamental_score", "smart_money_score",
    ]
    out["conviction_score"] = out[_signal_cols].sum(axis=1).clip(-100, 100).round(1)
    out["conviction_label"] = out["conviction_score"].map(_label)

    # ── Pass through raw fields for display / CSV ─────────────────────────
    _passthrough = [
        "Company", "Sector", "Industry", "Market Cap",
        "Recom", "Target Price", "Price",
        "RSI", "Beta",
        "Perf Month", "Perf Quart", "Perf Half", "Perf Year", "Perf YTD",
        "SMA20", "SMA50", "SMA200",
        "52W High", "52W Low",
        "EPS Q/Q", "EPS next Y", "EPS next 5Y",
        "Insider Trans", "Inst Trans",
        "Float Short", "Short Ratio",
    ]
    for col in _passthrough:
        if col in df.columns:
            out[col] = df[col].values

    return out.sort_values("conviction_score", ascending=False).reset_index(drop=True)
