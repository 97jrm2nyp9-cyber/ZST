"""Sub-scores and composite for the M&A target dashboard.

All sub-scores are normalized to 0-100. The composite is a weighted blend
defined in config.yaml. The strategic_fit sub-score is NOT computed here:
it's recomputed client-side from the acquirer-fit matrix so the dashboard
can switch acquirer lens without a refetch. We ship per-candidate
sector_tag and the matrix; the page does the dot product.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np


def _winsorize(values: list[float], lo_pct: float = 5, hi_pct: float = 95) -> list[float]:
    arr = np.array([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return values
    lo = np.percentile(arr, lo_pct)
    hi = np.percentile(arr, hi_pct)
    return [None if v is None else float(np.clip(v, lo, hi)) for v in values]


def _minmax(values: list[Optional[float]]) -> list[Optional[float]]:
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return [None] * len(values)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return [50.0 if v is not None else None for v in values]
    return [None if v is None else 100.0 * (v - lo) / (hi - lo) for v in values]


def deal_stage_score(stage: str, stage_map: dict) -> float:
    return float(stage_map.get(stage, 0))


def valuation_discount_scores(records: list[dict]) -> list[float]:
    """Cross-sectional: lower EV/EBITDA and P/E within peer_group => higher score.

    Returns a list aligned with `records`. Records missing both multiples get 50.
    """
    # Group indices by peer_group
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        groups.setdefault(r["peer_group"], []).append(i)

    out: list[Optional[float]] = [None] * len(records)
    for _, idxs in groups.items():
        evs = [records[i].get("ev_ebitda") for i in idxs]
        pes = [records[i].get("trailing_pe") for i in idxs]
        evs_w = _winsorize(evs)
        pes_w = _winsorize(pes)

        # Cross-sectional z-score within the group, inverted (cheap = high score).
        def zscore_inv(vals: list[Optional[float]]) -> list[Optional[float]]:
            arr = np.array([v for v in vals if v is not None], dtype=float)
            if arr.size < 2:
                return [None] * len(vals)
            mu, sd = float(arr.mean()), float(arr.std(ddof=0))
            if sd < 1e-12:
                return [0.0 if v is not None else None for v in vals]
            return [None if v is None else -(v - mu) / sd for v in vals]

        z_ev = zscore_inv(evs_w)
        z_pe = zscore_inv(pes_w)

        # Average the two z-scores per row, dropping Nones.
        combined: list[Optional[float]] = []
        for ze, zp in zip(z_ev, z_pe):
            vals = [v for v in (ze, zp) if v is not None]
            combined.append(sum(vals) / len(vals) if vals else None)

        scaled = _minmax(combined)
        for local_i, global_i in enumerate(idxs):
            out[global_i] = scaled[local_i]

    return [50.0 if v is None else v for v in out]


def premium_history_scores(records: list[dict], premium_map: dict) -> list[float]:
    pcts = [premium_map.get(r["sector_tag"], 25) for r in records]
    scaled = _minmax([float(p) for p in pcts])
    return [50.0 if v is None else v for v in scaled]


def liquidity_scores(records: list[dict], sector_median_dv: dict) -> list[float]:
    raw = []
    for r in records:
        dv = r.get("avg_dollar_volume_30d")
        if dv is None or dv <= 0:
            dv = sector_median_dv.get(r["sector_tag"])
        if dv is None or dv <= 0:
            raw.append(None)
        else:
            raw.append(math.log10(dv))
    scaled = _minmax(raw)
    return [50.0 if v is None else v for v in scaled]


def sector_median_dollar_volume(records: list[dict]) -> dict:
    by_sector: dict[str, list[float]] = {}
    for r in records:
        dv = r.get("avg_dollar_volume_30d")
        if dv and dv > 0:
            by_sector.setdefault(r["sector_tag"], []).append(dv)
    return {k: float(np.median(v)) for k, v in by_sector.items()}


def compose(records: list[dict], weights: dict) -> list[dict]:
    """Mutates records adding sub-scores and composite (excluding strategic_fit)."""
    for r in records:
        # Pre-composite (excludes strategic_fit which is computed client-side).
        w = weights
        # composite_base = composite without the strategic_fit term
        r["composite_base"] = (
            w["deal_stage"] * r["score_deal_stage"]
            + w["valuation_discount"] * r["score_valuation_discount"]
            + w["premium_history"] * r["score_premium_history"]
            + w["liquidity"] * r["score_liquidity"]
        )
    return records
