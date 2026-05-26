"""Refresh the M&A dashboard data file.

Reads universe.yaml + config.yaml, pulls live metrics from yfinance for
public candidates, computes sub-scores and a partial composite (excluding
strategic_fit, which is computed client-side from the acquirer matrix),
and writes docs/ma_dashboard/data/candidates.json.

Usage:
    python -m ma_dashboard.refresh
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import yaml

from ma_dashboard import data_sources, scoring

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_PATH = ROOT / "ma_dashboard" / "universe.yaml"
CONFIG_PATH = ROOT / "ma_dashboard" / "config.yaml"
OUT_DIR = ROOT / "docs" / "ma_dashboard" / "data"


def load_yaml(p: Path) -> dict:
    with p.open() as f:
        return yaml.safe_load(f)


_STATIC_FIELDS = (
    ("price", "static_price"),
    ("market_cap_b", "static_market_cap_b"),
    ("avg_dollar_volume_30d", "static_avg_dollar_volume_30d"),
    ("trailing_pe", "static_trailing_pe"),
    ("ev_ebitda", "static_ev_ebitda"),
)


def _apply_static_fallback(rec: dict, source: dict) -> None:
    """Fill any missing live metric from the static_* seed values."""
    for live_key, static_key in _STATIC_FIELDS:
        if rec.get(live_key) is None and source.get(static_key) is not None:
            rec[live_key] = source[static_key]


def build_records(universe: dict, log=print) -> list[dict]:
    records: list[dict] = []
    candidates = universe["candidates"]
    for c in candidates:
        rec = dict(c)
        if c.get("is_private"):
            rec.update(
                price=None,
                market_cap_b=c.get("static_market_cap_b"),
                avg_dollar_volume_30d=None,
                trailing_pe=None,
                ev_ebitda=None,
            )
            log(f"  {c['ticker']:>10}  private  cap={rec['market_cap_b']}")
        else:
            try:
                m = data_sources.fetch_metrics(c["ticker"])
                rec.update(data_sources.metrics_to_dict(m))
            except Exception as e:
                log(f"  {c['ticker']:>10}  fetch failed: {e}")
                rec.update(
                    price=None, market_cap_b=None, avg_dollar_volume_30d=None,
                    trailing_pe=None, ev_ebitda=None,
                )
            _apply_static_fallback(rec, c)
            log(
                f"  {c['ticker']:>10}  px={rec.get('price')}  cap_b={rec.get('market_cap_b')}  "
                f"ev/ebitda={rec.get('ev_ebitda')}  pe={rec.get('trailing_pe')}"
            )
        records.append(rec)
    return records


def score_all(records: list[dict], config: dict) -> list[dict]:
    weights = config["weights"]
    stage_map = config["deal_stage_scores"]
    premium_map = config["sector_premium_pct"]

    # Sub-scores
    for r in records:
        r["score_deal_stage"] = scoring.deal_stage_score(r["deal_stage"], stage_map)

    val_scores = scoring.valuation_discount_scores(records)
    prem_scores = scoring.premium_history_scores(records, premium_map)
    sector_dv = scoring.sector_median_dollar_volume(records)
    liq_scores = scoring.liquidity_scores(records, sector_dv)

    for r, v, p, l in zip(records, val_scores, prem_scores, liq_scores):
        r["score_valuation_discount"] = v
        r["score_premium_history"] = p
        r["score_liquidity"] = l
        # Est. takeout value (B) at sector-median premium.
        cap_b = r.get("market_cap_b")
        prem_pct = premium_map.get(r["sector_tag"], 25)
        r["est_takeout_value_b"] = (
            cap_b * (1 + prem_pct / 100.0) if cap_b is not None else None
        )

    scoring.compose(records, weights)
    return records


def main() -> int:
    universe = load_yaml(UNIVERSE_PATH)
    config = load_yaml(CONFIG_PATH)

    print(f"Refreshing {len(universe['candidates'])} candidates...")
    records = build_records(universe)
    records = score_all(records, config)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates_path = OUT_DIR / "candidates.json"
    updated_path = OUT_DIR / "last_updated.json"

    payload = {
        "weights": config["weights"],
        "acquirer_fit": config["acquirer_fit"],
        "sector_premium_pct": config["sector_premium_pct"],
        "candidates": records,
    }
    candidates_path.write_text(json.dumps(payload, indent=2, default=str))

    meta = {
        "generated_at_utc": dt.datetime.utcnow().isoformat() + "Z",
        "candidate_count": len(records),
    }
    updated_path.write_text(json.dumps(meta, indent=2))

    print(f"Wrote {candidates_path} ({len(records)} rows)")
    print(f"Wrote {updated_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
