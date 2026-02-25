"""
tests/test_web_research_policy.py
=================================
Research policy trigger + evidence score fixed-rule tests.
"""

from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines.research_policy import (
    cap_requests_by_budget,
    compute_evidence_score,
    should_run_web_research,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _item(kind: str, *, ticker: str = "AAPL", days_old: int = 0, trust_tier: float = 1.0, title: str = "beat", snippet: str = "") -> dict:
    ts = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    return {
        "url": f"https://sec.gov/{kind}/{days_old}",
        "title": title,
        "published_at": ts,
        "snippet": snippet,
        "source": "sec.gov",
        "retrieved_at": ts,
        "hash": f"{kind}-{days_old}-{ticker}",
        "kind": kind,
        "desk": "fundamental",
        "ticker": ticker,
        "trust_tier": trust_tier,
        "resolver_path": "test",
    }


def test_earnings_imminent_date_missing_should_run():
    state = {
        "audit": {"research": {"web_queries_total": 0, "web_queries_by_ticker": {}}},
        "evidence_score": 0,
        "research_round": 0,
        "max_research_rounds": 2,
        "last_research_delta": 0,
    }
    reqs = [{
        "desk": "fundamental",
        "kind": "press_release_or_ir",
        "ticker": "AAPL",
        "query": "",  # missing required field on high-impact kind
        "priority": 2,
        "recency_days": 14,
        "max_items": 3,
        "rationale": "earnings <=14d and date/time unknown",
    }]
    d = should_run_web_research(state=state, evidence_requests=reqs, desk_outputs={}, user_request="")
    assert d["run"] is True
    assert "high_impact_missing_fields" in d["reason"]


def test_low_impact_low_uncertainty_false():
    state = {
        "audit": {"research": {"web_queries_total": 0, "web_queries_by_ticker": {}}},
        "evidence_score": 10,
        "research_round": 0,
        "max_research_rounds": 2,
        "last_research_delta": 0,
    }
    desk_outputs = {
        "macro": {"confidence": 0.8, "needs_more_data": False, "data_quality": {"missing_pct": 0.0}},
        "fundamental": {"confidence": 0.8, "needs_more_data": False, "data_quality": {"missing_pct": 0.0}},
        "sentiment": {"confidence": 0.8, "needs_more_data": False, "data_quality": {"missing_pct": 0.0}},
    }
    d = should_run_web_research(state=state, desk_outputs=desk_outputs, evidence_requests=[], user_request="")
    assert d["run"] is False


def test_disagreement_score_over_threshold_true():
    state = {
        "audit": {"research": {"web_queries_total": 0, "web_queries_by_ticker": {}}},
        "evidence_score": 0,
        "research_round": 0,
        "max_research_rounds": 2,
        "last_research_delta": 0,
    }
    reqs = [{
        "desk": "macro", "kind": "macro_headline_context", "ticker": "AAPL", "query": "driver",
        "priority": 3, "recency_days": 7, "max_items": 3, "rationale": "test",
    }]
    d = should_run_web_research(
        state=state,
        evidence_requests=reqs,
        disagreement_score=0.75,
        desk_outputs={},
        user_request="",
    )
    assert d["run"] is True
    assert "disagreement" in d["reason"]


def test_budget_cap_applies():
    reqs = [
        {"desk": "fundamental", "kind": "ownership_identity", "ticker": "AAPL", "priority": 1},
        {"desk": "fundamental", "kind": "ownership_identity", "ticker": "AAPL", "priority": 2},
        {"desk": "fundamental", "kind": "ownership_identity", "ticker": "AAPL", "priority": 3},
        {"desk": "fundamental", "kind": "ownership_identity", "ticker": "AAPL", "priority": 4},
    ]
    capped = cap_requests_by_budget(
        reqs,
        queries_used_total=0,
        queries_used_by_ticker={},
        max_web_queries_per_run=6,
        max_web_queries_per_ticker=3,
    )
    assert len(capped) == 3


def test_evidence_score_formula_fixed():
    # 4 buckets hit => coverage 40
    store = {
        "h1": _item("press_release_or_ir", days_old=0, trust_tier=1.0, title="beat"),
        "h2": _item("macro_headline_context", days_old=2, trust_tier=1.0, title="growth"),
        "h3": _item("ownership_identity", days_old=5, trust_tier=1.0, title="buy"),
        "h4": _item("valuation_context", days_old=10, trust_tier=1.0, title="expensive"),
    }
    out = compute_evidence_score(store, _now_iso())
    assert out["coverage"] == 40
    # freshness avg of [25,20,15,10] => 18
    assert out["freshness"] == 18
    assert out["source_trust"] == 25
    assert out["contradiction_penalty"] == 0
    assert out["score"] == 83


def test_evidence_score_break_threshold():
    state = {
        "audit": {"research": {"web_queries_total": 0, "web_queries_by_ticker": {}}},
        "evidence_score": 80,
        "research_round": 0,
        "max_research_rounds": 2,
        "last_research_delta": 0,
    }
    reqs = [{
        "desk": "macro", "kind": "macro_headline_context", "ticker": "AAPL", "query": "driver",
        "priority": 1, "recency_days": 7, "max_items": 3, "rationale": "test",
    }]
    d = should_run_web_research(state=state, evidence_requests=reqs, desk_outputs={}, user_request="")
    assert d["run"] is False
    assert d["reason"] == "evidence_score_enough"
