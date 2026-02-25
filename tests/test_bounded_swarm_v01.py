"""
tests/test_bounded_swarm_v01.py
===============================
Bounded Swarm v0.1 planner/rerun selector regression tests.
"""

from __future__ import annotations

from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import investment_team
from schemas.common import create_initial_state


def _base_router_state() -> dict:
    state = create_initial_state(user_request="AAPL 리서치 보강", mode="mock", seed=7)
    state["target_ticker"] = "AAPL"
    state["iteration_count"] = 1
    state["macro_analysis"] = {"open_questions": []}
    state["fundamental_analysis"] = {"open_questions": []}
    state["sentiment_analysis"] = {"open_questions": []}
    state["technical_analysis"] = {}
    state["evidence_requests"] = []
    state["evidence_store"] = {}
    state["research_round"] = 0
    state["audit"] = {"research": {"web_queries_total": 0, "web_queries_by_ticker": {}}}
    return state


def test_router_emits_swarm_metadata_and_request_enrichment():
    state = _base_router_state()
    out = investment_team.research_router_node(state)

    assert isinstance(out.get("_swarm_candidates"), list)
    assert isinstance(out.get("_swarm_plan"), list)
    assert len(out.get("_swarm_plan", [])) >= 4

    req = out["_swarm_plan"][0]
    assert req.get("source_tag")
    assert req.get("request_id")
    assert req.get("expected_bucket") in {"earnings", "macro", "ownership", "valuation", "other"}
    assert isinstance(req.get("impacted_desks"), list)


def test_router_seeds_only_missing_buckets():
    state = _base_router_state()
    now = datetime.now(timezone.utc).isoformat()
    state["evidence_store"] = {
        "m1": {
            "hash": "m1",
            "kind": "macro_headline_context",
            "ticker": "__GLOBAL__",
            "published_at": now,
            "retrieved_at": now,
            "title": "macro",
            "url": "https://www.federalreserve.gov/news",
            "source": "federalreserve.gov",
            "trust_tier": 1.0,
        }
    }

    out = investment_team.research_router_node(state)
    seeded_kinds = {
        str(req.get("kind", ""))
        for req in out.get("_swarm_plan", [])
        if str(req.get("source_tag", "")) == "seed"
    }

    assert "macro_headline_context" not in seeded_kinds
    assert "press_release_or_ir" in seeded_kinds
    assert "ownership_identity" in seeded_kinds
    assert "valuation_context" in seeded_kinds


def test_research_executor_rerun_selector_top_k_and_quant_excluded(monkeypatch):
    def fake_resolve(req, *, sec, web, as_of):
        kind = str(req.get("kind", "web_search"))
        rid = str(req.get("request_id", f"{kind}-x"))
        item = {
            "hash": f"h-{rid}",
            "url": "https://sec.gov/mock",
            "title": kind,
            "published_at": as_of,
            "source": "sec.gov",
            "retrieved_at": as_of,
            "snippet": "s",
            "kind": kind,
            "desk": req.get("desk", "fundamental"),
            "ticker": req.get("ticker", "AAPL"),
            "trust_tier": 1.0,
            "resolver_path": "fake",
        }
        return [item], "fake"

    monkeypatch.setattr(investment_team, "_resolve_request_with_priority", fake_resolve)

    now = datetime.now(timezone.utc).isoformat()
    state = {
        "run_id": "swarm-rerun-test",
        "iteration_count": 1,
        "mode": "mock",
        "as_of": now,
        "_research_plan": [
            {
                "desk": "fundamental",
                "kind": "press_release_or_ir",
                "ticker": "AAPL",
                "query": "earnings date",
                "priority": 1,
                "recency_days": 30,
                "max_items": 2,
                "rationale": "test1",
                "request_id": "r1",
            },
            {
                "desk": "fundamental",
                "kind": "ownership_identity",
                "ticker": "AAPL",
                "query": "ownership changes",
                "priority": 2,
                "recency_days": 90,
                "max_items": 2,
                "rationale": "test2",
                "request_id": "r2",
            },
        ],
        "completed_tasks": {"macro": True, "fundamental": True, "sentiment": True, "quant": True},
        "audit": {"research": {"web_queries_total": 0, "web_queries_by_ticker": {}}},
        "evidence_store": {},
        "research_round": 0,
        "macro_analysis": {},
        "fundamental_analysis": {},
        "sentiment_analysis": {},
        "technical_analysis": {},
    }
    out = investment_team.research_executor_node(state)

    assert out["completed_tasks"]["fundamental"] is False
    assert out["completed_tasks"]["sentiment"] is False
    assert out["completed_tasks"]["macro"] is True
    assert out["completed_tasks"]["quant"] is True

    assert out["_rerun_plan"]["selected_desks"] == ["fundamental", "sentiment"]
    assert len(out["_executed_requests"]) == 2
    assert out["_evidence_delta_kinds"]["press_release_or_ir"] == 1
    assert out["_evidence_delta_kinds"]["ownership_identity"] == 1
