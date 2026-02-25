"""
tests/test_research_loop_breaks.py
==================================
Research loop break conditions + last_research_delta definition tests.
"""

from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import investment_team
from engines.research_policy import should_run_web_research


def test_break_after_max_research_rounds():
    state = {
        "audit": {"research": {"web_queries_total": 0, "web_queries_by_ticker": {}}},
        "evidence_score": 10,
        "research_round": 2,
        "max_research_rounds": 2,
        "last_research_delta": 3,
    }
    reqs = [{
        "desk": "macro", "kind": "macro_headline_context", "ticker": "AAPL", "query": "driver",
        "priority": 1, "recency_days": 7, "max_items": 3, "rationale": "test",
    }]
    out = should_run_web_research(state=state, evidence_requests=reqs, desk_outputs={})
    assert out["run"] is False
    assert out["reason"] == "max_research_rounds"


def test_break_when_added_delta_below_2():
    state = {
        "audit": {"research": {"web_queries_total": 0, "web_queries_by_ticker": {}}},
        "evidence_score": 30,
        "research_round": 1,
        "max_research_rounds": 2,
        "last_research_delta": 1,
    }
    reqs = [{
        "desk": "macro", "kind": "macro_headline_context", "ticker": "AAPL", "query": "driver",
        "priority": 1, "recency_days": 7, "max_items": 3, "rationale": "test",
    }]
    out = should_run_web_research(state=state, evidence_requests=reqs, desk_outputs={})
    assert out["run"] is False
    assert out["reason"] == "low_added_evidence_delta"


def test_last_research_delta_equals_new_unique_hash_count(monkeypatch):
    def fake_resolve(req, *, sec, web, as_of):
        items = [
            {
                "hash": "h1", "url": "https://sec.gov/a", "title": "t1", "published_at": as_of, "source": "sec.gov",
                "retrieved_at": as_of, "snippet": "s", "kind": req["kind"], "desk": req["desk"],
                "ticker": req["ticker"], "trust_tier": 1.0, "resolver_path": "fake",
            },
            {
                "hash": "h1", "url": "https://sec.gov/a2", "title": "t1", "published_at": as_of, "source": "sec.gov",
                "retrieved_at": as_of, "snippet": "s", "kind": req["kind"], "desk": req["desk"],
                "ticker": req["ticker"], "trust_tier": 1.0, "resolver_path": "fake",
            },
            {
                "hash": "h2", "url": "https://sec.gov/b", "title": "t2", "published_at": as_of, "source": "sec.gov",
                "retrieved_at": as_of, "snippet": "s", "kind": req["kind"], "desk": req["desk"],
                "ticker": req["ticker"], "trust_tier": 1.0, "resolver_path": "fake",
            },
        ]
        return items, "fake"

    monkeypatch.setattr(investment_team, "_resolve_request_with_priority", fake_resolve)

    now = datetime.now(timezone.utc).isoformat()
    state = {
        "run_id": "test-run",
        "iteration_count": 1,
        "mode": "mock",
        "as_of": now,
        "_research_plan": [{
            "desk": "fundamental", "kind": "ownership_identity", "ticker": "AAPL", "query": "q",
            "priority": 1, "recency_days": 30, "max_items": 5, "rationale": "test",
        }],
        "evidence_store": {
            "h0": {
                "hash": "h0", "url": "https://sec.gov/c", "title": "old", "published_at": now, "source": "sec.gov",
                "retrieved_at": now, "snippet": "", "kind": "ownership_identity", "desk": "fundamental",
                "ticker": "AAPL", "trust_tier": 1.0, "resolver_path": "old",
            }
        },
        "research_round": 0,
        "completed_tasks": {"macro": True, "fundamental": True, "sentiment": True, "quant": True},
        "audit": {"research": {"web_queries_total": 0, "web_queries_by_ticker": {}}},
    }
    out = investment_team.research_executor_node(state)
    assert out["last_research_delta"] == 2
    assert out["research_round"] == 1
