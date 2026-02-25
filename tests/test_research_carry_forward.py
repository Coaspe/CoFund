"""
tests/test_research_carry_forward.py
====================================
Carry-forward guarantees for research loop.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import investment_team


def test_research_executor_marks_only_rerun_desks():
    state = {
        "run_id": "carry-forward-test",
        "iteration_count": 1,
        "mode": "mock",
        "as_of": "2026-01-01T00:00:00+00:00",
        "_research_plan": [{
            "desk": "fundamental", "kind": "ownership_identity", "ticker": "AAPL", "query": "q",
            "priority": 1, "recency_days": 30, "max_items": 2, "rationale": "test",
        }],
        "completed_tasks": {"macro": True, "fundamental": True, "sentiment": True, "quant": True},
        "audit": {"research": {"web_queries_total": 0, "web_queries_by_ticker": {}}},
        "evidence_store": {},
        "research_round": 0,
    }
    out = investment_team.research_executor_node(state)
    assert out["completed_tasks"]["fundamental"] is False
    assert out["completed_tasks"]["macro"] is True
    assert out["completed_tasks"]["sentiment"] is True
    assert out["completed_tasks"]["quant"] is True


def test_non_rerun_desk_outputs_not_overwritten():
    # desk completed => research desk node returns {} (carry-forward)
    state = {"completed_tasks": {"macro": True}, "macro_analysis": {"summary": "old"}}
    out = investment_team.macro_analyst_research_node(state)
    assert out == {}


def test_evidence_requests_reducer_append_dedupe():
    from schemas.common import _merge_evidence_requests

    a = [{"desk": "macro", "kind": "macro_headline_context", "ticker": "AAPL", "query": "x"}]
    b = [
        {"desk": "macro", "kind": "macro_headline_context", "ticker": "AAPL", "query": "x"},
        {"desk": "fundamental", "kind": "ownership_identity", "ticker": "AAPL", "query": "y"},
    ]
    out = _merge_evidence_requests(a, b)
    assert len(out) == 2
