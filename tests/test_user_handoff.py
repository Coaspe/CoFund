"""
tests/test_user_handoff.py
==========================
User handoff behavior when autonomous loop cannot resolve blocking issues.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import investment_team
from schemas.common import create_initial_state


def _base_state() -> dict:
    state = create_initial_state(user_request="NVDA 과열 점검", mode="mock", seed=42)
    state["target_ticker"] = "NVDA"
    state["iteration_count"] = 1
    state["macro_analysis"] = {"open_questions": []}
    state["fundamental_analysis"] = {"open_questions": []}
    state["sentiment_analysis"] = {"open_questions": []}
    state["technical_analysis"] = {}
    state["evidence_store"] = {}
    state["evidence_requests"] = []
    state["audit"] = {"research": {"web_queries_total": 0, "web_queries_by_ticker": {}}}
    return state


def test_handoff_required_on_hard_stop_with_blocking_issue(monkeypatch):
    state = _base_state()
    state["research_round"] = 2
    state["max_research_rounds"] = 2

    monkeypatch.setattr(
        investment_team,
        "plan_runtime_recovery",
        lambda _s, _d: {
            "issues": [{"code": "newsapi_upgrade_required", "desk": "sentiment", "detail": "426 Upgrade Required"}],
            "actions": [],
            "evidence_requests": [],
            "notes": [],
        },
    )

    events = []

    def _capture(*args, **kwargs):
        node_name = kwargs.get("node_name")
        if node_name is None and len(args) >= 2:
            node_name = args[1]
        events.append(node_name)

    monkeypatch.setattr(investment_team.telemetry, "log_event", _capture)

    out = investment_team.research_router_node(state)
    assert out["_run_research"] is False
    assert out["research_stop_reason"] == "max_research_rounds"
    assert out["user_action_required"] is True
    assert out["user_action_items"][0]["code"] == "newsapi_upgrade_required"
    assert "human_handoff" in events


def test_no_handoff_while_research_still_running(monkeypatch):
    state = _base_state()
    state["research_round"] = 0
    state["max_research_rounds"] = 2
    state["evidence_requests"] = [
        {
            "desk": "macro",
            "kind": "macro_headline_context",
            "ticker": "NVDA",
            "query": "macro driver",
            "priority": 1,
            "recency_days": 7,
            "max_items": 3,
            "rationale": "test",
        }
    ]
    state["macro_analysis"] = {"needs_more_data": True, "confidence": 0.4, "data_quality": {"missing_pct": 0.4}, "open_questions": []}
    state["fundamental_analysis"] = {"needs_more_data": True, "confidence": 0.4, "data_quality": {"missing_pct": 0.4}, "open_questions": []}
    state["sentiment_analysis"] = {"needs_more_data": True, "confidence": 0.4, "data_quality": {"missing_pct": 0.4}, "open_questions": []}

    monkeypatch.setattr(
        investment_team,
        "plan_runtime_recovery",
        lambda _s, _d: {
            "issues": [{"code": "fmp_endpoint_restricted", "desk": "fundamental", "detail": "403"}],
            "actions": [],
            "evidence_requests": [],
            "notes": [],
        },
    )

    out = investment_team.research_router_node(state)
    assert out["_run_research"] is True
    assert out["user_action_required"] is False
    assert out["user_action_items"] == []
