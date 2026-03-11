"""
tests/test_research_telemetry.py
================================
Ensures folded research telemetry is recorded on the parent node event.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import investment_team


def test_research_executor_exit_event_carries_research_round_summary(monkeypatch):
    def fake_resolve(req, **kwargs):
        kind = str(req.get("kind", "web_search"))
        rid = str(req.get("request_id", f"{kind}-x"))
        as_of = str(kwargs.get("as_of", ""))
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

    events: list[dict] = []

    def _capture(*args, **kwargs):
        events.append(
            {
                "node_name": kwargs.get("node_name"),
                "phase": kwargs.get("phase"),
                "outputs_summary": kwargs.get("outputs_summary"),
            }
        )

    monkeypatch.setattr(investment_team.telemetry, "log_event", _capture)

    now = datetime.now(timezone.utc).isoformat()
    state = {
        "run_id": "folded-research-telemetry",
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
            }
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
    assert out["research_round"] == 1

    exit_events = [
        event for event in events
        if event.get("node_name") == "research_executor" and event.get("phase") == "exit"
    ]
    assert exit_events

    summary = exit_events[-1]["outputs_summary"] or {}
    research_round = summary.get("research_round", {})
    rerun = summary.get("rerun", {})

    assert summary.get("queries_executed") == 1
    assert research_round.get("round") == 1
    assert research_round.get("queries_executed") == 1
    assert "last_research_delta" in research_round
    assert "evidence_score" in research_round
    assert "fundamental" in (rerun.get("selected_desks") or [])
