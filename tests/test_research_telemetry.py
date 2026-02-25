"""
tests/test_research_telemetry.py
================================
Ensures research_round event is recorded in events.jsonl.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from investment_team import main as run_pipeline


def test_research_round_event_logged():
    final = run_pipeline(mode="mock", seed=42)
    run_id = final["run_id"]
    events_path = Path("runs") / run_id / "events.jsonl"
    assert events_path.exists()

    found = False
    with open(events_path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e.get("node_name") == "research_round":
                found = True
                out = e.get("outputs_summary", {}) or {}
                assert "research_round" in out
                assert "queries_executed" in out
                assert "last_research_delta" in out
                assert "evidence_score" in out
                break
    assert found, "research_round event not found"
