"""
tests/test_graph_end_to_end.py — E2E 파이프라인 검증
=====================================================
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import telemetry
import investment_team
from investment_team import main as run_pipeline


def test_pipeline_completes():
    """파이프라인이 정상 종료되고 모든 필수 필드가 존재."""
    final = run_pipeline(mode="mock", seed=42)

    assert final is not None
    assert final.get("run_id")
    assert final.get("iteration_count", 0) >= 1

    # 모든 데스크가 완료 상태
    ct = final.get("completed_tasks", {})
    for desk in ["macro", "fundamental", "sentiment", "quant"]:
        assert ct.get(desk, False), f"{desk} not completed"

    # 에이전트 출력 존재
    assert final.get("macro_analysis", {}).get("macro_regime")
    assert "structural_risk_flag" in final.get("fundamental_analysis", {})
    assert final.get("sentiment_analysis", {}).get("tilt_factor") is not None
    assert final.get("technical_analysis", {}).get("decision")

    # Risk + Report
    assert final.get("risk_assessment", {}).get("grade")
    assert len(final.get("final_report", "")) > 10


def test_evidence_present():
    """모든 데스크 출력에 evidence가 존재 (R2)."""
    final = run_pipeline(mode="mock", seed=77)

    for key in ["macro_analysis", "fundamental_analysis", "sentiment_analysis", "technical_analysis"]:
        output = final.get(key, {})
        ev = output.get("evidence", [])
        assert len(ev) > 0, f"{key} has no evidence — violates R2"


def test_events_jsonl_created():
    """events.jsonl 생성 확인."""
    final = run_pipeline(mode="mock", seed=99)
    run_id = final["run_id"]
    events_path = Path("runs") / run_id / "events.jsonl"
    assert events_path.exists()

    nodes_logged = set()
    with open(events_path) as f:
        for line in f:
            e = json.loads(line)
            nodes_logged.add(e["node_name"])

    # 최소 orchestrator, risk_manager, report_writer가 로깅됨
    for expected in ["orchestrator", "risk_manager", "report_writer"]:
        assert expected in nodes_logged, f"{expected} not in events.jsonl"


def test_desk_events_record_graph_node_and_agent_id():
    """_log should emit graph node names, agent_id, and owner_agent_id consistently."""
    run_id = "identity-log-test"
    run_dir = Path("runs") / run_id
    if run_dir.exists():
        for path in run_dir.iterdir():
            path.unlink()
        run_dir.rmdir()

    telemetry.init_run(run_id, mode="mock")
    state = {
        "run_id": run_id,
        "iteration_count": 1,
        "target_ticker": "NVDA",
    }

    investment_team._log(state, "macro_analyst", "enter")
    investment_team._log(state, "research_router", "enter")

    events_path = run_dir / "events.jsonl"
    assert events_path.exists()

    with open(events_path, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]

    assert events[0]["node_name"] == "macro_analyst"
    assert events[0]["agent_id"] == "macro"
    assert events[0]["owner_agent_id"] == "macro"
    assert events[1]["node_name"] == "research_router"
    assert events[1]["agent_id"] == "research_manager"
    assert events[1]["owner_agent_id"] == "research_manager"


def test_no_broker_code():
    """R0: 소스코드에 broker/order 관련 코드가 없음."""
    root = Path(__file__).resolve().parent.parent
    banned_patterns = ["place_" + "order", "submit_" + "order", "broker.exe" + "cute", "execute_" + "trade"]
    for py_file in root.rglob("*.py"):
        if any(skip in str(py_file) for skip in (".venv", "runs", "tests", "__pycache__")):
            continue
        content = py_file.read_text()
        for pattern in banned_patterns:
            assert pattern not in content, f"R0 violation: '{pattern}' found in {py_file.name}"


if __name__ == "__main__":
    test_pipeline_completes()
    print("✅ test_pipeline_completes PASSED")
    test_evidence_present()
    print("✅ test_evidence_present PASSED")
    test_events_jsonl_created()
    print("✅ test_events_jsonl_created PASSED")
    test_no_broker_code()
    print("✅ test_no_broker_code PASSED")
