"""
tests/test_risk_barrier.py — Risk Manager 1회 실행 검증
=======================================================
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from investment_team import main as run_pipeline


def test_risk_evaluates_once():
    """
    Risk Manager가 iteration당 정확히 1회만 '실제 평가'를 수행하는지 확인.
    events.jsonl에서 risk_manager exit 이벤트 수를 세서 검증.
    """
    final_state = run_pipeline(mode="mock", seed=42)
    run_id = final_state["run_id"]
    events_path = Path("runs") / run_id / "events.jsonl"

    assert events_path.exists(), "events.jsonl 미생성"

    risk_exits = []
    with open(events_path) as f:
        for line in f:
            event = json.loads(line)
            if event["node_name"] == "risk_manager" and event["phase"] == "exit":
                risk_exits.append(event)

    iterations = final_state.get("iteration_count", 0)

    # risk_manager exit 횟수는 iteration 수 이하여야 함
    assert len(risk_exits) <= iterations, (
        f"Risk Manager가 {len(risk_exits)}회 exit했으나 iteration은 {iterations}회. "
        "중복 실행 의심."
    )

    # 중복 iteration 체크
    evaluated_iters = [e["iteration"] for e in risk_exits if not e.get("outputs_summary", {}).get("skipped")]
    assert len(evaluated_iters) == len(set(evaluated_iters)), (
        f"동일 iteration에서 risk_manager 중복 실행: {evaluated_iters}"
    )


if __name__ == "__main__":
    test_risk_evaluates_once()
    print("✅ test_risk_evaluates_once PASSED")
