"""
eval/smoke_run.py — Smoke Test: 파이프라인 20회 실행 검증
=========================================================
CHANGELOG:
  v1.0 (2026-02-22) — 신규 생성.

검증 항목:
  - 무한루프 없음 (iteration <= MAX_ITERATIONS)
  - 필수 필드 존재 (run_id, final_report, risk_assessment)
  - 예외 미발생
  - events.jsonl 생성
"""

from __future__ import annotations

import sys
import os
import traceback
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from investment_team import main as run_pipeline


def smoke_test(n_runs: int = 20) -> None:
    """Mock 모드로 파이프라인을 n_runs회 실행하고 결과를 요약."""
    print(f"🔥 Smoke Test — {n_runs} runs")
    print("=" * 60)

    results = {"pass": 0, "fail": 0, "errors": []}

    for i in range(n_runs):
        seed = 42 + i
        try:
            final_state = run_pipeline(mode="mock", seed=seed)

            # 검증 항목
            checks = []

            # 1. run_id 존재
            run_id = final_state.get("run_id", "")
            checks.append(("run_id exists", bool(run_id)))

            # 2. final_report 비어있지 않음
            report = final_state.get("final_report", "")
            checks.append(("final_report non-empty", len(report) > 10))

            # 3. risk_assessment 존재
            risk = final_state.get("risk_assessment", {})
            checks.append(("risk_assessment exists", bool(risk)))

            # 4. iteration 한도 내
            iteration = final_state.get("iteration_count", 0)
            checks.append(("iteration <= MAX+1", iteration <= 4))

            # 5. events.jsonl 존재
            events_path = Path("runs") / run_id / "events.jsonl"
            checks.append(("events.jsonl exists", events_path.exists()))

            # 6. final_state.json 존재
            state_path = Path("runs") / run_id / "final_state.json"
            checks.append(("final_state.json exists", state_path.exists()))

            failed = [name for name, ok in checks if not ok]
            if failed:
                results["fail"] += 1
                results["errors"].append(f"Run {i} (seed={seed}): FAILED checks: {failed}")
                print(f"   ❌ Run {i:2d} (seed={seed}): FAIL — {failed}")
            else:
                results["pass"] += 1
                print(f"   ✅ Run {i:2d} (seed={seed}): PASS (iter={iteration})")

        except Exception as exc:
            results["fail"] += 1
            tb = traceback.format_exc()
            results["errors"].append(f"Run {i} (seed={seed}): EXCEPTION — {exc}")
            print(f"   💥 Run {i:2d} (seed={seed}): EXCEPTION — {exc}")
            # 첫 3개 에러만 상세 출력
            if results["fail"] <= 3:
                print(tb)

    # 요약
    print(f"\n{'=' * 60}")
    print(f"📊 Smoke Test 결과: {results['pass']}/{n_runs} PASS, {results['fail']}/{n_runs} FAIL")
    if results["errors"]:
        print(f"\n❌ 실패 항목:")
        for e in results["errors"][:10]:
            print(f"   - {e}")
    else:
        print("✅ 모든 실행 통과!")
    print("=" * 60)

    sys.exit(0 if results["fail"] == 0 else 1)


if __name__ == "__main__":
    smoke_test(20)
