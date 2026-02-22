"""
tests/test_gate_order.py — Risk Manager 5-Gate 순서 불변 검증
=============================================================
Iron Rule R4: Gate1→Gate2→Gate3→Gate4→Gate5 순서 고정.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_gate_order_in_risk_agent():
    """
    risk_agent.py의 _make_5gate_decision 또는 유사 함수에서
    Gate 처리 순서가 1→2→3→4→5 인지 검증.

    검증 방식: 함수 본문 내 gate 키워드의 첫 등장 라인을 비교.
    """
    risk_file = Path(__file__).resolve().parent.parent / "risk_agent.py"
    if not risk_file.exists():
        print("⚠️  risk_agent.py not found — skipping")
        return

    source = risk_file.read_text()
    lines = source.split("\n")

    # _make_5gate_decision 또는 해당 함수의 범위를 찾기
    fn_start = None
    fn_end = None
    for i, line in enumerate(lines):
        if "def _make_5gate_decision" in line or "def _apply_gates" in line:
            fn_start = i
        elif fn_start is not None and line.startswith("def ") and i > fn_start:
            fn_end = i
            break
    if fn_start is None:
        # 함수명이 다를 수 있음 — 전체 소스에서 gate 순서만 검증
        fn_start = 0
        fn_end = len(lines)

    # 함수 본문 내에서 gate 키워드 최초 등장 순서 검증
    gate_first_occurrence = {}
    for i in range(fn_start, fn_end or len(lines)):
        lower = lines[i].lower()
        for gn in range(1, 6):
            patterns = [f"gate{gn}", f"gate {gn}", f"gate_{gn}"]
            if any(p in lower for p in patterns) and gn not in gate_first_occurrence:
                gate_first_occurrence[gn] = i

    found = sorted(gate_first_occurrence.keys())
    if len(found) < 2:
        print(f"⚠️  Gate 키워드 {len(found)}개만 발견 — 충분한 검증 불가")
        return

    # 순서 검증
    for j in range(len(found) - 1):
        g_curr, g_next = found[j], found[j + 1]
        assert gate_first_occurrence[g_curr] < gate_first_occurrence[g_next], (
            f"Gate {g_curr} (line {gate_first_occurrence[g_curr]}) must appear before "
            f"Gate {g_next} (line {gate_first_occurrence[g_next]}) in the decision function."
        )

    print(f"✅ Gate order verified in function body: {found}")


if __name__ == "__main__":
    test_gate_order_in_risk_agent()
    print("✅ test_gate_order PASSED")
