"""
tests/test_graph_finishes.py — 그래프가 항상 END에 도달하는지 검증
==================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from investment_team import main as run_pipeline


def test_graph_finishes_mock():
    """Mock 모드에서 파이프라인이 정상 종료되는지 확인."""
    final_state = run_pipeline(mode="mock", seed=42)
    assert final_state is not None
    assert "final_report" in final_state
    assert len(final_state["final_report"]) > 0
    assert final_state.get("iteration_count", 0) >= 1


def test_graph_finishes_multiple_seeds():
    """다양한 seed로 정상 종료 확인."""
    for seed in [1, 7, 42, 99, 123]:
        final_state = run_pipeline(mode="mock", seed=seed)
        assert final_state is not None
        assert "final_report" in final_state


if __name__ == "__main__":
    test_graph_finishes_mock()
    print("✅ test_graph_finishes_mock PASSED")
    test_graph_finishes_multiple_seeds()
    print("✅ test_graph_finishes_multiple_seeds PASSED")
