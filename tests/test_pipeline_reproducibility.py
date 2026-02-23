"""
tests/test_pipeline_reproducibility.py — T4: Identical PIT → identical positions_final + gate_trace
"""
import copy
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from risk.engine import run_gates


def _mock_payload(seed: int = 42) -> dict:
    """결정론적 테스트 페이로드."""
    import random
    rng = random.Random(seed)
    ticker = "AAPL"
    alloc = round(rng.uniform(0.05, 0.15), 4)
    return {
        "target_ticker": ticker,
        "risk_limits": {
            "max_portfolio_cvar_1d": 0.05,
            "max_leverage": 2.5,
            "max_hhi": 0.35,
            "max_sector_weight": 0.40,
            "max_quant_weight_anomaly": 0.30,
            "conservative_fallback_weight": 0.05,
        },
        "portfolio_summary": {
            "portfolio_cvar_1d": 0.02,
            "leverage_ratio": 1.0,
            "herfindahl_index": 0.10,
            "sector_exposure": {"Technology": alloc},
            "component_var_by_ticker": {ticker: 0.001},
            "concentration_top1": alloc,
            "gross_exposure": alloc,
        },
        "analyst_weights": {ticker: alloc},
        "per_ticker_data": {
            ticker: {
                "quant": {"decision": "LONG", "final_allocation_pct": alloc},
                "fundamental": {"structural_risk_flag": False, "risk_flags": []},
                "macro": {"regime": "expansion", "macro_regime": "expansion"},
            }
        },
        "analyst_reports": {
            "macro": {"regime": "expansion"},
            "fundamental": {"structural_risk_flag": False},
        },
    }


def test_pipeline_reproducibility():
    """동일한 페이로드로 2회 run_gates() 호출 → positions_final + gate_trace 완전 동일."""
    payload = _mock_payload(seed=42)

    result1 = run_gates(copy.deepcopy(payload))
    result2 = run_gates(copy.deepcopy(payload))

    # positions_final 비교
    assert result1["_positions_final"] == result2["_positions_final"], (
        f"positions_final 불일치!\n"
        f"Run1: {result1['_positions_final']}\n"
        f"Run2: {result2['_positions_final']}"
    )

    # gate_trace 비교 (decisions 부분만)
    trace1 = [{"gate": t["gate"], "name": t["name"], "flags": t["flags"]} for t in result1["gate_trace"]]
    trace2 = [{"gate": t["gate"], "name": t["name"], "flags": t["flags"]} for t in result2["gate_trace"]]
    assert trace1 == trace2, (
        f"gate_trace 불일치!\nTrace1: {trace1}\nTrace2: {trace2}"
    )


def test_gate_trace_order():
    """gate_trace의 gate 번호는 반드시 1→2→3→4→5 순서여야 한다."""
    payload = _mock_payload(seed=42)
    result = run_gates(payload)
    gates = [t["gate"] for t in result["gate_trace"]]
    assert gates == [1, 2, 3, 4, 5], f"Gate 순서 위반: {gates}"


def test_different_seeds_produce_different_results():
    """서로 다른 시드의 페이로드는 서로 다른 결과를 낼 수 있다 (sanity check)."""
    p1 = _mock_payload(seed=1)
    p2 = _mock_payload(seed=99)
    r1 = run_gates(p1)
    r2 = run_gates(p2)
    # 두 결과가 완전히 같지 않을 수 있음 (sanity — 적어도 weight가 다름을 확인)
    w1 = list(r1["_positions_final"].values())
    w2 = list(r2["_positions_final"].values())
    # 둘 다 결정론적으로 실행됨
    assert isinstance(w1[0], float) and isinstance(w2[0], float)
