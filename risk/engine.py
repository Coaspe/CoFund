"""
risk/engine.py — Risk Gate Engine (Invariant B: fixed 1→2→3→4→5 order)
=======================================================================
GATES 배열 순서는 절대 변경 불가. 각 Gate는 이전 Gate의 positions_out을 입력으로 받는다.
"""
from __future__ import annotations

from typing import Any, Optional

from risk.gates.gate1_hard_limits import Gate1HardLimits
from risk.gates.gate2_concentration import Gate2Concentration
from risk.gates.gate3_structural import Gate3Structural
from risk.gates.gate4_regime_fit import Gate4RegimeFit
from risk.gates.gate5_model_anomaly import Gate5ModelAnomaly

# 절대 변경 불가 — Invariant B
GATES = [
    Gate1HardLimits,
    Gate2Concentration,
    Gate3Structural,
    Gate4RegimeFit,
    Gate5ModelAnomaly,
]

DEFAULT_LIMITS = {
    "max_portfolio_cvar_1d": 0.05,
    "max_leverage": 2.5,
    "max_hhi": 0.35,
    "max_sector_weight": 0.40,
    "max_quant_weight_anomaly": 0.30,
    "conservative_fallback_weight": 0.05,
}


def run_gates(
    payload: dict,
    positions_in: Optional[dict[str, float]] = None,
    lim: Optional[dict] = None,
) -> dict:
    """
    5-Gate 리스크 엔진을 순서대로 실행합니다.

    Args:
        payload: aggregate_risk_payload() 결과
        positions_in: {ticker -> proposed_weight} (없으면 payload에서 추출)
        lim: 리스크 한도 dict (없으면 기본값)

    Returns:
        {
            per_ticker_decisions: {ticker -> {final_weight, decision, flags, rationale_short}},
            portfolio_actions: {hedge_recommendations, gross_net_adjustment},
            orchestrator_feedback: {required, reasons, detail},
            gate_trace: [gate1_entry, gate2_entry, ..., gate5_entry],
            _positions_final: {ticker -> final_weight},
        }
    """
    lim = {**DEFAULT_LIMITS, **(lim or {}), **(payload.get("risk_limits", {}))}

    # positions_in 결정: 우선순위: 인자 > payload.analyst_weights > 균등비중
    if positions_in is None:
        positions_in = dict(payload.get("analyst_weights", {}))
    if not positions_in:
        tickers = list(payload.get("per_ticker_data", {}).keys())
        positions_in = {t: round(1.0 / len(tickers), 6) for t in tickers} if tickers else {}

    # Accumulate gate results
    positions = dict(positions_in)
    all_flags: list[str] = []
    feedback_required = False
    feedback_reasons: list[str] = []
    feedback_detail_parts: list[str] = []
    gate_trace: list[dict] = []

    # 순서 강제 — Invariant B (절대 변경 불가)
    for GateCls in GATES:
        positions, gate_flags, gate_feedback, trace_entry = GateCls.apply(payload, positions, lim)
        all_flags.extend(gate_flags)
        gate_trace.append(trace_entry)
        if gate_feedback["required"]:
            feedback_required = True
            feedback_reasons.extend(gate_feedback["reasons"])
            if gate_feedback["detail"]:
                feedback_detail_parts.append(gate_feedback["detail"])

    # 최종 결정 조립
    proposed = dict(positions_in)
    per_ticker_decisions: dict[str, dict] = {}
    for t, final_w in positions.items():
        proposed_w = proposed.get(t, 0.0)
        if final_w == 0.0 and proposed_w != 0.0:
            decision = "reject_local"
        elif abs(final_w) < abs(proposed_w) - 1e-8:
            decision = "reduce"
        else:
            decision = "approve"

        ticker_flags = [f for f in all_flags if t in f]
        per_ticker_decisions[t] = {
            "final_weight": final_w,
            "decision": decision,
            "flags": ticker_flags,
            "rationale_short": _rationale(decision, ticker_flags),
        }

    return {
        "per_ticker_decisions": per_ticker_decisions,
        "portfolio_actions": {
            "hedge_recommendations": [],
            "gross_net_adjustment": None,
        },
        "orchestrator_feedback": {
            "required": feedback_required,
            "reasons": list(dict.fromkeys(feedback_reasons)),
            "detail": " ".join(feedback_detail_parts) if feedback_detail_parts else "모든 Gate 통과.",
        },
        "gate_trace": gate_trace,
        "_positions_final": {t: d["final_weight"] for t, d in per_ticker_decisions.items()},
    }


def _rationale(decision: str, flags: list[str]) -> str:
    if decision == "approve":
        return "전 Gate 통과. 원안 승인."
    if decision == "reject_local":
        return f"구조적 리스크 감지 ({', '.join(flags[:2])}) → 비중 0."
    return f"리스크 한도 초과 ({', '.join(flags[:2])}) → 비중 축소."
