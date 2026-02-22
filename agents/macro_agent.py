"""
agents/macro_agent.py — ② Macro Analyst Agent
===============================================
CHANGELOG:
  v1.0 (2026-02-22) — 신규 생성. engine + mock LLM decision.

아키텍처: data_provider → macro_engine (Python compute) → LLM/mock (해석만) → MacroOutput
Iron Rule R1: LLM은 수치 연산 금지. 엔진이 산출한 bucket/flag만 해석.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any

from engines.macro_engine import compute_macro_features, compute_overlay_guidance
from schemas.common import make_evidence, make_risk_flag
from schemas.taxonomy import map_macro_regime_to_canonical


def macro_analyst_run(
    ticker: str,
    macro_indicators: dict,
    *,
    run_id: str = "",
    as_of: str = "",
    horizon_days: int = 30,
    source_name: str = "mock",
) -> dict:
    """
    Macro Analyst 전체 파이프라인: engine compute → mock LLM decision → output dict.

    Args:
        ticker: 분석 대상 종목
        macro_indicators: macro_engine에 전달할 지표 딕트
        run_id: 실행 ID
        as_of: 기준 시각
        horizon_days: 분석 기간
        source_name: 데이터 출처명

    Returns:
        MacroOutput 호환 dict
    """
    as_of = as_of or datetime.now(timezone.utc).isoformat()

    # ── Python 연산 (엔진) ────────────────────────────────────────────
    features = compute_macro_features(macro_indicators)
    overlay = compute_overlay_guidance(features)

    regime = features["macro_regime"]
    tail_risk = features["tail_risk_warning"]

    # ── Evidence 생성 ─────────────────────────────────────────────────
    evidence = []
    q = 0.3 if source_name == "mock" else 0.7
    for key in ["yield_curve_spread", "hy_oas", "inflation_expectation", "cpi_yoy", "pmi", "fed_funds_rate", "gdp_growth"]:
        val = macro_indicators.get(key)
        if val is not None:
            evidence.append(make_evidence(metric=key, value=val, source_name=source_name, quality=q, as_of=as_of))

    for bucket_key in ["curve_state", "credit_stress_level", "inflation_state", "growth_state"]:
        evidence.append(make_evidence(
            metric=bucket_key, value=features[bucket_key],
            source_name="macro_engine", source_type="model", quality=0.9, as_of=as_of,
        ))

    # ── 리스크 플래그 ─────────────────────────────────────────────────
    risk_flags = []
    if tail_risk:
        risk_flags.append(make_risk_flag("macro_tail_risk", "high", "Credit stress or inverted curve + stressed credit"))
    if regime in ("contraction", "stagflation"):
        risk_flags.append(make_risk_flag("macro_headwind", "high", f"Macro regime: {regime}"))

    # ── Mock LLM 결정 (Python 연산값만 해석) ──────────────────────────
    if regime in ("contraction",):
        primary_decision = "bearish"
        recommendation = "reject"
        confidence = 0.75
    elif regime in ("stagflation",):
        primary_decision = "bearish"
        recommendation = "allow_with_limits"
        confidence = 0.55
    elif regime in ("late_cycle",):
        primary_decision = "neutral"
        recommendation = "allow_with_limits"
        confidence = 0.50
    elif regime in ("goldilocks", "expansion"):
        primary_decision = "bullish"
        recommendation = "allow"
        confidence = 0.65
    else:
        primary_decision = "neutral"
        recommendation = "allow_with_limits"
        confidence = 0.55

    # data_ok = evidence 중 주요 지표가 있는지
    has_key_data = any(e["metric"] in ("yield_curve_spread", "hy_oas", "gdp_growth") for e in evidence if e.get("value") is not None)
    data_ok = has_key_data

    limitations = []
    if source_name == "mock":
        limitations.append("Mock 데이터 사용 — 실제 매크로 지표와 차이 가능")
    if not has_key_data:
        limitations.append("핵심 매크로 지표 부재 — 레짐 판단 신뢰도 낮음")

    return {
        "agent_type": "macro",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of": as_of,
        "ticker": ticker,
        "horizon_days": horizon_days,
        "primary_decision": primary_decision,
        "recommendation": recommendation,
        "confidence": confidence if data_ok else min(confidence, 0.4),
        "risk_flags": risk_flags,
        "evidence": evidence,
        "limitations": limitations,
        "data_ok": data_ok,
        "summary": f"Macro regime: {regime}. {overlay.get('equity_overlay_guidance', '')}",
        "status": "ok",
        # Macro-specific: canonical taxonomy
        "macro_regime_raw": regime,                              # original engine value
        "macro_regime": map_macro_regime_to_canonical(regime),   # canonical (risk reads this)
        "overlay_guidance": overlay,
        "tail_risk_warning": tail_risk,
        "indicators": features,
        # Backward compat
        "regime": map_macro_regime_to_canonical(regime),
        "gdp_growth": macro_indicators.get("gdp_growth"),
        "interest_rate": macro_indicators.get("fed_funds_rate"),
    }
