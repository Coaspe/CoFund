"""
agents/fundamental_agent.py — ③ Fundamental Analyst Agent
==========================================================
CHANGELOG:
  v1.0 (2026-02-22) — 신규 생성. engine + mock LLM decision.

아키텍처: data_provider → fundamental_engine (Python compute) → LLM/mock → FundamentalOutput
Risk Manager Gate3에 직접 공급 가능: structural_risk_flag, risk_flags codes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from engines.fundamental_engine import compute_structural_risk
from schemas.common import make_evidence, make_risk_flag


def fundamental_analyst_run(
    ticker: str,
    financials: dict,
    sec_data: dict | None = None,
    *,
    run_id: str = "",
    as_of: str = "",
    horizon_days: int = 90,
    source_name: str = "mock",
) -> dict:
    """
    Fundamental Analyst 파이프라인: engine compute → mock decision → output dict.

    structural_risk_flag=True → recommendation=reject, confidence<=0.4
    """
    as_of = as_of or datetime.now(timezone.utc).isoformat()

    # ── Python 연산 (엔진) ────────────────────────────────────────────
    sr = compute_structural_risk(financials, sec_data)

    structural_flag = sr["structural_risk_flag"]
    hard = sr["hard_red_flags"]
    soft = sr["soft_flags"]
    altman = sr["altman"]
    coverage = sr["coverage"]
    fcf = sr["fcf_quality"]

    # ── Evidence 생성 ─────────────────────────────────────────────────
    evidence = []
    q = 0.3 if source_name == "mock" else 0.7

    for key in ["revenue_growth", "operating_margin", "roe", "debt_to_equity", "pe_ratio", "free_cash_flow"]:
        val = financials.get(key)
        if val is not None:
            evidence.append(make_evidence(metric=key, value=val, source_name=source_name, quality=q, as_of=as_of))

    if altman["z_score"] is not None:
        evidence.append(make_evidence(metric="altman_z_score", value=altman["z_score"], source_name="fundamental_engine", source_type="model", quality=0.9, as_of=as_of, note=f"bucket={altman['bucket']}"))
    if coverage["interest_coverage"] is not None:
        evidence.append(make_evidence(metric="interest_coverage", value=coverage["interest_coverage"], source_name="fundamental_engine", source_type="model", quality=0.9, as_of=as_of))
    if coverage["debt_to_ebitda"] is not None:
        evidence.append(make_evidence(metric="debt_to_ebitda", value=coverage["debt_to_ebitda"], source_name="fundamental_engine", source_type="model", quality=0.9, as_of=as_of))

    # ── Risk Flags (schemas.common 형식) ──────────────────────────────
    risk_flags = [make_risk_flag(**f) for f in hard + soft]

    # ── Mock LLM 결정 ─────────────────────────────────────────────────
    if structural_flag:
        primary_decision = "avoid"
        recommendation = "reject"
        confidence = 0.35
        notes = (
            f"Structural risk detected: {', '.join(f['code'] for f in hard)}. "
            "투자 부적합 — Gate3 reject_local 권고."
        )
    elif len(soft) > 0:
        primary_decision = "neutral"
        recommendation = "allow_with_limits"
        confidence = 0.50
        notes = f"Soft flags: {', '.join(f['code'] for f in soft)}. 비중 축소 권고."
    else:
        primary_decision = "bullish" if financials.get("revenue_growth", 0) > 5 else "neutral"
        recommendation = "allow"
        confidence = 0.65
        notes = "핵심 구조적 리스크 없음. 재무 건전성 양호."

    data_ok = any(e["metric"] in ("revenue_growth", "pe_ratio", "altman_z_score") for e in evidence if e.get("value") is not None)

    limitations = []
    if source_name == "mock":
        limitations.append("Mock 재무제표 — 실제 SEC filing과 차이 가능")
    if sec_data is None:
        limitations.append("SEC 텍스트 분석 미수행 — going concern/restatement 확인 불가")

    # Valuation block
    valuation = {
        "pe_ratio": financials.get("pe_ratio"),
        "revenue_growth_pct": financials.get("revenue_growth"),
        "roe_pct": financials.get("roe"),
        "debt_to_equity": financials.get("debt_to_equity"),
    }

    return {
        "agent_type": "fundamental",
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
        "summary": notes,
        "status": "ok",
        # Fundamental-specific
        "valuation_block": valuation,
        "quality_block": {"altman": altman, "coverage": coverage, "fcf_quality": fcf},
        "structural_risk_flag": structural_flag,
        "hard_red_flags": hard,
        "soft_flags": soft,
        "notes_for_risk_manager": notes,
        # Backward compat
        "sector": financials.get("sector", "Unknown"),
        "revenue_growth": financials.get("revenue_growth"),
        "roe": financials.get("roe"),
        "debt_to_equity": financials.get("debt_to_equity"),
        "pe_ratio": financials.get("pe_ratio"),
    }
