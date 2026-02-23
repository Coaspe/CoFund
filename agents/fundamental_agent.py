"""
agents/fundamental_agent.py — ③ Fundamental Analyst Agent (v2)
===============================================================
v2: factor_scores (5축) + valuation_stretch + key_drivers + what_to_watch + scenario_notes.
LLM 없음. Python-only. deterministic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from engines.fundamental_engine import (
    compute_structural_risk,
    compute_factor_scores,
    compute_valuation_stretch,
)
from schemas.common import make_evidence, make_risk_flag


# ── Template helpers ──────────────────────────────────────────────────────────

def _build_key_drivers(sr: dict, vs: dict, fs: dict) -> list[str]:
    drivers = []
    altman  = sr.get("altman", {})
    cov     = sr.get("coverage", {})
    fcf     = sr.get("fcf_quality", {})

    stretch_level = vs.get("stretch_level", "low")
    if stretch_level == "high":
        drivers.append(f"Valuation stretch: HIGH — {vs.get('rationale', '')[:60]}")
    elif stretch_level == "medium":
        drivers.append(f"Valuation stretch: MEDIUM — {vs.get('rationale', '')[:60]}")

    for code in (sr.get("hard_red_flags") or []):
        drivers.append(f"Hard flag: {code.get('code', code)} (critical)")

    if altman.get("z_score") is not None:
        drivers.append(f"Altman Z={altman['z_score']:.2f} ({altman['bucket']})")
    if cov.get("interest_coverage") is not None:
        drivers.append(f"ICR={cov['interest_coverage']:.1f} ({cov['icr_bucket']})")
    if fcf.get("fcf_margin_pct") is not None:
        drivers.append(f"FCF margin={fcf['fcf_margin_pct']:.1f}%")

    # Factor scores
    if fs.get("quality_score", 0.5) > 0.75:
        drivers.append("Quality: HIGH (ROE/margin 우수)")
    elif fs.get("quality_score", 0.5) < 0.35:
        drivers.append("Quality: LOW (수익성 부실)")
    if fs.get("growth_score", 0.5) > 0.75:
        drivers.append("Growth: HIGH (매출 가속)")
    if fs.get("cashflow_score", 0.5) < 0.30:
        drivers.append("Cashflow: WEAK (FCF 음수 또는 적자)")

    return drivers[:6]


def _build_what_to_watch(sr: dict, vs: dict, financials: dict) -> list[str]:
    items = []
    if vs.get("stretch_level") == "high":
        items.append("실적 미스/가이던스 하향 시 밸류에이션 리레이팅 위험 → 즉시 재검토")
    if vs.get("stretch_level") == "unknown":
        items.append("Valuation data 부족 — PE/PS 데이터 확보 후 stretch 재평가 필요")
    if sr.get("fcf_quality", {}).get("fcf_quality_flag"):
        items.append("FCF 2분기 연속 음수 시 구조적 리스크 상향 가능")
    for soft in sr.get("soft_flags") or []:
        code = soft.get("code", "")
        if "leverage" in code:
            items.append("부채비율 지속 상승 시 금리 충격 취약 → 이자보상배율 분기별 확인")
        if "coverage" in code:
            items.append("ICR 1.5 하회 시 채무불이행 위험 모니터링")
    if financials.get("revenue_growth", 0) > 20:
        items.append("성장주 특례: 성장 둔화(QoQ -5%p↓) 시 밸류에이션 급격 재평가 가능")
    items.append("다음 실적 발표 후 매출 성장률 및 마진 트렌드 재확인")
    return items[:5]


def _build_scenario_notes(sr: dict, vs: dict, fs: dict, financials: dict) -> dict:
    structural = sr.get("structural_risk_flag", False)
    stretch    = vs.get("stretch_level", "low")
    quality    = fs.get("quality_score", 0.5)
    growth     = fs.get("growth_score", 0.5)
    rev_growth = financials.get("revenue_growth", 0) or 0

    if structural:
        return {
            "bull": "구조적 리스크 해소(SEC 클리어, 재무 개선) 시에만 진입 재고.",
            "base": "현재 투자 부적합. 관망 유지.",
            "bear": "리스크 현실화 시 급락 가능. 포지션 보유 금지.",
        }
    if stretch == "high" and quality < 0.5:
        return {
            "bull": "실적 서프라이즈 + 금리 인하 시에만 밸류 정당화 가능.",
            "base": "밸류에이션 부담 + 퀄리티 미흡. 신규 매수 자제, 기존 비중 축소.",
            "bear": "실적 하회 시 대폭 조정 가능. 목표가 대폭 하향 시나리오 대비.",
        }
    if quality > 0.70 and growth > 0.60:
        return {
            "bull": "퀄리티 + 성장 조합 → 실적 주도 재평가. 중기 분할 매수 유효.",
            "base": "현 수준 유지. 분기 실적 확인 후 비중 조정 검토.",
            "bear": "시장 전반 리스크오프 시 베타 하락. 섹터 ETF 헤지 고려.",
        }
    if rev_growth > 20:
        return {
            "bull": "성장 가속 + 시장 기대 초과 시 밸류에이션 프리미엄 유지.",
            "base": "성장주 특례: 중기 보유. 단지 성장 둔화 트리거 모니터링.",
            "bear": "성장 실망 시 고PER 주 급락. 분할 진입 + 손절선 설정 필수.",
        }
    return {
        "bull": "실적 개선 + 섹터 로테이션 수혜 시 상승 여력.",
        "base": "현 수준 중립. 실적 확인 후 재판단.",
        "bear": "매크로 충격 시 밸류 하방 압박. 방어적 비중 유지.",
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def fundamental_analyst_run(
    ticker: str,
    financials: dict,
    sec_data: Optional[dict] = None,
    *,
    run_id: str = "",
    as_of: str = "",
    horizon_days: int = 90,
    source_name: str = "mock",
    history: Optional[dict] = None,
    peers: Optional[list] = None,
) -> dict:
    """
    Fundamental Analyst v2: structural risk + factor_scores + valuation_stretch
    + key_drivers/what_to_watch/scenario_notes. Python-only.
    """
    as_of = as_of or datetime.now(timezone.utc).isoformat()

    # ── Engine calls ─────────────────────────────────────────────────
    sr = compute_structural_risk(financials, sec_data)
    fs = compute_factor_scores(financials, history, peers)
    vs = compute_valuation_stretch(financials, history, peers)

    structural_flag  = sr["structural_risk_flag"]
    hard             = sr["hard_red_flags"]
    soft             = sr["soft_flags"]
    altman           = sr["altman"]
    coverage         = sr["coverage"]
    fcf              = sr["fcf_quality"]
    stretch_level    = vs.get("stretch_level", "low")
    stretch_flag     = vs.get("valuation_stretch_flag", False)

    # ── Evidence ──────────────────────────────────────────────────────
    q = 0.3 if source_name == "mock" else 0.7
    evidence = []
    for key in ["revenue_growth", "operating_margin", "roe", "debt_to_equity",
                "pe_ratio", "ps_ratio", "free_cash_flow", "earnings_growth"]:
        val = financials.get(key)
        if val is not None:
            evidence.append(make_evidence(metric=key, value=val, source_name=source_name, quality=q, as_of=as_of))
    if altman["z_score"] is not None:
        evidence.append(make_evidence(metric="altman_z_score", value=altman["z_score"],
                                      source_name="fundamental_engine", source_type="model", quality=0.9, as_of=as_of,
                                      note=f"bucket={altman['bucket']}"))
    if coverage["interest_coverage"] is not None:
        evidence.append(make_evidence(metric="interest_coverage", value=coverage["interest_coverage"],
                                      source_name="fundamental_engine", source_type="model", quality=0.9, as_of=as_of))

    # ── Risk flags ────────────────────────────────────────────────────
    risk_flags = [make_risk_flag(**f) for f in hard + soft]

    # ── Decision ──────────────────────────────────────────────────────
    growth_score   = fs.get("growth_score", 0.5)
    quality_score  = fs.get("quality_score", 0.5)
    rev_growth     = financials.get("revenue_growth", 0) or 0
    growth_special = rev_growth > 20 and stretch_level == "medium"

    if structural_flag:
        primary_decision = "avoid"; recommendation = "reject"; confidence = 0.35
        notes = f"Structural risk: {', '.join(f['code'] for f in hard)}. Gate3 reject 권고."
    elif stretch_flag and growth_score < 0.5:
        primary_decision = "bearish"; recommendation = "allow_with_limits"; confidence = 0.45
        notes = f"Valuation stretch {stretch_level} + low growth. 비중 축소 권고."
    elif growth_special:
        primary_decision = "neutral"; recommendation = "allow_with_limits"; confidence = 0.55
        notes = f"성장주 특례: stretch {stretch_level} but growth {rev_growth:.0f}%. 트리거 모니터링."
    elif quality_score > 0.65 and fs.get("value_score", 0.5) > 0.50:
        primary_decision = "bullish"; recommendation = "allow"; confidence = 0.65
        notes = "퀄리티 + 밸류 조합 양호. 구조적 리스크 없음."
    elif soft:
        primary_decision = "neutral"; recommendation = "allow_with_limits"; confidence = 0.50
        notes = f"Soft flags: {', '.join(f['code'] for f in soft)}. 비중 축소 권고."
    else:
        primary_decision = "bullish" if rev_growth > 5 else "neutral"
        recommendation = "allow"; confidence = 0.60
        notes = "핵심 구조적 리스크 없음. 재무 건전성 양호."

    # ── signal_strength from factor scores ───────────────────────────
    avg_score = (quality_score + growth_score + fs.get("cashflow_score", 0.5)) / 3
    if stretch_flag:
        avg_score *= 0.7
    if stretch_level == "unknown":
        avg_score *= 0.8  # conservative if valuation unknown
    signal_strength = round(avg_score, 3)

    # ── Data quality ──────────────────────────────────────────────────
    data_ok = any(e["metric"] in ("revenue_growth", "pe_ratio", "altman_z_score") for e in evidence if e.get("value") is not None)
    missing = [k for k in ["pe_ratio", "revenue_growth", "roe"] if not financials.get(k)]

    limitations = []
    if source_name == "mock":
        limitations.append("Mock 재무제표 — 실제 SEC filing과 차이 가능")
    if sec_data is None:
        limitations.append("SEC 텍스트 분석 미수행 — going concern/restatement 확인 불가")
    if not history:
        limitations.append("밸류에이션 히스토리 없음 — 절대 임계치 사용")
    if stretch_level == "unknown":
        limitations.append("Valuation data 부족 — stretch 판단 불가")

    # ── Narrative outputs ─────────────────────────────────────────────
    key_drivers    = _build_key_drivers(sr, vs, fs)
    what_to_watch  = _build_what_to_watch(sr, vs, financials)
    scenario_notes = _build_scenario_notes(sr, vs, fs, financials)

    data_quality = {
        "missing_fields": missing,
        "is_mock":        source_name == "mock",
        "anomaly_flags":  ["valuation_unknown"] if stretch_level == "unknown" else [],
        "source_timestamps": {},
    }

    return {
        "agent_type":      "fundamental",
        "run_id":          run_id,
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "as_of":           as_of,
        "ticker":          ticker,
        "horizon_days":    horizon_days,
        "primary_decision": primary_decision,
        "recommendation":  recommendation,
        "confidence":      confidence if data_ok else min(confidence, 0.4),
        "signal_strength": signal_strength,
        "risk_flags":      risk_flags,
        "evidence":        evidence,
        "data_quality":    data_quality,
        "limitations":     limitations,
        "data_ok":         data_ok,
        "summary":         notes,
        "status":          "ok",
        # Structured
        "factor_scores":         fs,
        "valuation_stretch":     vs,
        "valuation_block": {
            "pe_ratio": financials.get("pe_ratio"),
            "revenue_growth_pct": financials.get("revenue_growth"),
            "roe_pct": financials.get("roe"),
            "debt_to_equity": financials.get("debt_to_equity"),
        },
        "quality_block": {"altman": altman, "coverage": coverage, "fcf_quality": fcf},
        "structural_risk_flag": structural_flag,
        "hard_red_flags":       hard,
        "soft_flags":           soft,
        "key_drivers":          key_drivers,
        "what_to_watch":        what_to_watch,
        "scenario_notes":       scenario_notes,
        "notes_for_risk_manager": notes,
        # Backward compat
        "sector":          financials.get("sector", "Unknown"),
        "revenue_growth":  financials.get("revenue_growth"),
        "roe":             financials.get("roe"),
        "debt_to_equity":  financials.get("debt_to_equity"),
        "pe_ratio":        financials.get("pe_ratio"),
    }
