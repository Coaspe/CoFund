"""
agents/fundamental_agent.py — ③ Fundamental Analyst Agent (v2)
===============================================================
v2: factor_scores (5축) + valuation_stretch + key_drivers + what_to_watch + scenario_notes.
LLM 없음. Python-only. deterministic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from agents.autonomy_overlay import apply_llm_overlay_fundamental
from engines.fundamental_engine import (
    compute_structural_risk,
    compute_factor_scores,
    compute_valuation_stretch,
)
from schemas.common import make_evidence, make_risk_flag


# ── Template helpers ──────────────────────────────────────────────────────────

def _build_key_drivers(sr: dict, vs: dict, fs: dict) -> list[str]:
    drivers = []
    altman = sr.get("altman", {})
    cov = sr.get("coverage", {})
    fcf = sr.get("fcf_quality", {})

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
    stretch = vs.get("stretch_level", "low")
    quality = fs.get("quality_score", 0.5)
    growth = fs.get("growth_score", 0.5)
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


_FUNDA_EVIDENCE_KINDS = {
    "valuation_context",
    "ownership_identity",
    "press_release_or_ir",
    "sec_filing",
    "web_search",
}
_PATCHABLE_FIELDS = {
    "key_drivers",
    "what_to_watch",
    "scenario_notes",
    "open_questions",
    "decision_sensitivity",
    "followups",
    "react_trace",
}


def _request_key(req: dict) -> tuple:
    query = " ".join(str(req.get("query", "")).strip().lower().split())
    return (
        str(req.get("desk", "")).strip().lower(),
        str(req.get("kind", "")).strip().lower(),
        str(req.get("ticker", "")).strip().upper(),
        str(req.get("series_id", "")).strip(),
        query,
    )


def _merge_requests(base: list[dict], extra: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen = set()
    for req in (base or []) + (extra or []):
        if not isinstance(req, dict):
            continue
        k = _request_key(req)
        if k in seen:
            continue
        seen.add(k)
        out.append(req)
    return out


def _build_evidence_digest(state: Optional[dict], ticker: str, max_items: int = 7) -> list[dict]:
    store = (state or {}).get("evidence_store", {})
    if not isinstance(store, dict):
        return []
    out: list[dict] = []
    for item in store.values():
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", ""))
        desk = str(item.get("desk", ""))
        if kind and kind not in _FUNDA_EVIDENCE_KINDS and desk not in ("fundamental", ""):
            continue
        item_ticker = str(item.get("ticker", "")).upper()
        if item_ticker and item_ticker != ticker.upper():
            continue
        out.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "published_at": item.get("published_at", ""),
                "trust_tier": item.get("trust_tier", 0.4),
                "kind": kind,
                "resolver_path": item.get("resolver_path", ""),
            }
        )
    out.sort(key=lambda x: (x.get("published_at") or ""), reverse=True)
    return out[:max_items]


def _default_open_questions(ticker: str, focus_areas: list[str], stretch: str) -> list[dict]:
    out: list[dict] = []
    for focus in focus_areas[:2]:
        out.append(
            {
                "q": f"{focus} 관점에서 {ticker} 밸류에이션/실적 정합성이 유지되는가?",
                "why": "결론 변경 가능성이 높은 펀더멘털 공백 확인",
                "kind": "valuation_context",
                "priority": 2,
                "recency_days": 90,
            }
        )
    if stretch in ("high", "unknown"):
        out.append(
            {
                "q": f"{ticker}의 최근 공시/IR에서 밸류에이션 정당화 근거가 제시됐는가?",
                "why": "stretch 구간에서 내러티브 기반 리레이팅 위험 점검",
                "kind": "press_release_or_ir",
                "priority": 2,
                "recency_days": 30,
            }
        )
    if not out:
        out.append(
            {
                "q": f"{ticker}의 ownership/valuation 업데이트 중 결론을 바꿀 신호가 있는가?",
                "why": "핵심 변수 변화 탐지",
                "kind": "ownership_identity",
                "priority": 3,
                "recency_days": 90,
            }
        )
    return out[:5]


def _default_decision_sensitivity(stretch: str, structural_flag: bool) -> list[dict]:
    out = [
        {
            "if": "다음 분기 가이던스가 컨센서스 대비 하향",
            "then_change": "펀더멘털 결론을 보수적으로 1단계 하향",
            "impact": "high",
        }
    ]
    if stretch in ("high", "unknown"):
        out.append(
            {
                "if": "peer 대비 valuation premium 축소 근거 확보",
                "then_change": "과열 판단을 중립으로 완화",
                "impact": "medium",
            }
        )
    if structural_flag:
        out.append(
            {
                "if": "구조적 리스크 플래그 해소 공시 확인",
                "then_change": "avoid 판단 해제 검토",
                "impact": "high",
            }
        )
    return out[:5]


def _default_followups() -> list[dict]:
    return [
        {
            "type": "run_research",
            "detail": "SEC/IR/ownership 근거를 보강해 valuation 논리를 검증",
            "params": {"desk": "fundamental"},
        },
        {
            "type": "rerun_desk",
            "detail": "증거 반영 후 펀더멘털 결론 재산출",
            "params": {"desk": "fundamental"},
        },
    ]


def _default_react_trace(has_evidence: bool) -> list[dict]:
    return [
        {"phase": "THOUGHT", "summary": "결론을 바꿀 재무/공시 공백을 식별"},
        {"phase": "ACTION", "summary": "질문과 리서치 요청을 우선순위화"},
        {
            "phase": "OBSERVATION",
            "summary": "신규 증거로 밸류/구조 리스크 서술을 갱신" if has_evidence else "증거 수집 전 기본 판단 유지",
        },
        {"phase": "REFLECTION", "summary": "조건부 결론 민감도를 정리"},
    ]


def _apply_overlay_patch(output: dict, patch: dict) -> None:
    if not isinstance(patch, dict):
        return
    for key in _PATCHABLE_FIELDS:
        if key in patch and patch[key]:
            output[key] = patch[key]
    if patch.get("evidence_requests"):
        output["evidence_requests"] = _merge_requests(
            output.get("evidence_requests", []),
            patch.get("evidence_requests", []),
        )
        output["needs_more_data"] = bool(output.get("evidence_requests"))


def _generate_evidence_requests(
    ticker: str,
    sr: dict,
    vs: dict,
    fs: dict,
    financials: dict,
    asset_type: str = "EQUITY",
    focus_areas: Optional[list[str]] = None,
) -> list:
    """펀더멘탈 evidence request 생성."""
    at = str(asset_type or "EQUITY").upper()
    if at in {"ETF", "INDEX"}:
        reqs = [
            {
                "desk": "fundamental",
                "kind": "valuation_context",
                "ticker": ticker,
                "query": f"{ticker} holdings top 10 sector weights factor exposures index forward PE",
                "priority": 1,
                "recency_days": 30,
                "max_items": 5,
                "rationale": f"{at.lower()} mode: holdings/sector/valuation aggregate",
            },
            {
                "desk": "fundamental",
                "kind": "web_search",
                "ticker": ticker,
                "query": f"{ticker} ETF flow creation redemption tracking error expense ratio liquidity",
                "priority": 2,
                "recency_days": 30,
                "max_items": 5,
                "rationale": f"{at.lower()} mode: flow/liquidity diagnostics",
            },
        ]
        for focus in (focus_areas or [])[:2]:
            reqs.append(
                {
                    "desk": "fundamental",
                    "kind": "valuation_context",
                    "ticker": ticker,
                    "query": f"{ticker} {focus} sector breadth valuation",
                    "priority": 3,
                    "recency_days": 120,
                    "max_items": 3,
                    "rationale": f"{at.lower()} focus follow-up: {focus}",
                }
            )
        return reqs

    reqs = []
    stretch = vs.get("stretch_level", "low")
    if stretch in ("high", "unknown") and vs.get("confidence_down"):
        reqs.append(
            {
                "desk": "fundamental",
                "kind": "valuation_context",
                "ticker": ticker,
                "query": f"{ticker} PE ratio historical 5 year average peer comparison",
                "priority": 2,
                "recency_days": 90,
                "max_items": 5,
                "rationale": f"valuation stretch={stretch} but history/peers missing",
            }
        )
    if not financials.get("insider_transactions") and not financials.get("institutional_holders"):
        reqs.append(
            {
                "desk": "fundamental",
                "kind": "ownership_identity",
                "ticker": ticker,
                "query": f"{ticker} insider trading institutional ownership SEC Form 4",
                "priority": 3,
                "recency_days": 30,
                "max_items": 3,
                "rationale": "insider/institutional data missing",
            }
        )
    earnings_date = financials.get("next_earnings_date")
    earnings_time = financials.get("next_earnings_time")
    earnings_in_days = financials.get("earnings_in_days")
    if earnings_in_days is not None and earnings_in_days <= 14 and (not earnings_date or not earnings_time):
        reqs.append(
            {
                "desk": "fundamental",
                "kind": "press_release_or_ir",
                "ticker": ticker,
                "query": f"{ticker} earnings date next quarter investor relations",
                "priority": 2,
                "recency_days": 14,
                "max_items": 3,
                "rationale": "earnings <=14d and date/time unknown",
            }
        )
        reqs.append(
            {
                "desk": "fundamental",
                "kind": "web_search",
                "ticker": ticker,
                "query": f"{ticker} earnings call date time announced",
                "priority": 2,
                "recency_days": 14,
                "max_items": 3,
                "rationale": "earnings imminent but schedule fields missing",
            }
        )
    elif not earnings_date:
        reqs.append(
            {
                "desk": "fundamental",
                "kind": "press_release_or_ir",
                "ticker": ticker,
                "query": f"{ticker} earnings date next quarter investor relations",
                "priority": 2,
                "recency_days": 14,
                "max_items": 3,
                "rationale": "next earnings date unknown",
            }
        )
    for flag in sr.get("hard_red_flags", []):
        code = flag.get("code", "")
        if code in ("going_concern", "accounting_fraud", "restatement"):
            reqs.append(
                {
                    "desk": "fundamental",
                    "kind": "sec_filing",
                    "ticker": ticker,
                    "query": f"{ticker} {code} SEC filing 10-K 10-Q",
                    "priority": 1,
                    "recency_days": 90,
                    "max_items": 3,
                    "rationale": f"structural risk flag: {code}",
                }
            )
            break

    for focus in (focus_areas or [])[:2]:
        reqs.append(
            {
                "desk": "fundamental",
                "kind": "valuation_context",
                "ticker": ticker,
                "query": f"{ticker} {focus} valuation context",
                "priority": 3,
                "recency_days": 120,
                "max_items": 3,
                "rationale": f"focus area follow-up: {focus}",
            }
        )

    return reqs


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
    focus_areas: Optional[list[str]] = None,
    state: Optional[dict] = None,
    asset_type: str = "EQUITY",
) -> dict:
    """
    Fundamental Analyst v2: structural risk + factor_scores + valuation_stretch
    + key_drivers/what_to_watch/scenario_notes. Python-only.
    """
    as_of = as_of or datetime.now(timezone.utc).isoformat()
    focus_areas = [str(x).strip() for x in (focus_areas or []) if str(x).strip()]
    asset_type = str(asset_type or "EQUITY").upper()

    sr = compute_structural_risk(financials, sec_data)
    fs = compute_factor_scores(financials, history, peers)
    vs = compute_valuation_stretch(financials, history, peers)

    structural_flag = sr["structural_risk_flag"]
    hard = sr["hard_red_flags"]
    soft = sr["soft_flags"]
    altman = sr["altman"]
    coverage = sr["coverage"]
    fcf = sr["fcf_quality"]
    stretch_level = vs.get("stretch_level", "low")
    stretch_flag = vs.get("valuation_stretch_flag", False)

    q = 0.3 if source_name == "mock" else 0.7
    evidence = []
    for key in [
        "revenue_growth",
        "operating_margin",
        "roe",
        "debt_to_equity",
        "pe_ratio",
        "ps_ratio",
        "free_cash_flow",
        "earnings_growth",
    ]:
        val = financials.get(key)
        if val is not None:
            evidence.append(make_evidence(metric=key, value=val, source_name=source_name, quality=q, as_of=as_of))
    if altman["z_score"] is not None:
        evidence.append(
            make_evidence(
                metric="altman_z_score",
                value=altman["z_score"],
                source_name="fundamental_engine",
                source_type="model",
                quality=0.9,
                as_of=as_of,
                note=f"bucket={altman['bucket']}",
            )
        )
    if coverage["interest_coverage"] is not None:
        evidence.append(
            make_evidence(
                metric="interest_coverage",
                value=coverage["interest_coverage"],
                source_name="fundamental_engine",
                source_type="model",
                quality=0.9,
                as_of=as_of,
            )
        )

    risk_flags = [make_risk_flag(**f) for f in hard + soft]

    growth_score = fs.get("growth_score", 0.5)
    quality_score = fs.get("quality_score", 0.5)
    rev_growth = financials.get("revenue_growth", 0) or 0
    growth_special = rev_growth > 20 and stretch_level == "medium"

    if structural_flag:
        primary_decision = "avoid"
        recommendation = "reject"
        confidence = 0.35
        notes = f"Structural risk: {', '.join(f['code'] for f in hard)}. Gate3 reject 권고."
    elif stretch_flag and growth_score < 0.5:
        primary_decision = "bearish"
        recommendation = "allow_with_limits"
        confidence = 0.45
        notes = f"Valuation stretch {stretch_level} + low growth. 비중 축소 권고."
    elif growth_special:
        primary_decision = "neutral"
        recommendation = "allow_with_limits"
        confidence = 0.55
        notes = f"성장주 특례: stretch {stretch_level} but growth {rev_growth:.0f}%. 트리거 모니터링."
    elif quality_score > 0.65 and fs.get("value_score", 0.5) > 0.50:
        primary_decision = "bullish"
        recommendation = "allow"
        confidence = 0.65
        notes = "퀄리티 + 밸류 조합 양호. 구조적 리스크 없음."
    elif soft:
        primary_decision = "neutral"
        recommendation = "allow_with_limits"
        confidence = 0.50
        notes = f"Soft flags: {', '.join(f['code'] for f in soft)}. 비중 축소 권고."
    else:
        primary_decision = "bullish" if rev_growth > 5 else "neutral"
        recommendation = "allow"
        confidence = 0.60
        notes = "핵심 구조적 리스크 없음. 재무 건전성 양호."

    if asset_type in {"ETF", "INDEX"}:
        primary_decision = "neutral"
        recommendation = "allow_with_limits"
        confidence = min(confidence, 0.55)
        notes = (
            f"{asset_type} 모드: 기업 단일 재무 대신 holdings/sector/flow/aggregate valuation 중심으로 평가."
        )

    avg_score = (quality_score + growth_score + fs.get("cashflow_score", 0.5)) / 3
    if stretch_flag:
        avg_score *= 0.7
    if stretch_level == "unknown":
        avg_score *= 0.8
    signal_strength = round(avg_score, 3)

    data_ok = any(
        e["metric"] in ("revenue_growth", "pe_ratio", "altman_z_score") for e in evidence if e.get("value") is not None
    )
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

    ev_reqs = _generate_evidence_requests(
        ticker,
        sr,
        vs,
        fs,
        financials,
        asset_type=asset_type,
        focus_areas=focus_areas,
    )

    key_drivers = _build_key_drivers(sr, vs, fs)
    what_to_watch = _build_what_to_watch(sr, vs, financials)
    scenario_notes = _build_scenario_notes(sr, vs, fs, financials)
    evidence_digest = _build_evidence_digest(state, ticker)
    if evidence_digest:
        title = str(evidence_digest[0].get("title", "external evidence")).strip()[:90]
        kinds = sorted({str(item.get("kind", "")).strip() for item in evidence_digest if item.get("kind")})
        kind_label = ", ".join(kinds[:2]) if kinds else "fundamental_evidence"
        resolver = str(evidence_digest[0].get("resolver_path", "")).strip() or "unknown"
        key_drivers = (
            key_drivers
            + [
                f"Evidence update: {title}",
                f"Evidence coverage: {len(evidence_digest)} items ({kind_label})",
            ]
        )[:6]
        what_to_watch = (
            what_to_watch
            + [
                "신규 근거가 밸류에이션 결론과 충돌하는지 재확인",
                f"최신 근거 경로({resolver})의 공시/IR 후속 확인",
            ]
        )[:5]

    open_questions = _default_open_questions(ticker, focus_areas, stretch_level)
    decision_sensitivity = _default_decision_sensitivity(stretch_level, structural_flag)
    followups = _default_followups()
    react_trace = _default_react_trace(bool(evidence_digest))

    total_fields = 3
    missing_pct = round(len(missing) / total_fields, 3)
    data_quality = {
        "missing_pct": missing_pct,
        "freshness_days": 0.0,
        "warnings": list(missing),
        "missing_fields": missing,
        "is_mock": source_name == "mock",
        "anomaly_flags": ["valuation_unknown"] if stretch_level == "unknown" else [],
        "source_timestamps": {},
    }

    output = {
        "agent_type": "fundamental",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of": as_of,
        "ticker": ticker,
        "asset_type": asset_type,
        "analysis_mode": "etf_index" if asset_type in {"ETF", "INDEX"} else "equity",
        "horizon_days": horizon_days,
        "focus_areas": focus_areas,
        "primary_decision": primary_decision,
        "recommendation": recommendation,
        "confidence": confidence if data_ok else min(confidence, 0.4),
        "signal_strength": signal_strength,
        "risk_flags": risk_flags,
        "evidence": evidence,
        "data_quality": data_quality,
        "limitations": limitations,
        "data_ok": data_ok,
        "summary": notes,
        "status": "ok",
        # Structured
        "factor_scores": fs,
        "valuation_stretch": vs,
        "valuation_block": {
            "pe_ratio": financials.get("pe_ratio"),
            "revenue_growth_pct": financials.get("revenue_growth"),
            "roe_pct": financials.get("roe"),
            "debt_to_equity": financials.get("debt_to_equity"),
        },
        "quality_block": {"altman": altman, "coverage": coverage, "fcf_quality": fcf},
        "structural_risk_flag": structural_flag,
        "hard_red_flags": hard,
        "soft_flags": soft,
        "key_drivers": key_drivers,
        "what_to_watch": what_to_watch,
        "scenario_notes": scenario_notes,
        "evidence_digest": evidence_digest,
        "open_questions": open_questions,
        "decision_sensitivity": decision_sensitivity,
        "followups": followups,
        "react_trace": react_trace,
        "notes_for_risk_manager": notes,
        "needs_more_data": bool(ev_reqs),
        "evidence_requests": ev_reqs,
        "etf_index_block": {
            "holdings_top10_weight_pct": financials.get("holdings_top10_weight_pct"),
            "sector_weights": financials.get("sector_weights", {}),
            "factor_exposures": financials.get("factor_exposures", {}),
            "index_forward_pe": financials.get("index_forward_pe"),
            "net_flow_1m": financials.get("net_flow_1m"),
            "tracking_error": financials.get("tracking_error"),
            "expense_ratio": financials.get("expense_ratio"),
            "liquidity_score": financials.get("liquidity_score"),
        },
        # Backward compat
        "sector": financials.get("sector", "ETF/Index Basket" if asset_type in {"ETF", "INDEX"} else "Unknown"),
        "revenue_growth": None if asset_type in {"ETF", "INDEX"} else financials.get("revenue_growth"),
        "roe": None if asset_type in {"ETF", "INDEX"} else financials.get("roe"),
        "debt_to_equity": financials.get("debt_to_equity"),
        "pe_ratio": None if asset_type in {"ETF", "INDEX"} else financials.get("pe_ratio"),
    }

    patch = apply_llm_overlay_fundamental(output, state, focus_areas, evidence_digest)
    _apply_overlay_patch(output, patch)

    if not output.get("open_questions"):
        output["open_questions"] = open_questions
    if not output.get("decision_sensitivity"):
        output["decision_sensitivity"] = decision_sensitivity
    if not output.get("followups"):
        output["followups"] = followups

    return output
