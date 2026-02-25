"""
agents/macro_agent.py — ② Macro Analyst Agent (v2)
====================================================
v2: 5축 분해 + risk_on_off + key_drivers + what_to_watch + scenario_notes
LLM 없음. Python-only. deterministic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from agents.autonomy_overlay import apply_llm_overlay_macro
from engines.macro_engine import (
    compute_macro_features,
    compute_overlay_guidance,
    compute_macro_axes,
    compute_risk_on_off,
)
from schemas.common import make_evidence, make_risk_flag
from schemas.taxonomy import map_macro_regime_to_canonical


# ── Template helpers ──────────────────────────────────────────────────────────

def _axis_label(axis_name: str, state: str, score: int) -> str:
    arrows = {3: "↑↑", 2: "↑", 1: "↑", 0: "→", -1: "↓", -2: "↓", -3: "↓↓"}
    return f"{axis_name.capitalize()} {arrows.get(score, '→')} ({state})"


def _build_key_drivers(axes: dict, ron: dict, features: dict, indicators: dict) -> list[str]:
    drivers = []
    for name, ax in axes.items():
        score = ax["score"]
        state = ax["state"]
        if abs(score) >= 2:
            sign = "+" if score > 0 else "-"
            drivers.append(_axis_label(name, state, score) + f"  [{sign}{abs(score)}]")

    hy = indicators.get("hy_oas")
    yc = indicators.get("yield_curve_spread")
    pmi = indicators.get("pmi")

    if hy and hy > 400:
        drivers.append(f"Credit spreads elevated (HY OAS {hy:.0f}bp)")
    if yc is not None and yc < -0.10:
        drivers.append(f"Yield curve inverted ({yc:+.2f}%)")
    if pmi and pmi < 50:
        drivers.append(f"PMI in contraction ({pmi:.1f})")
    if ron.get("tail_risk_warning"):
        drivers.append(f"Tail risk: {ron.get('tail_risk_level', 'medium').upper()}")
    return drivers[:6]


def _build_what_to_watch(axes: dict, indicators: dict) -> list[str]:
    items = []
    hy = indicators.get("hy_oas")
    yc = indicators.get("yield_curve_spread")
    pmi = indicators.get("pmi")
    ffr = indicators.get("fed_funds_rate")

    if hy and hy > 400:
        normal_hy = 300
        items.append(f"HY OAS < {normal_hy}bp 복귀 시 크레딧 완화 신호")
    if yc is not None and yc < 0:
        items.append("10Y-2Y 스프레드 플러스 전환 시 레짐 개선 확인")
    if pmi:
        items.append(f"PMI 50 상향 돌파 여부 (현재 {pmi:.1f})")
    if ffr:
        items.append(f"Fed 금리 인하 신호 — 현 FFR {ffr:.2f}%")
    if axes.get("growth", {}).get("score", 0) < -1:
        items.append("GDP/PMI 반등 확인 후 리스크 자산 편입 재고")
    items.append("다음 FOMC / CPI 발표 결과 모니터링")
    return items[:5]


def _build_scenario_notes(regime: str, ron: dict) -> dict:
    tail = ron.get("tail_risk_warning", False)

    scenarios: dict[str, dict] = {
        "goldilocks": {
            "bull": "성장 가속 + 인플레 완화 지속 → 리스크 자산 전반 강세. 주식 비중 확대 적기.",
            "base": "Goldilocks 지속. 이익 성장 + 밸류에이션 재평가. 분할 매수 유효.",
            "bear": "인플레 재점화 또는 연준 매파 전환 시 조정. 금리 민감 섹터 익스포저 축소.",
        },
        "reflation": {
            "bull": "실적 서프라이즈 + 원자재 상승 수혜 — 가치주/금융/에너지 강세.",
            "base": "금리 상승 속도 조절 국면. 듀레이션 축소, 실물 자산 비중 유지.",
            "bear": "인플레 통제 실패 시 긴축 가속 → 성장주 밸류에이션 압박. 현금 비중 확대.",
        },
        "late_cycle": {
            "bull": "연착륙 성공 시 방어 + 퀄리티 주 상대 우위 유지. 하방 헤지 비용 저렴.",
            "base": "점진적 노출 축소. 퀄리티/저변동 팩터 위주, 사이클 익스포저 감소.",
            "bear": "경기 침체 선반영 시 급락. 채권 듀레이션 확대, 현금 20-30% 권고.",
        },
        "stagflation": {
            "bull": "에너지·원자재·실물 헤지로 인플레 수혜 제한적. 리얼 에셋 유지.",
            "base": "방어 섹터(헬스케어·필수소비재)와 현금 혼합. 주식 비중 최소화.",
            "bear": "스태그 심화 시 전 자산 동반 하락 가능. 현금 + 금·단기채 최대 방어.",
        },
        "contraction": {
            "bull": "정책 반전(금리 인하/QE) 확인 직후 반등. 그때까지 현금 대기.",
            "base": "현금·국채·금 위주. 주식 최소. 경기 바닥 확인 후 분할 편입.",
            "bear": "침체 장기화 시 기업 실적 급락, 신용 이벤트 위험. 숏 헤지 필수.",
        },
        "expansion": {
            "bull": "성장 모멘텀 지속 + 유동성 확장 → 사이클·성장주 강세.",
            "base": "정상 확장기. 분산 포트폴리오, 중기 롱 포지션 유지.",
            "bear": "공급 충격 또는 지정학 이벤트로 급격한 레짐 전환 위험. 헤지 비중 일부 확보.",
        },
    }

    note = dict(scenarios.get(regime, scenarios["expansion"]))
    if tail:
        note["bear"] = "⚠️ Tail Risk 活性化 — " + note["bear"]
    return note


_MACRO_EVIDENCE_KINDS = {"macro_headline_context", "macro_release", "press_release_or_ir"}
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
    return (
        req.get("desk", ""),
        req.get("kind", ""),
        req.get("ticker", ""),
        req.get("series_id", ""),
        req.get("query", ""),
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


def _build_evidence_digest(state: Optional[dict], ticker: str, max_items: int = 5) -> list[dict]:
    store = (state or {}).get("evidence_store", {})
    if not isinstance(store, dict):
        return []
    out: list[dict] = []
    for item in store.values():
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", ""))
        desk = str(item.get("desk", ""))
        if kind and kind not in _MACRO_EVIDENCE_KINDS and desk not in ("macro", ""):
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


def _default_open_questions(ticker: str, focus_areas: list[str], regime: str) -> list[dict]:
    out: list[dict] = []
    for focus in focus_areas[:2]:
        out.append(
            {
                "q": f"{focus}가 {ticker}의 매크로 레짐({regime})을 바꿀 촉매인지?",
                "why": "결론 변경 가능성이 큰 매크로 불확실성 확인",
                "kind": "macro_headline_context",
                "priority": 2,
                "recency_days": 7,
            }
        )
    if not out:
        out.append(
            {
                "q": f"{ticker} 관련 최근 매크로 헤드라인 중 레짐 전환 신호가 있는가?",
                "why": "risk_on/off 판단의 최신 근거 보강",
                "kind": "macro_headline_context",
                "priority": 2,
                "recency_days": 7,
            }
        )
    return out[:5]


def _default_decision_sensitivity(regime: str) -> list[dict]:
    return [
        {
            "if": "HY OAS가 50bp 이상 추가 확대",
            "then_change": "매크로 결론을 한 단계 보수적으로 조정",
            "impact": "high",
        },
        {
            "if": f"현재 레짐({regime})이 다음 주요 지표 발표 후 유지",
            "then_change": "기존 결론 유지",
            "impact": "medium",
        },
    ]


def _default_followups() -> list[dict]:
    return [
        {
            "type": "run_research",
            "detail": "매크로 이벤트/공식 릴리즈 근거를 추가 수집",
            "params": {"kind": "macro_headline_context"},
        },
        {
            "type": "rerun_desk",
            "detail": "새 근거 반영 후 매크로 데스크 재평가",
            "params": {"desk": "macro"},
        },
    ]


def _default_react_trace(has_evidence: bool) -> list[dict]:
    return [
        {"phase": "THOUGHT", "summary": "결론을 바꿀 거시 변수 공백을 점검"},
        {"phase": "ACTION", "summary": "우선순위 질문과 리서치 요청을 구조화"},
        {
            "phase": "OBSERVATION",
            "summary": "증거 반영 여부를 핵심 드라이버에 업데이트" if has_evidence else "신규 증거 대기 상태",
        },
        {"phase": "REFLECTION", "summary": "결론 민감도와 조건부 시나리오를 재확인"},
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
    axes: dict,
    ron: dict,
    features: dict,
    indicators: dict,
    focus_areas: Optional[list[str]] = None,
) -> list:
    """매크로 evidence request 생성: regime flip/axes 급변 + driver 불명."""
    reqs = []
    risk_on_off = ron.get("risk_on_off", "neutral")
    if risk_on_off == "risk_off" and not indicators.get("yield_curve_spread"):
        reqs.append(
            {
                "desk": "macro",
                "kind": "macro_headline_context",
                "ticker": ticker,
                "query": f"{ticker} macro headwind driver recession risk",
                "priority": 2,
                "recency_days": 7,
                "max_items": 5,
                "rationale": "risk_off but missing yield curve — need context",
            }
        )
    if ron.get("tail_risk_warning") and not indicators.get("hy_oas"):
        reqs.append(
            {
                "desk": "macro",
                "kind": "macro_headline_context",
                "ticker": ticker,
                "query": "high yield credit spread stress financial conditions",
                "priority": 1,
                "recency_days": 3,
                "max_items": 3,
                "rationale": "tail_risk active but HY OAS missing",
            }
        )
    g_score = axes.get("growth", {}).get("score", 0)
    if abs(g_score) >= 2 and not indicators.get("pmi") and not indicators.get("gdp_growth"):
        reqs.append(
            {
                "desk": "macro",
                "kind": "macro_headline_context",
                "ticker": ticker,
                "query": "US PMI GDP growth contraction latest",
                "priority": 2,
                "recency_days": 7,
                "max_items": 3,
                "rationale": f"growth axis extreme ({g_score}) but PMI/GDP missing",
            }
        )
    prev_regime = indicators.get("prev_macro_regime")
    curr_regime = features.get("macro_regime")
    if prev_regime and curr_regime and prev_regime != curr_regime and not indicators.get("regime_flip_driver"):
        reqs.append(
            {
                "desk": "macro",
                "kind": "macro_headline_context",
                "ticker": ticker,
                "query": f"{ticker} macro regime flip driver",
                "priority": 1,
                "recency_days": 7,
                "max_items": 4,
                "rationale": f"regime flip {prev_regime}->{curr_regime} with unknown driver",
            }
        )

    for focus in (focus_areas or [])[:2]:
        reqs.append(
            {
                "desk": "macro",
                "kind": "macro_headline_context",
                "ticker": ticker,
                "query": f"{ticker} {focus} macro impact",
                "priority": 3,
                "recency_days": 14,
                "max_items": 3,
                "rationale": f"focus area follow-up: {focus}",
            }
        )
    return reqs


# ── Main pipeline ─────────────────────────────────────────────────────────────

def macro_analyst_run(
    ticker: str,
    macro_indicators: dict,
    *,
    run_id: str = "",
    as_of: str = "",
    horizon_days: int = 30,
    source_name: str = "mock",
    focus_areas: Optional[list[str]] = None,
    state: Optional[dict] = None,
) -> dict:
    """
    Macro Analyst v2: engine compute → 5축/risk_on_off → key_drivers/what_to_watch/scenario_notes.
    LLM overlay is optional and patch-only.
    """
    as_of = as_of or datetime.now(timezone.utc).isoformat()
    focus_areas = [str(x).strip() for x in (focus_areas or []) if str(x).strip()]

    # ── Engine calls ─────────────────────────────────────────────────
    features = compute_macro_features(macro_indicators)
    overlay = compute_overlay_guidance(features)
    axes = compute_macro_axes(macro_indicators)
    ron = compute_risk_on_off(axes, macro_indicators)

    regime = features["macro_regime"]
    tail_risk = ron["tail_risk_warning"]
    risk_on_off = ron["risk_on_off"]

    # ── Evidence ──────────────────────────────────────────────────────
    q = 0.3 if source_name == "mock" else 0.7
    evidence = []
    for key in [
        "yield_curve_spread",
        "hy_oas",
        "inflation_expectation",
        "cpi_yoy",
        "core_cpi_yoy",
        "pmi",
        "fed_funds_rate",
        "gdp_growth",
        "unemployment_rate",
        "financial_conditions_index",
    ]:
        val = macro_indicators.get(key)
        if val is not None:
            evidence.append(make_evidence(metric=key, value=val, source_name=source_name, quality=q, as_of=as_of))
    for bk in ["curve_state", "credit_stress_level", "inflation_state", "growth_state"]:
        evidence.append(
            make_evidence(
                metric=bk,
                value=features[bk],
                source_name="macro_engine",
                source_type="model",
                quality=0.9,
                as_of=as_of,
            )
        )

    # ── Risk flags ────────────────────────────────────────────────────
    risk_flags = []
    if tail_risk:
        risk_flags.append(
            make_risk_flag(
                "macro_tail_risk",
                ron.get("tail_risk_level", "high"),
                "Credit stress or yield curve + FCI",
            )
        )
    if regime in ("contraction", "stagflation"):
        risk_flags.append(make_risk_flag("macro_headwind", "high", f"Macro regime: {regime}"))

    # ── Primary decision ──────────────────────────────────────────────
    inflation_hot = axes.get("inflation", {}).get("score", 0) >= 2
    rates_rising = axes.get("rates", {}).get("score", 0) <= -1

    if risk_on_off == "risk_off" and tail_risk:
        primary_decision = "bearish"
        recommendation = "reject"
        confidence = 0.80
    elif risk_on_off == "risk_off":
        primary_decision = "bearish"
        recommendation = "allow_with_limits"
        confidence = 0.65
    elif risk_on_off == "risk_on" and inflation_hot and rates_rising:
        primary_decision = "neutral"
        recommendation = "allow_with_limits"
        confidence = 0.55
    elif risk_on_off == "risk_on":
        primary_decision = "bullish"
        recommendation = "allow"
        confidence = 0.65
    elif regime == "late_cycle":
        primary_decision = "neutral"
        recommendation = "allow_with_limits"
        confidence = 0.50
    else:
        primary_decision = "neutral"
        recommendation = "allow_with_limits"
        confidence = 0.55

    # ── Data quality ──────────────────────────────────────────────────
    keyed = {k for k in ["yield_curve_spread", "hy_oas", "gdp_growth", "pmi"] if macro_indicators.get(k) is not None}
    data_ok = len(keyed) >= 2
    missing = [k for k in ["yield_curve_spread", "hy_oas", "pmi", "cpi_yoy"] if macro_indicators.get(k) is None]
    if not data_ok:
        confidence = min(confidence, 0.40)

    limitations = []
    if source_name == "mock":
        limitations.append("Mock 데이터 사용 — 실제 매크로 지표와 차이 가능")
    if missing:
        limitations.append(f"누락 지표: {', '.join(missing)}")

    ev_reqs = _generate_evidence_requests(ticker, axes, ron, features, macro_indicators, focus_areas)

    # ── key_drivers / what_to_watch / scenario_notes ──────────────────
    key_drivers = _build_key_drivers(axes, ron, features, macro_indicators)
    what_to_watch = _build_what_to_watch(axes, macro_indicators)
    scenario_notes = _build_scenario_notes(regime, ron)
    evidence_digest = _build_evidence_digest(state, ticker)
    if evidence_digest:
        title = str(evidence_digest[0].get("title", "external evidence")).strip()[:90]
        kinds = sorted({str(item.get("kind", "")).strip() for item in evidence_digest if item.get("kind")})
        kind_label = ", ".join(kinds[:2]) if kinds else "macro_evidence"
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
                "신규 증거와 매크로 축 점수의 불일치 여부 재확인",
                f"최신 근거 경로({resolver})의 후속 발표 추적",
            ]
        )[:5]

    open_questions = _default_open_questions(ticker, focus_areas, regime)
    decision_sensitivity = _default_decision_sensitivity(regime)
    followups = _default_followups()
    react_trace = _default_react_trace(bool(evidence_digest))

    total_fields = 4
    missing_pct = round(len(missing) / total_fields, 3)
    data_quality = {
        "missing_pct": missing_pct,
        "freshness_days": 0.0,
        "warnings": list(missing),
        "missing_fields": missing,
        "is_mock": source_name == "mock",
        "anomaly_flags": ["tail_risk_active"] if tail_risk else [],
        "source_timestamps": {},
    }

    output = {
        "agent_type": "macro",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of": as_of,
        "ticker": ticker,
        "horizon_days": horizon_days,
        "focus_areas": focus_areas,
        "primary_decision": primary_decision,
        "recommendation": recommendation,
        "confidence": confidence if data_ok else min(confidence, 0.40),
        "signal_strength": abs(ron["risk_score"]) / 100.0,
        "risk_flags": risk_flags,
        "evidence": evidence,
        "data_quality": data_quality,
        "limitations": limitations,
        "data_ok": data_ok,
        "summary": f"리스크: {risk_on_off}. 레짐: {regime}. {overlay.get('equity_overlay_guidance', '')}",
        "status": "ok",
        # Structured outputs
        "macro_axes": axes,
        "risk_on_off": ron,
        "overlay_guidance": overlay,
        "key_drivers": key_drivers,
        "what_to_watch": what_to_watch,
        "scenario_notes": scenario_notes,
        "evidence_digest": evidence_digest,
        "open_questions": open_questions,
        "decision_sensitivity": decision_sensitivity,
        "followups": followups,
        "react_trace": react_trace,
        # Evidence requests
        "needs_more_data": bool(ev_reqs),
        "evidence_requests": ev_reqs,
        # Backward compat
        "macro_regime_raw": regime,
        "macro_regime": map_macro_regime_to_canonical(regime),
        "tail_risk_warning": tail_risk,
        "indicators": features,
        "regime": map_macro_regime_to_canonical(regime),
        "gdp_growth": macro_indicators.get("gdp_growth"),
        "interest_rate": macro_indicators.get("fed_funds_rate"),
    }

    patch = apply_llm_overlay_macro(output, state, focus_areas, evidence_digest)
    _apply_overlay_patch(output, patch)

    if not output.get("open_questions"):
        output["open_questions"] = open_questions
    if not output.get("decision_sensitivity"):
        output["decision_sensitivity"] = decision_sensitivity
    if not output.get("followups"):
        output["followups"] = followups

    return output
