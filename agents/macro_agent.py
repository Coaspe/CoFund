"""
agents/macro_agent.py — ② Macro Analyst Agent (v2)
====================================================
v2: 5축 분해 + risk_on_off + key_drivers + what_to_watch + scenario_notes
LLM 없음. Python-only. deterministic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

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
    hy  = indicators.get("hy_oas")
    yc  = indicators.get("yield_curve_spread")
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
    risk_on_off = ron.get("risk_on_off", "neutral")
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

    note = scenarios.get(regime, scenarios["expansion"])
    if tail:
        note["bear"] = "⚠️ Tail Risk 活性化 — " + note["bear"]
    return note


# ── Main pipeline ─────────────────────────────────────────────────────────────

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
    Macro Analyst v2: engine compute → 5축/risk_on_off → key_drivers/what_to_watch/scenario_notes.
    LLM 없음. Python-only.
    """
    as_of = as_of or datetime.now(timezone.utc).isoformat()

    # ── Engine calls ─────────────────────────────────────────────────
    features = compute_macro_features(macro_indicators)
    overlay  = compute_overlay_guidance(features)
    axes     = compute_macro_axes(macro_indicators)
    ron      = compute_risk_on_off(axes, macro_indicators)

    regime       = features["macro_regime"]
    # Enhanced tail_risk from ron (more precise than basic features)
    tail_risk    = ron["tail_risk_warning"]
    risk_on_off  = ron["risk_on_off"]

    # ── Evidence ──────────────────────────────────────────────────────
    q = 0.3 if source_name == "mock" else 0.7
    evidence = []
    for key in ["yield_curve_spread", "hy_oas", "inflation_expectation", "cpi_yoy",
                "core_cpi_yoy", "pmi", "fed_funds_rate", "gdp_growth",
                "unemployment_rate", "financial_conditions_index"]:
        val = macro_indicators.get(key)
        if val is not None:
            evidence.append(make_evidence(metric=key, value=val, source_name=source_name, quality=q, as_of=as_of))
    for bk in ["curve_state", "credit_stress_level", "inflation_state", "growth_state"]:
        evidence.append(make_evidence(metric=bk, value=features[bk], source_name="macro_engine",
                                      source_type="model", quality=0.9, as_of=as_of))

    # ── Risk flags ────────────────────────────────────────────────────
    risk_flags = []
    if tail_risk:
        risk_flags.append(make_risk_flag("macro_tail_risk", ron.get("tail_risk_level", "high"),
                                         "Credit stress or yield curve + FCI"))
    if regime in ("contraction", "stagflation"):
        risk_flags.append(make_risk_flag("macro_headwind", "high", f"Macro regime: {regime}"))

    # ── Primary decision ──────────────────────────────────────────────
    inflation_hot = axes.get("inflation", {}).get("score", 0) >= 2
    rates_rising  = axes.get("rates",    {}).get("score", 0) <= -1

    if risk_on_off == "risk_off" and tail_risk:
        primary_decision = "bearish"; recommendation = "reject"; confidence = 0.80
    elif risk_on_off == "risk_off":
        primary_decision = "bearish"; recommendation = "allow_with_limits"; confidence = 0.65
    elif risk_on_off == "risk_on" and inflation_hot and rates_rising:
        primary_decision = "neutral"; recommendation = "allow_with_limits"; confidence = 0.55
    elif risk_on_off == "risk_on":
        primary_decision = "bullish"; recommendation = "allow"; confidence = 0.65
    elif regime == "late_cycle":
        primary_decision = "neutral"; recommendation = "allow_with_limits"; confidence = 0.50
    else:
        primary_decision = "neutral"; recommendation = "allow_with_limits"; confidence = 0.55

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

    # ── key_drivers / what_to_watch / scenario_notes ──────────────────
    key_drivers    = _build_key_drivers(axes, ron, features, macro_indicators)
    what_to_watch  = _build_what_to_watch(axes, macro_indicators)
    scenario_notes = _build_scenario_notes(regime, ron)

    data_quality = {
        "missing_fields":   missing,
        "is_mock":          source_name == "mock",
        "anomaly_flags":    ["tail_risk_active"] if tail_risk else [],
        "source_timestamps": {},
    }

    return {
        "agent_type":      "macro",
        "run_id":          run_id,
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "as_of":           as_of,
        "ticker":          ticker,
        "horizon_days":    horizon_days,
        "primary_decision": primary_decision,
        "recommendation":  recommendation,
        "confidence":      confidence if data_ok else min(confidence, 0.40),
        "signal_strength": abs(ron["risk_score"]) / 100.0,
        "risk_flags":      risk_flags,
        "evidence":        evidence,
        "data_quality":    data_quality,
        "limitations":     limitations,
        "data_ok":         data_ok,
        "summary":         f"리스크: {risk_on_off}. 레짐: {regime}. {overlay.get('equity_overlay_guidance', '')}",
        "status":          "ok",
        # Structured outputs
        "macro_axes":      axes,
        "risk_on_off":     ron,
        "overlay_guidance": overlay,
        "key_drivers":     key_drivers,
        "what_to_watch":   what_to_watch,
        "scenario_notes":  scenario_notes,
        # Backward compat
        "macro_regime_raw": regime,
        "macro_regime":    map_macro_regime_to_canonical(regime),
        "tail_risk_warning": tail_risk,
        "indicators":      features,
        "regime":          map_macro_regime_to_canonical(regime),
        "gdp_growth":      macro_indicators.get("gdp_growth"),
        "interest_rate":   macro_indicators.get("fed_funds_rate"),
    }
