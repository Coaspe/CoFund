"""
engines/macro_engine.py — Macro 순수 연산 엔진
===============================================
CHANGELOG:
  v1.0 (2026-02-22) — 신규 생성.

데이터 수집 금지. 입력=지표 dict, 출력=버켓/레짐/오버레이 JSON.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def compute_macro_features(indicators: dict) -> dict:
    """
    핵심 매크로 지표에서 state bucket + regime을 파생.

    Args:
        indicators: {
            "yield_curve_spread": float (e.g., T10Y2Y bp),
            "hy_oas": float (High-Yield OAS, bp),
            "inflation_expectation": float (e.g., T5YIFR, %),
            "cpi_yoy": float (%),
            "pmi": float (0-100),
            "fed_funds_rate": float (%),
            "gdp_growth": float (%),
        }

    Returns:
        Macro feature dict with state buckets and derived regime.
    """
    yc = indicators.get("yield_curve_spread")
    hy = indicators.get("hy_oas")
    infl = indicators.get("inflation_expectation")
    cpi = indicators.get("cpi_yoy")
    pmi = indicators.get("pmi")
    ffr = indicators.get("fed_funds_rate")
    gdp = indicators.get("gdp_growth")

    # ── Yield Curve State ─────────────────────────────────────────────
    if yc is not None:
        if yc < -0.20:
            curve_state = "inverted"
        elif yc < 0.30:
            curve_state = "flat"
        else:
            curve_state = "steep"
    else:
        curve_state = "unknown"

    # ── Credit Stress Level ───────────────────────────────────────────
    if hy is not None:
        if hy > 600:
            credit_stress = "crisis"
        elif hy > 400:
            credit_stress = "stressed"
        else:
            credit_stress = "normal"
    else:
        credit_stress = "unknown"

    # ── Inflation State ───────────────────────────────────────────────
    infl_val = infl if infl is not None else cpi
    if infl_val is not None:
        if infl_val > 4.0:
            inflation_state = "rising"
        elif infl_val < 2.0:
            inflation_state = "falling"
        else:
            inflation_state = "anchored"
    else:
        inflation_state = "unknown"

    # ── Growth State ──────────────────────────────────────────────────
    growth_indicator = pmi if pmi is not None else (50 + (gdp or 0) * 5)
    if growth_indicator is not None:
        if growth_indicator > 55:
            growth_state = "above_trend"
        elif growth_indicator > 48:
            growth_state = "near_trend"
        else:
            growth_state = "below_trend"
    else:
        growth_state = "unknown"

    # ── Real Policy Stance ────────────────────────────────────────────
    if ffr is not None and infl_val is not None:
        real_rate = ffr - infl_val
        if real_rate < -1.5:
            policy_stance = "deeply_negative"
        elif real_rate < 0:
            policy_stance = "mildly_negative"
        elif real_rate < 1.0:
            policy_stance = "neutral"
        else:
            policy_stance = "restrictive"
    else:
        policy_stance = "unknown"

    # ── Credit Stress Override ────────────────────────────────────────
    credit_stress_override = credit_stress == "crisis"

    # ── Regime 파생 ───────────────────────────────────────────────────
    if credit_stress_override:
        regime = "contraction"
    elif curve_state == "inverted" and growth_state == "below_trend":
        regime = "contraction"
    elif growth_state == "above_trend" and inflation_state == "falling":
        regime = "goldilocks"
    elif growth_state == "above_trend" and inflation_state == "rising":
        regime = "reflation"
    elif growth_state == "below_trend" and inflation_state == "rising":
        regime = "stagflation"
    elif curve_state == "flat" and growth_state == "near_trend":
        regime = "late_cycle"
    else:
        regime = "expansion"

    # ── Tail Risk ─────────────────────────────────────────────────────
    tail_risk = credit_stress_override or (
        curve_state == "inverted" and credit_stress == "stressed"
    )

    return {
        "curve_state": curve_state,
        "credit_stress_level": credit_stress,
        "inflation_state": inflation_state,
        "growth_state": growth_state,
        "real_policy_stance": policy_stance,
        "credit_stress_override": credit_stress_override,
        "macro_regime": regime,
        "tail_risk_warning": tail_risk,
    }


def compute_overlay_guidance(features: dict) -> dict:
    """매크로 레짐 기반 오버레이 가이던스 JSON."""
    regime = features.get("macro_regime", "expansion")
    tail = features.get("tail_risk_warning", False)

    guidance_map = {
        "expansion": {
            "equity_overlay": "cyclicals, growth > value",
            "risk_overlay": "normal net exposure",
        },
        "goldilocks": {
            "equity_overlay": "growth favored, tech overweight",
            "risk_overlay": "can increase net, favorable vol",
        },
        "reflation": {
            "equity_overlay": "value, commodities, financials",
            "risk_overlay": "duration risk — monitor rates",
        },
        "late_cycle": {
            "equity_overlay": "quality + low-vol tilt, reduce beta",
            "risk_overlay": "reduce net exposure 10-20%",
        },
        "stagflation": {
            "equity_overlay": "defensives, energy, real assets only",
            "risk_overlay": "cut gross+net, raise cash 20-40%",
        },
        "contraction": {
            "equity_overlay": "minimal equity. cash, treasuries, gold",
            "risk_overlay": "cut net to near-zero, max defensive",
        },
    }

    g = guidance_map.get(regime, guidance_map["expansion"])
    if tail:
        g["risk_overlay"] = "TAIL RISK — " + g["risk_overlay"]

    return {
        "regime": regime,
        "equity_overlay_guidance": g["equity_overlay"],
        "risk_overlay": g["risk_overlay"],
        "tail_risk_warning": tail,
    }
