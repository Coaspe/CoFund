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


# ── v2 additions ──────────────────────────────────────────────────────────────

def compute_macro_axes(indicators: dict) -> dict:
    """
    5축 분해: growth/inflation/rates/credit/liquidity → {state, score(-3..+3)}.
    입력 결측 지표는 score=0(중립)으로 처리.
    """
    gdp = indicators.get("gdp_growth")
    pmi = indicators.get("pmi")
    cpi = indicators.get("cpi_yoy")
    core_cpi = indicators.get("core_cpi_yoy")
    infl_exp = indicators.get("inflation_expectation")
    ffr = indicators.get("fed_funds_rate")
    yc = indicators.get("yield_curve_spread")
    hy = indicators.get("hy_oas")
    fci = indicators.get("financial_conditions_index")

    # ── Growth ────────────────────────────────────────────────────────
    g_score = 0
    if pmi is not None:
        if pmi > 57: g_score += 3
        elif pmi > 55: g_score += 2
        elif pmi > 50: g_score += 1
        elif pmi < 45: g_score -= 3
        elif pmi < 48: g_score -= 2
        elif pmi < 50: g_score -= 1
    elif gdp is not None:
        if gdp > 4: g_score += 2
        elif gdp > 2: g_score += 1
        elif gdp < 0: g_score -= 3
        elif gdp < 1: g_score -= 1
    g_score = max(-3, min(3, g_score))
    g_state = {3: "hot", 2: "hot", 1: "ok", 0: "ok", -1: "cool", -2: "cool", -3: "recession"}[g_score]

    # ── Inflation ─────────────────────────────────────────────────────
    infl_val = core_cpi or cpi or infl_exp
    i_score = 0
    if infl_val is not None:
        if infl_val > 6:   i_score = 3
        elif infl_val > 4: i_score = 2
        elif infl_val > 2.5: i_score = 1
        elif infl_val > 1.5: i_score = 0
        elif infl_val > 0:   i_score = -1
        else:                i_score = -2
    i_score = max(-3, min(3, i_score))
    i_state = {3: "hot", 2: "hot", 1: "ok", 0: "ok", -1: "cool", -2: "deflation", -3: "deflation"}[i_score]

    # ── Rates ─────────────────────────────────────────────────────────
    r_score = 0
    if ffr is not None:
        if ffr > 5:  r_score -= 2
        elif ffr > 3: r_score -= 1
        elif ffr < 1: r_score += 2
        elif ffr < 2: r_score += 1
    if yc is not None:
        if yc < -0.50: r_score -= 1   # deeply inverted = very restrictive
        elif yc > 2.0: r_score += 1   # steep = accommodative
    r_score = max(-3, min(3, r_score))
    r_state = "rising" if r_score <= -1 else ("falling" if r_score >= 1 else "flat")

    # ── Credit ────────────────────────────────────────────────────────
    c_score = 0
    if hy is not None:
        if hy > 700:   c_score = -3
        elif hy > 500: c_score = -2
        elif hy > 350: c_score = -1
        elif hy < 250: c_score = 1
    if fci is not None:
        if fci > 1.0:   c_score -= 1
        elif fci < -1.0: c_score += 1
    c_score = max(-3, min(3, c_score))
    c_state = {3: "easy", 2: "easy", 1: "easy", 0: "normal", -1: "tight", -2: "stressed", -3: "stressed"}[c_score]

    # ── Liquidity ─────────────────────────────────────────────────────
    l_score = 0
    if ffr is not None:
        if ffr > 5:  l_score -= 2
        elif ffr > 3: l_score -= 1
        elif ffr < 1: l_score += 2
        elif ffr < 2: l_score += 1
    if yc is not None and yc < 0:
        l_score -= 1   # inverted => drained liquidity
    if fci is not None:
        if fci < -0.5: l_score += 1
        elif fci > 0.5: l_score -= 1
    l_score = max(-3, min(3, l_score))
    l_state = "easy" if l_score >= 1 else ("tight" if l_score <= -1 else "neutral")

    return {
        "growth":    {"state": g_state, "score": g_score},
        "inflation": {"state": i_state, "score": i_score},
        "rates":     {"state": r_state, "score": r_score},
        "credit":    {"state": c_state, "score": c_score},
        "liquidity": {"state": l_state, "score": l_score},
    }


def compute_risk_on_off(axes: dict, indicators: dict) -> dict:
    """
    5축 합성 → risk_on|neutral|risk_off (+risk_score -100..+100) + enhanced tail_risk.
    """
    yc  = indicators.get("yield_curve_spread")
    hy  = indicators.get("hy_oas")
    fci = indicators.get("financial_conditions_index")

    g = axes.get("growth",    {}).get("score", 0)
    i = axes.get("inflation", {}).get("score", 0)
    r = axes.get("rates",     {}).get("score", 0)  # positive = falling (good)
    c = axes.get("credit",    {}).get("score", 0)
    l = axes.get("liquidity", {}).get("score", 0)

    # growth(+), credit(+), liquidity(+) → risk-on; inflation(−), rates(−) → risk-off
    raw = g * 20 + (-i * 10) + r * 15 + c * 20 + l * 15
    risk_score = int(max(-100, min(100, round(raw))))

    if risk_score >= 30:  risk_on_off = "risk_on"
    elif risk_score <= -20: risk_on_off = "risk_off"
    else:                 risk_on_off = "neutral"

    # Tail risk: need ≥2 of inverted yc / stressed hy / tight fci
    curve_inv  = yc  is not None and yc  < -0.20
    credit_str = hy  is not None and hy  > 500
    fci_tight  = fci is not None and fci > 0.5
    tail_count = sum([curve_inv, credit_str, fci_tight])
    tail_warning = tail_count >= 2 or (tail_count >= 1 and risk_on_off == "risk_off")
    tail_level   = "high" if tail_count >= 2 else ("medium" if tail_count == 1 else "low")

    return {
        "risk_on_off": risk_on_off,
        "risk_score":  risk_score,
        "tail_risk_warning": tail_warning,
        "tail_risk_level":   tail_level,
        "component_scores":  {"growth": g, "inflation": i, "rates": r, "credit": c, "liquidity": l},
    }
