"""
risk_manager_agent.py — ⑥ Risk Manager Agent
==============================================
최고 리스크 책임자(CRO) 역할: 4명 애널리스트 제안을 5단계 게이트로 필터링.

설계 원칙:
  Python Layer : 포트폴리오 CVaR, HHI, Component VaR, 유동성 등 연산
  LLM   Layer  : 연산 결과 JSON 기반 5-Gate 의사결정 (수익 < 리스크 통제)

의존 패키지:
  pip install numpy pandas langchain-openai langgraph pydantic

실행:
  python risk_manager_agent.py
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import numpy as np

try:
    import pandas as pd
    HAS_PD = True
except ImportError:
    HAS_PD = False

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = object  # type: ignore

try:
    from langchain_core.messages import SystemMessage, HumanMessage
    HAS_LC = True
except ImportError:
    HAS_LC = False

from schemas.common import (
    InvestmentState,
    first_not_none,
    compute_signed_weight,
    compute_disagreement_score,
)
from schemas.taxonomy import map_macro_regime_to_canonical, RISK_OFF_REGIMES
from llm.router import get_llm

warnings.filterwarnings("ignore")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 리스크 한도 상수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LIMITS = {
    "max_leverage": 2.0,
    "max_gross_exposure": 2.0,
    "max_net_exposure": 0.8,
    "max_portfolio_cvar_1d": 0.015,      # 1.5%
    "max_single_name_weight": 0.15,      # 15%
    "max_sector_weight": 0.35,           # 35%
    "max_hhi": 0.25,                     # HHI 임계값
    "max_quant_weight_anomaly": 0.20,    # 퀀트 추천 비중 이상 임계
    "conservative_fallback_weight": 0.03,
    "liquidity_days_warning": 5,         # 청산 소요일 경고
}


def _normalize_ticker_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = value.replace(",", " ").split()
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        ticker = str(item or "").strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        out.append(ticker)
    return out


def _normalize_weight_map(value: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(value, dict):
        return out
    for ticker, weight in value.items():
        try:
            wt = float(weight)
        except (TypeError, ValueError):
            continue
        if np.isnan(wt) or np.isinf(wt):
            continue
        key = str(ticker or "").strip().upper()
        if key:
            out[key] = wt
    total = float(sum(max(0.0, abs(v)) for v in out.values()))
    if total > 1e-12:
        out = {t: float(v) / total for t, v in out.items()}
    return out


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    out = _coerce_float(value)
    return default if out is None else float(out)


def _extract_portfolio_mandate(payload: dict) -> dict:
    orch_mandate = payload.get("portfolio_mandate", {}) or {}
    constraints = orch_mandate.get("constraints", {}) if isinstance(orch_mandate, dict) else {}
    allocator_guidance = payload.get("allocator_guidance", {}) or {}
    if not isinstance(allocator_guidance, dict):
        allocator_guidance = {}
    raw_ctx = payload.get("portfolio_context", {}) or {}
    if not isinstance(raw_ctx, dict):
        raw_ctx = {}

    def _pick_list(*keys: str) -> list[str]:
        for key in keys:
            if key in constraints and constraints.get(key):
                return _normalize_ticker_list(constraints.get(key))
            if key in raw_ctx and raw_ctx.get(key):
                return _normalize_ticker_list(raw_ctx.get(key))
        return []

    def _pick_float(*keys: str) -> Optional[float]:
        for key in keys:
            if key in constraints:
                val = _coerce_float(constraints.get(key))
                if val is not None:
                    return val
            if key in allocator_guidance:
                val = _coerce_float(allocator_guidance.get(key))
                if val is not None:
                    return val
            if key in raw_ctx:
                val = _coerce_float(raw_ctx.get(key))
                if val is not None:
                    return val
        return None

    return {
        "allowed_tickers": _pick_list("allowed_tickers", "allowed_universe"),
        "blocked_tickers": _pick_list("blocked_tickers", "forbidden_tickers"),
        "required_tickers": _pick_list("required_tickers", "must_include"),
        "max_single_name_weight": _pick_float("max_single_name_weight"),
        "target_gross_exposure": _pick_float("target_gross_exposure"),
        "target_net_exposure": _pick_float("target_net_exposure"),
        "max_drawdown_pct": _pick_float("max_drawdown_pct"),
    }


def _event_applies_to_ticker(event: dict, ticker: str) -> bool:
    event_ticker = str(event.get("ticker", "")).strip().upper()
    if event_ticker in {"", "__GLOBAL__"}:
        return True
    affected = {
        str(item).strip().upper()
        for item in (event.get("affected_tickers", []) or [])
        if str(item).strip()
    }
    if ticker in affected:
        return True
    return event_ticker == ticker


def _build_event_risk_summary(event_calendar: list[dict], tickers_in_play: list[str]) -> dict:
    imminent: list[dict] = []
    for raw in event_calendar or []:
        if not isinstance(raw, dict):
            continue
        status = str(raw.get("status", "")).strip().lower()
        if status not in {"imminent", "triggered"}:
            continue
        if tickers_in_play and not any(_event_applies_to_ticker(raw, t) for t in tickers_in_play):
            continue
        imminent.append(raw)

    confirmed = [item for item in imminent if bool(item.get("confirmed"))]
    high_priority = [item for item in imminent if int(item.get("priority", 3) or 3) <= 1]
    nearest_days = None
    for item in imminent:
        days = item.get("days_to_event")
        if isinstance(days, int):
            nearest_days = days if nearest_days is None else min(nearest_days, days)

    severity = "low"
    if len(high_priority) >= 2 or any(str(item.get("status", "")).strip().lower() == "triggered" for item in imminent):
        severity = "critical"
    elif high_priority or len(confirmed) >= 2:
        severity = "high"
    elif imminent:
        severity = "medium"

    return {
        "status": "ok",
        "imminent_count": len(imminent),
        "confirmed_count": len(confirmed),
        "high_priority_count": len(high_priority),
        "nearest_days": nearest_days,
        "severity": severity,
        "events": imminent[:6],
    }


def _ticker_risk_bucket(ticker: str, meta: dict) -> str:
    tt = str(ticker).strip().upper()
    sector = str((meta or {}).get("sector", "")).strip().lower()
    asset_type = str((meta or {}).get("asset_type", "")).strip().lower()
    if tt in {"TLT", "IEF", "SHY", "BND"} or "fixed income" in sector or asset_type in {"bond", "fixed_income"}:
        return "duration"
    if tt in {"GLD", "IAU", "SLV"} or "commodity" in sector or "precious" in sector:
        return "real_asset"
    if tt in {"XLE", "USO", "OIH"} or sector == "energy":
        return "energy"
    if tt in {"QQQ", "XLK"} or sector == "technology":
        return "growth_equity"
    return "broad_equity"


_STRESS_SCENARIO_SHOCKS = {
    "equity_gap_down": {
        "broad_equity": -0.08,
        "growth_equity": -0.10,
        "energy": -0.06,
        "duration": 0.02,
        "real_asset": 0.015,
    },
    "rates_up_100bp": {
        "broad_equity": -0.03,
        "growth_equity": -0.05,
        "energy": -0.01,
        "duration": -0.09,
        "real_asset": -0.015,
    },
    "rates_down_100bp": {
        "broad_equity": 0.02,
        "growth_equity": 0.04,
        "energy": -0.01,
        "duration": 0.07,
        "real_asset": 0.01,
    },
    "oil_spike": {
        "broad_equity": -0.03,
        "growth_equity": -0.04,
        "energy": 0.08,
        "duration": -0.02,
        "real_asset": 0.025,
    },
    "vol_spike": {
        "broad_equity": -0.05,
        "growth_equity": -0.07,
        "energy": -0.04,
        "duration": 0.015,
        "real_asset": 0.02,
    },
}


def _build_stress_test_summary(
    payload: dict,
    tickers_in_play: list[str],
    ticker_weights: dict[str, float],
    lim: dict,
    canonical_regime: str,
    event_risk: dict,
) -> dict:
    positions_meta = payload.get("positions_metadata", {}) or {}
    macro = (payload.get("analyst_reports", {}) or {}).get("macro", {}) or {}
    pricing_divergence = macro.get("pricing_divergence", {}) if isinstance(macro.get("pricing_divergence", {}), dict) else {}
    divergence_signal = str(pricing_divergence.get("overall_signal", "")).strip().lower()

    multiplier = 1.0
    if canonical_regime in RISK_OFF_REGIMES:
        multiplier += 0.2
    if divergence_signal in {"hawkish_surprise_risk", "panic_not_priced", "credit_stress_not_priced"}:
        multiplier += 0.1
    if event_risk.get("severity") == "high":
        multiplier += 0.1
    elif event_risk.get("severity") == "critical":
        multiplier += 0.2

    scenarios: list[dict] = []
    warning_loss = max(float(lim.get("max_portfolio_cvar_1d", 0.015)) * 2.0, 0.03)
    breach_loss = max(float(lim.get("max_portfolio_cvar_1d", 0.015)) * 3.0, 0.05)
    critical_loss = max(float(lim.get("max_portfolio_cvar_1d", 0.015)) * 4.0, 0.08)
    worst_case_loss = 0.0
    worst_scenario = ""

    for name, shock_map in _STRESS_SCENARIO_SHOCKS.items():
        projected_return = 0.0
        ticker_impacts = []
        for ticker in tickers_in_play:
            meta = positions_meta.get(ticker, {}) if isinstance(positions_meta, dict) else {}
            bucket = _ticker_risk_bucket(ticker, meta if isinstance(meta, dict) else {})
            shock = float(shock_map.get(bucket, -0.03)) * multiplier
            weight = float(ticker_weights.get(ticker, 0.0))
            contribution = weight * shock
            projected_return += contribution
            ticker_impacts.append(
                {
                    "ticker": ticker,
                    "bucket": bucket,
                    "shock": round(shock, 4),
                    "weighted_contribution": round(contribution, 4),
                }
            )
        projected_loss = max(0.0, -projected_return)
        severity = "normal"
        if projected_loss >= critical_loss:
            severity = "critical"
        elif projected_loss >= breach_loss:
            severity = "high"
        elif projected_loss >= warning_loss:
            severity = "medium"
        scenarios.append(
            {
                "name": name,
                "projected_return": round(projected_return, 4),
                "projected_loss": round(projected_loss, 4),
                "severity": severity,
                "ticker_impacts": ticker_impacts[:6],
            }
        )
        if projected_loss > worst_case_loss:
            worst_case_loss = projected_loss
            worst_scenario = name

    breached = [item for item in scenarios if item["severity"] in {"medium", "high", "critical"}]
    severity = "low"
    if worst_case_loss >= critical_loss or sum(1 for item in breached if item["severity"] in {"high", "critical"}) >= 2:
        severity = "critical"
    elif worst_case_loss >= breach_loss:
        severity = "high"
    elif breached:
        severity = "medium"

    return {
        "status": "ok",
        "severity": severity,
        "stress_multiplier": round(multiplier, 2),
        "warning_loss_threshold": round(warning_loss, 4),
        "breach_loss_threshold": round(breach_loss, 4),
        "worst_scenario": worst_scenario,
        "worst_case_loss": round(worst_case_loss, 4),
        "breached_scenarios": [item["name"] for item in breached],
        "scenarios": scenarios,
    }


def _build_liquidity_risk_summary(
    payload: dict,
    tickers_in_play: list[str],
    lim: dict,
    canonical_regime: str,
    event_risk: dict,
) -> dict:
    summary = payload.get("portfolio_risk_summary", {}) or {}
    liquidity_days = summary.get("liquidity_score_by_ticker", {}) or {}
    macro = (payload.get("analyst_reports", {}) or {}).get("macro", {}) or {}
    indicators = {}
    if isinstance(macro.get("raw_market_inputs", {}), dict):
        indicators = macro.get("raw_market_inputs", {}) or {}
    vix = _coerce_float(indicators.get("vix_level"))
    if vix is None:
        vix = _coerce_float(indicators.get("vix_index"))

    multiplier = 1.0
    if canonical_regime in RISK_OFF_REGIMES:
        multiplier += 0.4
    if vix is not None and vix >= 30:
        multiplier += 0.5
    elif vix is not None and vix >= 20:
        multiplier += 0.25
    if event_risk.get("severity") == "high":
        multiplier += 0.2
    elif event_risk.get("severity") == "critical":
        multiplier += 0.4
    monitoring_actions = payload.get("monitoring_actions", {}) or {}
    if bool(monitoring_actions.get("risk_refresh_required")):
        multiplier += 0.2
    multiplier = min(multiplier, 2.5)

    stressed_days = {}
    max_base = 0.0
    max_stressed = 0.0
    worst_ticker = ""
    warning_days = float(lim.get("liquidity_days_warning", 5))
    for ticker in tickers_in_play:
        base_days = _safe_float(liquidity_days.get(ticker), 0.0)
        stressed = base_days * multiplier
        stressed_days[ticker] = round(stressed, 2)
        if stressed > max_stressed:
            max_stressed = stressed
            max_base = base_days
            worst_ticker = ticker

    funding_score = 0.0
    leverage = _safe_float(summary.get("leverage_ratio"), 0.0)
    gross = _safe_float(summary.get("total_gross_exposure"), 0.0)
    cvar = _safe_float(summary.get("portfolio_cvar_1d"), 0.0)
    funding_score += min(leverage / max(float(lim.get("max_leverage", 2.0)), 1e-8), 1.5)
    funding_score += min(gross / max(float(lim.get("max_gross_exposure", 2.0)), 1e-8), 1.5)
    funding_score += min(cvar / max(float(lim.get("max_portfolio_cvar_1d", 0.015)), 1e-8), 2.0)
    if event_risk.get("severity") == "high":
        funding_score += 0.5
    elif event_risk.get("severity") == "critical":
        funding_score += 1.0

    severity = "low"
    if max_stressed >= warning_days * 3 or funding_score >= 4.0:
        severity = "critical"
    elif max_stressed >= warning_days * 2 or funding_score >= 3.0:
        severity = "high"
    elif max_stressed >= warning_days or funding_score >= 2.0:
        severity = "medium"

    funding_stress_level = "normal"
    if funding_score >= 4.0:
        funding_stress_level = "critical"
    elif funding_score >= 3.0:
        funding_stress_level = "elevated"
    elif funding_score >= 2.0:
        funding_stress_level = "watch"

    return {
        "status": "ok",
        "severity": severity,
        "stress_multiplier": round(multiplier, 2),
        "warning_days_threshold": round(warning_days, 2),
        "worst_ticker": worst_ticker,
        "max_base_days": round(max_base, 2),
        "max_stressed_days": round(max_stressed, 2),
        "stressed_days_by_ticker": stressed_days,
        "funding_stress_score": round(funding_score, 2),
        "funding_stress_level": funding_stress_level,
    }


def _build_kill_switch(
    summary: dict,
    lim: dict,
    stress_summary: dict,
    liquidity_risk: dict,
    event_risk: dict,
    feedback_reasons: list[str],
) -> dict:
    gross = _safe_float(summary.get("total_gross_exposure"), 0.0)
    net = _safe_float(summary.get("total_net_exposure"), 0.0)
    cvar = _safe_float(summary.get("portfolio_cvar_1d"), 0.0)
    severe_conditions: list[str] = []
    if cvar > float(lim.get("max_portfolio_cvar_1d", 0.015)):
        severe_conditions.append("cvar_breach")
    if str(stress_summary.get("severity", "")).strip().lower() in {"high", "critical"}:
        severe_conditions.append("stress_test_breach")
    if str(liquidity_risk.get("severity", "")).strip().lower() in {"high", "critical"}:
        severe_conditions.append("liquidity_stress")
    if str(event_risk.get("severity", "")).strip().lower() == "critical":
        severe_conditions.append("event_cluster")

    active = False
    severity = "low"
    if "stress_test_breach" in severe_conditions and "liquidity_stress" in severe_conditions:
        active = True
        severity = "critical"
    elif len(severe_conditions) >= 3:
        active = True
        severity = "critical"
    elif len(severe_conditions) == 2:
        active = True
        severity = "high"
    elif severe_conditions:
        severity = "medium"

    target_gross = None
    target_net = None
    if active:
        target_gross = round(min(gross * 0.5, 0.35), 4)
        target_net = round(min(abs(net), 0.2), 4)
    elif severe_conditions:
        target_gross = round(min(gross * 0.8, 0.6), 4)
        target_net = round(min(abs(net), 0.35), 4)

    reason = " / ".join(severe_conditions) if severe_conditions else "no_kill_switch_conditions"
    return {
        "active": active,
        "severity": severity,
        "conditions": severe_conditions,
        "target_gross_exposure": target_gross,
        "target_net_exposure": target_net,
        "freeze_new_risk": active,
        "reason": reason,
        "feedback_reasons_snapshot": list(dict.fromkeys(feedback_reasons))[:8],
    }


def _build_escalation_plan(
    event_risk: dict,
    stress_summary: dict,
    liquidity_risk: dict,
    kill_switch: dict,
    monitoring_actions: dict,
) -> dict:
    breach_types: list[str] = []
    rerun_desks: set[str] = set(monitoring_actions.get("selected_desks", []) or [])
    actions: list[str] = []

    if event_risk.get("imminent_count", 0):
        breach_types.append("event_risk")
        actions.append("refresh_event_risk")
    if str(stress_summary.get("severity", "")).strip().lower() in {"medium", "high", "critical"}:
        breach_types.append("stress_test")
        rerun_desks.update({"macro", "quant"})
        actions.append("rerun_macro_quant")
    if str(liquidity_risk.get("severity", "")).strip().lower() in {"medium", "high", "critical"}:
        breach_types.append("liquidity")
        rerun_desks.add("fundamental")
        actions.append("review_liquidity")
    if bool(kill_switch.get("active")):
        breach_types.append("kill_switch")
        actions.extend(["freeze_new_risk", "cut_gross_exposure"])

    severity = "low"
    if bool(kill_switch.get("active")) and str(kill_switch.get("severity", "")).strip().lower() == "critical":
        severity = "critical"
    elif len(breach_types) >= 3:
        severity = "high"
    elif len(breach_types) >= 1:
        severity = "medium"

    return {
        "status": "ok",
        "severity": severity,
        "breach_types": breach_types,
        "required_actions": actions,
        "rerun_desks": sorted(rerun_desks),
        "freeze_new_risk": bool(kill_switch.get("freeze_new_risk")),
        "monitoring_source": str(monitoring_actions.get("reason", "")).strip() or "risk_engine",
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 1-A.  calculate_portfolio_risk_summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calculate_portfolio_risk_summary(
    positions: Dict[str, dict],
    returns_matrix: Optional[np.ndarray] = None,
    ticker_order: Optional[List[str]] = None,
) -> dict:
    """
    포트폴리오 차원의 노출도, 꼬리 위험, 집중도를 계산합니다.

    Args:
        positions: {
            "AAPL": {
                "weight": 0.10,           # 포트폴리오 대비 비중 (부호 포함, -=Short)
                "sector": "Technology",
                "avg_daily_volume_usd": 5_000_000_000,  # 일평균 거래대금
                "position_notional_usd": 50_000_000,     # 포지션 명목가
            }, ...
        }
        returns_matrix: (T x N) 일별 로그 수익률 행렬 (N=종목 수, ticker_order 순)
        ticker_order:   returns_matrix 열 순서에 대응하는 종목 리스트

    Returns:
        {
            "total_gross_exposure":  float,
            "total_net_exposure":    float,
            "leverage_ratio":        float,
            "portfolio_cvar_1d":     float,    # 99% CVaR (양수, 손실 크기)
            "component_var_by_ticker": {ticker: float, ...},
            "herfindahl_index":      float,    # HHI
            "liquidity_score_by_ticker": {ticker: float, ...},  # 청산 소요일
            "sector_exposure":       {sector: float, ...},
            "error": str | None
        }
    """
    r: dict[str, Any] = {
        "total_gross_exposure": None,
        "total_net_exposure": None,
        "leverage_ratio": None,
        "portfolio_cvar_1d": None,
        "component_var_by_ticker": {},
        "herfindahl_index": None,
        "liquidity_score_by_ticker": {},
        "sector_exposure": {},
        "error": None,
    }
    try:
        tickers = list(positions.keys())
        weights = np.array([positions[t]["weight"] for t in tickers])

        # ── Exposure ──────────────────────────────────────────────────────
        gross = float(np.sum(np.abs(weights)))
        net = float(np.sum(weights))
        r["total_gross_exposure"] = round(gross, 4)
        r["total_net_exposure"] = round(net, 4)
        r["leverage_ratio"] = round(gross, 4)  # 자본 대비 총 노출

        # ── HHI (Herfindahl-Hirschman Index) ──────────────────────────────
        w_abs = np.abs(weights)
        w_norm = w_abs / (w_abs.sum() or 1.0)
        hhi = float(np.sum(w_norm ** 2))
        r["herfindahl_index"] = round(hhi, 4)

        # ── 섹터별 노출 ──────────────────────────────────────────────────
        sectors: dict[str, float] = {}
        for t in tickers:
            s = positions[t].get("sector", "Unknown")
            sectors[s] = sectors.get(s, 0.0) + positions[t]["weight"]
        r["sector_exposure"] = {k: round(v, 4) for k, v in sectors.items()}

        # ── 유동성 점수 (포지션 / 일평균 거래대금 = 청산 소요일) ──────────
        for t in tickers:
            adv = positions[t].get("avg_daily_volume_usd", 1e9)
            notional = positions[t].get("position_notional_usd", 0)
            days = abs(notional) / max(adv, 1.0)
            r["liquidity_score_by_ticker"][t] = round(days, 2)

        # ── 포트폴리오 CVaR & Component VaR ───────────────────────────────
        if returns_matrix is not None and ticker_order is not None:
            # 열 순서 맞추기
            idx_map = {t: i for i, t in enumerate(ticker_order)}
            w_vec = np.array([
                positions[t]["weight"]
                for t in ticker_order if t in positions
            ])

            # 포트폴리오 수익률
            port_ret = returns_matrix @ w_vec
            var_th = np.percentile(port_ret, 1)
            tail = port_ret[port_ret <= var_th]
            cvar = float(abs(np.mean(tail))) if len(tail) > 0 else float(abs(var_th))
            r["portfolio_cvar_1d"] = round(cvar, 6)

            # Component VaR (Marginal VaR × weight 근사)
            cov = np.cov(returns_matrix, rowvar=False)
            port_vol = float(np.sqrt(w_vec @ cov @ w_vec))
            if port_vol > 1e-12:
                marginal = (cov @ w_vec) / port_vol
                for i, t in enumerate(ticker_order):
                    if t in positions:
                        cv = float(w_vec[i] * marginal[i])
                        r["component_var_by_ticker"][t] = round(cv, 6)
        else:
            # returns_matrix 미제공 → 개별 비중 기반 단순 추정
            r["portfolio_cvar_1d"] = round(float(gross * 0.025), 6)
            for t in tickers:
                r["component_var_by_ticker"][t] = round(abs(positions[t]["weight"]) * 0.025, 6)

    except Exception as exc:
        r["error"] = str(exc)
    return r


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 1-B.  aggregate_risk_payload
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def aggregate_risk_payload(
    risk_summary: dict,
    analyst_reports: dict,
    limits: Optional[dict] = None,
) -> dict:
    """
    LLM에게 전달할 마스터 리스크 JSON을 조립합니다.

    Args:
        risk_summary:    calculate_portfolio_risk_summary 결과
        analyst_reports: {
            "macro":       dict,  # Macro Analyst 결과 (regime 등)
            "fundamental": dict,  # Funda Analyst 결과 (risk flags 등)
            "sentiment":   dict,  # Sentiment Analyst 결과
            "quant":       dict,  # Quant Analyst 결과 (추천 비중, z-score 등)
        }
        limits:          리스크 한도 딕셔너리 (None → 기본값)

    Returns:
        LLM에게 주입할 마스터 JSON dict
    """
    lim = limits or LIMITS

    payload = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "risk_limits": lim,
        "portfolio_risk_summary": risk_summary,
        "analyst_reports": analyst_reports,
    }
    return payload


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 2-A.  Pydantic 출력 스키마
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if HAS_PYDANTIC:
    class TickerDecision(BaseModel):
        final_weight: float = Field(ge=-1.0, le=1.0, description="최종 승인 비중 (부호 포함, -=Short)")
        decision: Literal["approve", "reduce", "reject_local"] = Field(
            description="approve=원안 승인, reduce=비중 감축, reject_local=비중 0 (피드백 불필요)"
        )
        flags: List[str] = Field(default_factory=list, description="리스크 플래그 목록")
        rationale_short: str = Field(description="판단 근거 1~2문장")

    class HedgeRecommendation(BaseModel):
        type: Literal["index_hedge", "sector_hedge", "factor_hedge"]
        direction: Literal["short", "long"] = "short"
        notional_suggestion: float = Field(ge=0.0, description="Notional 비율 (0~1)")
        reason: str

    class GrossNetAdjustment(BaseModel):
        target_gross_exposure: float
        target_net_exposure: float
        reason: str

    class PortfolioActions(BaseModel):
        hedge_recommendations: List[HedgeRecommendation] = Field(default_factory=list)
        gross_net_adjustment: Optional[GrossNetAdjustment] = None

    class OrchestratorFeedback(BaseModel):
        required: bool = Field(description="True면 Orchestrator에 피드백 루프 트리거")
        reasons: List[str] = Field(default_factory=list)
        detail: str = ""

    class RiskManagerOutput(BaseModel):
        """⑥ Risk Manager LLM 구조화 출력."""
        per_ticker_decisions: Dict[str, TickerDecision]
        portfolio_actions: PortfolioActions
        orchestrator_feedback: OrchestratorFeedback
else:
    RiskManagerOutput = dict  # type: ignore[assignment,misc]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 2-B.  System Prompt
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RISK_SYSTEM_PROMPT = """\
당신은 헤지펀드의 최고 리스크 책임자(CRO)입니다.
수익 기회보다 리스크 통제가 항상 우선입니다.
입력된 JSON 데이터를 바탕으로 아래 5단계 의사결정 게이트를 순차적으로 평가하십시오.

[Gate 1] 하드 리스크 한도 위반
  - portfolio_cvar_1d > risk_limits.max_portfolio_cvar_1d  → 신규 롱 축소/보류 + 인덱스 헷지 제안.
  - leverage_ratio > risk_limits.max_leverage               → 그로스 이하로 강제 축소.
  - 한도 위반 시 orchestrator_feedback.required = true.

[Gate 2] 집중도 (Sector / Factor)
  - 특정 섹터 노출이 risk_limits.max_sector_weight 초과 시 → 해당 섹터 종목 비중 제한.
  - herfindahl_index > risk_limits.max_hhi 시                → 분산 부족 경고, 비중 제한.
  - component_var_by_ticker 에서 단일 종목이 포트 VaR의 40% 이상이면 → reduce.

[Gate 3] 구조적 리스크 (개별 종목)
  - analyst_reports.fundamental 에 "default_risk", "accounting_fraud", "regulatory_action"
    플래그 중 하나라도 있으면, Quant가 어떤 추천을 하더라도 해당 종목 비중 0
    → decision = "reject_local" (피드백 루프 불필요).

[Gate 4] 레짐/전략 정합성
  - analyst_reports.macro.regime 이 "recession" 또는 "crisis" 일 때
    Quant의 방향이 적극 매수(LONG > 10%) 이면 → 방어적 리밸런싱 제안.
  - orchestrator_feedback.required = true.

[Gate 5] 데이터/모델 이상
  - Quant 추천 비중이 risk_limits.max_quant_weight_anomaly 초과 시
    → risk_limits.conservative_fallback_weight 로 제한.
  - orchestrator_feedback.required = true.

※ 위 게이트를 순서대로 적용하되, 각 게이트에서 비중이 축소/거부된 종목은
  이후 게이트에서 축소된 비중 기준으로 재평가하라.
※ 모든 게이트를 통과한 종목만 decision="approve"로 최종 확정하라.
※ 어떤 게이트에서든 orchestrator_feedback.required가 true가 되면
  reasons 리스트에 해당 사유를 추가하라.

출력은 반드시 아래 JSON 스키마를 따르라:
{
  "per_ticker_decisions": {
    "<TICKER>": {
      "final_weight": <float>,
      "decision": "approve | reduce | reject_local",
      "flags": ["..."],
      "rationale_short": "..."
    }
  },
  "portfolio_actions": {
    "hedge_recommendations": [ { "type": "...", "direction": "short", "notional_suggestion": <float>, "reason": "..." } ],
    "gross_net_adjustment": { "target_gross_exposure": <float>, "target_net_exposure": <float>, "reason": "..." }
  },
  "orchestrator_feedback": {
    "required": <bool>,
    "reasons": ["..."],
    "detail": "..."
  }
}"""


def _build_risk_human_msg(payload: dict) -> str:
    return (
        f"아래는 {payload.get('timestamp', '')} 기준 포트폴리오 리스크 평가 JSON입니다.\n"
        "5단계 Gate를 순서대로 평가하여 최종 리스크 결정을 내려주세요.\n\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 2-C.  LLM 호출 (실제 / Mock)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _call_llm(payload: dict) -> dict:
    """
    Risk decision pipeline:
      1. compute_risk_decision() → Python deterministic (source of truth)
      2. (Optional) LLM enriches rationale narrative ONLY
      3. LLM may NEVER override final_weight/decision/flags
    """
    # Step 1: Python deterministic decision (ALWAYS runs)
    decision = compute_risk_decision(payload)

    decision["_llm_enrichment_status"] = "skipped_no_llm"
    decision["_llm_enrichment_error"] = ""

    # Step 2: Optional LLM narrative enrichment
    llm = get_llm("risk_manager")
    if llm is not None and HAS_LC:
        try:
            decision = _enrich_risk_narrative_with_llm(llm, payload, decision)
            decision["_llm_enrichment_status"] = "ok_full"
        except Exception as exc:
            if _is_prompt_size_error(exc):
                try:
                    compact_payload = _compact_payload_for_enrichment(payload)
                    decision = _enrich_risk_narrative_with_llm(llm, compact_payload, decision)
                    decision["_llm_enrichment_status"] = "ok_compact_retry"
                    decision["_llm_enrichment_error"] = str(exc)[:220]
                    print("   [LLM] ⚠️ Narrative payload too large → compact retry succeeded")
                except Exception as retry_exc:
                    decision["_llm_enrichment_status"] = "failed_compact_retry"
                    decision["_llm_enrichment_error"] = str(retry_exc)[:220]
                    print(f"   [LLM] ⚠️ Narrative enrichment failed after compact retry: {retry_exc}")
            else:
                decision["_llm_enrichment_status"] = "failed_full"
                decision["_llm_enrichment_error"] = str(exc)[:220]
                print(f"   [LLM] ⚠️ Narrative enrichment failed (decision unchanged): {exc}")
    else:
        print("   [LLM] API 키 없음 → Python 규칙 기반 결정 (narrative 생략)")

    return decision


def _is_prompt_size_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "413" in msg
        or "request too large" in msg
        or "tokens per minute" in msg
        or "tpm" in msg
        or "max token" in msg
    )


def _compact_payload_for_enrichment(payload: dict) -> dict:
    reports = payload.get("analyst_reports", {}) or {}
    summary = payload.get("portfolio_risk_summary", {}) or {}
    compact_reports = {}
    for desk in ("macro", "fundamental", "sentiment", "quant"):
        src = reports.get(desk, {}) or {}
        compact_reports[desk] = {
            "primary_decision": src.get("primary_decision"),
            "confidence": src.get("confidence"),
            "summary": str(src.get("summary", ""))[:240],
            "key_drivers": list(src.get("key_drivers", []) or [])[:3],
            "risk_flags": list(src.get("risk_flags", []) or [])[:3],
        }
    return {
        "timestamp": payload.get("timestamp"),
        "risk_limits": payload.get("risk_limits", {}),
        "portfolio_risk_summary": {
            "total_gross_exposure": summary.get("total_gross_exposure"),
            "total_net_exposure": summary.get("total_net_exposure"),
            "portfolio_cvar_1d": summary.get("portfolio_cvar_1d"),
            "herfindahl_index": summary.get("herfindahl_index"),
            "sector_exposure": summary.get("sector_exposure", {}),
            "component_var_by_ticker": summary.get("component_var_by_ticker", {}),
        },
        "analyst_reports": compact_reports,
    }


def _enrich_risk_narrative_with_llm(llm, payload: dict, decision: dict) -> dict:
    """
    LLM enriches rationale_short and feedback.detail text ONLY.
    final_weight, decision, flags are IMMUTABLE from Python engine.
    """
    compact_decision = {
        "per_ticker_decisions": {
            t: {
                "final_weight": d.get("final_weight"),
                "decision": d.get("decision"),
                "flags": d.get("flags", []),
                "rationale_short": d.get("rationale_short", ""),
            }
            for t, d in (decision.get("per_ticker_decisions", {}) or {}).items()
        },
        "orchestrator_feedback": decision.get("orchestrator_feedback", {}),
    }
    prompt = (
        "Below is a risk payload and a Python-computed decision. "
        "Your job: generate ONLY a 1-2 sentence Korean rationale for each ticker decision, "
        "and a concise detail paragraph for orchestrator_feedback. "
        "DO NOT change final_weight, decision, or flags.\n\n"
        f"Payload:\n{json.dumps(payload, ensure_ascii=False, indent=1, default=str)}\n\n"
        f"Decision:\n{json.dumps(compact_decision, ensure_ascii=False, indent=1, default=str)}\n\n"
        "Return JSON with: {per_ticker_rationales: {TICKER: str}, feedback_detail: str}"
    )
    msgs = [HumanMessage(content=prompt)]
    raw = llm.invoke(msgs)
    try:
        enrichment = json.loads(raw.content)
    except (json.JSONDecodeError, AttributeError):
        return decision

    # Apply narrative ONLY (never override structure)
    for t, rationale in enrichment.get("per_ticker_rationales", {}).items():
        if t in decision.get("per_ticker_decisions", {}):
            decision["per_ticker_decisions"][t]["rationale_short"] = rationale
    if enrichment.get("feedback_detail"):
        decision["orchestrator_feedback"]["detail"] = enrichment["feedback_detail"]

    return decision



def compute_risk_decision(payload: dict) -> dict:
    """
    5-Gate policy engine — PURE PYTHON, deterministic.

    This is the source of truth for final_weight/decision/flags.
    LLM may NEVER override these values.

    Fixed bugs:
      - 0.0 weight: uses first_not_none (0.0 is valid, not falsy)
      - Signed weight: SHORT produces negative weight
      - Gate4: only triggers for active LONG, not SHORT (which is a hedge)
      - Regime taxonomy: uses canonical mapping
      - R6 Disagreement: auto weight reduction on high disagreement
      - R2 No Evidence: flags and weight cap on missing evidence
    """
    lim = payload.get("risk_limits", LIMITS)
    summary = payload.get("portfolio_risk_summary", {})
    reports = payload.get("analyst_reports", {})
    quant = reports.get("quant", {})
    macro = reports.get("macro", {})
    funda = reports.get("fundamental", {})
    senti = reports.get("sentiment", {})
    monitoring_actions = payload.get("monitoring_actions", {}) or {}
    event_calendar = payload.get("event_calendar", []) or []

    per_ticker: dict[str, dict] = {}
    feedback_required = False
    feedback_reasons: list[str] = []
    feedback_detail_parts: list[str] = []
    hedges: list[dict] = []
    gna: Optional[dict] = None

    # ── Bug fix #1: first_not_none prevents 0.0 → fallback ────────────────
    quant_decision = quant.get("decision", "HOLD")
    raw_alloc = first_not_none(quant, ["final_allocation_pct", "final_weight"], default=0.0)

    # ── Bug fix #2: signed weight ─────────────────────────────────────────
    signed_weight = compute_signed_weight(quant_decision, raw_alloc)

    # Optional: orchestrator/allocator proposed weights (B mode)
    raw_proposed = payload.get("positions_proposed", {})
    proposed_weights: dict[str, float] = {}
    if isinstance(raw_proposed, dict):
        for t, w in raw_proposed.items():
            try:
                wt = float(w)
            except (TypeError, ValueError):
                continue
            if np.isnan(wt) or np.isinf(wt):
                continue
            proposed_weights[str(t).strip().upper()] = wt

    tickers_in_play = list(summary.get("component_var_by_ticker", {}).keys())
    target = str(reports.get("_target_ticker", tickers_in_play[0] if tickers_in_play else "AAPL")).strip().upper()
    if proposed_weights:
        tickers_in_play = list(proposed_weights.keys())
        if target not in tickers_in_play:
            target = tickers_in_play[0]
        ticker_weights = {t: float(proposed_weights.get(t, 0.0)) for t in tickers_in_play}
    else:
        if not tickers_in_play:
            tickers_in_play = [target]
        ticker_weights: dict[str, float] = {}
        for t in tickers_in_play:
            w = signed_weight if t == target else summary.get("component_var_by_ticker", {}).get(t, 0.0)
            ticker_weights[t] = w

    ticker_flags: dict[str, list[str]] = {t: [] for t in tickers_in_play}
    ticker_decisions: dict[str, str] = {t: "approve" for t in tickers_in_play}

    # ── R9: No Evidence enforcement ───────────────────────────────────────
    for desk_name, desk_out in [("macro", macro), ("fundamental", funda),
                                 ("sentiment", senti), ("quant", quant)]:
        if not desk_out:
            continue
        ev = desk_out.get("evidence", [])
        dok = desk_out.get("data_ok", True)
        if not ev or not dok:
            flag_code = f"no_evidence_{desk_name}" if not ev else f"data_not_ok_{desk_name}"
            ticker_flags.setdefault(target, []).append(flag_code)
            feedback_detail_parts.append(f"{desk_name}: evidence/data_ok 부족 → 보수적 처리.")

    # ── R6: Disagreement score ────────────────────────────────────────────
    disagreement = compute_disagreement_score({
        "macro": macro, "fundamental": funda,
        "sentiment": senti, "quant": quant,
    })
    if disagreement > 0.5:
        feedback_required = True
        feedback_reasons.append("model_disagreement")
        feedback_detail_parts.append(
            f"Disagreement score {disagreement:.3f} > 0.5 → 비중 50% 자동 감축."
        )
        for t in tickers_in_play:
            ticker_weights[t] = round(ticker_weights[t] * 0.5, 4)
            ticker_flags.setdefault(t, []).append("disagreement_reduction")

    raw_regime = macro.get("macro_regime", macro.get("regime", "normal"))
    canonical_regime = map_macro_regime_to_canonical(raw_regime)
    event_risk = _build_event_risk_summary(event_calendar, tickers_in_play)
    stress_summary = _build_stress_test_summary(
        payload,
        tickers_in_play,
        ticker_weights,
        lim,
        canonical_regime=canonical_regime,
        event_risk=event_risk,
    )
    liquidity_risk = _build_liquidity_risk_summary(
        payload,
        tickers_in_play,
        lim,
        canonical_regime=canonical_regime,
        event_risk=event_risk,
    )

    # ── Gate 1: 하드 리스크 한도 ──────────────────────────────────────────
    cvar = first_not_none(summary, ["portfolio_cvar_1d"], default=0.0)
    leverage = first_not_none(summary, ["leverage_ratio"], default=0.0)

    if cvar > lim["max_portfolio_cvar_1d"]:
        feedback_required = True
        feedback_reasons.append("portfolio_risk_violation")
        feedback_detail_parts.append(
            f"포트폴리오 CVaR {cvar:.4f} > 한도 {lim['max_portfolio_cvar_1d']}."
        )
        hedges.append({
            "type": "index_hedge",
            "direction": "short",
            "notional_suggestion": round(min(cvar / lim["max_portfolio_cvar_1d"] * 0.1, 0.2), 4),
            "reason": f"CVaR 한도 초과 방어 (현재 {cvar:.4f} vs 한도 {lim['max_portfolio_cvar_1d']})",
        })
        for t in tickers_in_play:
            if ticker_weights[t] > 0:
                scale = lim["max_portfolio_cvar_1d"] / max(cvar, 1e-8)
                ticker_weights[t] = round(ticker_weights[t] * scale, 4)
                ticker_flags[t].append("cvar_limit_breach")
                ticker_decisions[t] = "reduce"

    if leverage > lim["max_leverage"]:
        feedback_required = True
        feedback_reasons.append("leverage_violation")
        feedback_detail_parts.append(f"레버리지 {leverage:.2f} > 한도 {lim['max_leverage']}.")
        scale = lim["max_leverage"] / max(leverage, 1e-8)
        for t in tickers_in_play:
            ticker_weights[t] = round(ticker_weights[t] * scale, 4)
            ticker_flags[t].append("leverage_breach")
            ticker_decisions[t] = "reduce"

    # ── Gate 2: 집중도 ────────────────────────────────────────────────────
    hhi = first_not_none(summary, ["herfindahl_index"], default=0.0)
    if hhi > lim["max_hhi"]:
        for t in tickers_in_play:
            ticker_flags[t].append("concentration_hhi")
        feedback_detail_parts.append(f"HHI {hhi:.4f} > 한도 {lim['max_hhi']}.")

    sector_exp = summary.get("sector_exposure", {})
    for sec, exp in sector_exp.items():
        if abs(exp) > lim["max_sector_weight"]:
            for t in tickers_in_play:
                ticker_flags[t].append(f"sector_overweight_{sec}")
            feedback_detail_parts.append(f"섹터 '{sec}' 노출 {exp:.2f} > 한도 {lim['max_sector_weight']}.")

    comp_var = summary.get("component_var_by_ticker", {})
    total_comp = sum(abs(v) for v in comp_var.values()) or 1.0
    for t, cv in comp_var.items():
        if abs(cv) / total_comp > 0.40:
            ticker_flags.setdefault(t, []).append("component_var_dominant")
            ticker_decisions[t] = "reduce"
            ticker_weights[t] = round(ticker_weights.get(t, 0.0) * 0.7, 4)

    # ── Gate 3: 구조적 리스크 (Funda 플래그) ──────────────────────────────
    funda_flags_raw = funda.get("risk_flags", [])
    funda_flags = [
        (f["code"] if isinstance(f, dict) else f) for f in funda_flags_raw
    ]
    structural = {"default_risk", "accounting_fraud", "regulatory_action",
                  "going_concern", "material_weakness", "restatement"}
    found_structural = [f for f in funda_flags if f in structural]
    if found_structural:
        ticker_weights[target] = 0.0
        ticker_decisions[target] = "reject_local"
        ticker_flags.setdefault(target, []).extend(found_structural)

    # ── Gate 4: 레짐/전략 정합성 (uses CANONICAL taxonomy) ────────────────
    # BUG FIX: Gate4 only triggers for active LONG, not SHORT.
    # SHORT during risk-off is a HEDGE, not a mismatch.
    is_active_long = signed_weight > 0.10
    if canonical_regime in RISK_OFF_REGIMES and is_active_long:
        feedback_required = True
        feedback_reasons.append("strategy_regime_mismatch")
        feedback_detail_parts.append(
            f"Macro 레짐 '{canonical_regime}'(raw='{raw_regime}') vs Quant 적극 매수 "
            f"({signed_weight:.0%}). 방어적 리밸런싱 필요."
        )
        for t in tickers_in_play:
            if ticker_decisions[t] != "reject_local":
                ticker_weights[t] = round(ticker_weights[t] * 0.5, 4)
                ticker_decisions[t] = "reduce"
                ticker_flags[t].append("macro_headwind")

        net_exp = first_not_none(summary, ["total_net_exposure"], default=0.5)
        gna = {
            "target_gross_exposure": round(min(leverage, lim["max_leverage"]) * 0.8, 2),
            "target_net_exposure": round(min(net_exp, 0.3), 2),
            "reason": f"Macro '{canonical_regime}' 레짐에 따른 Net Exposure 축소",
        }

    # ── Gate 4.5: Stress / Liquidity / Event clustering ──────────────────
    stress_severity = str(stress_summary.get("severity", "")).strip().lower()
    if stress_severity in {"high", "critical"}:
        feedback_required = True
        feedback_reasons.append("stress_test_breach")
        feedback_detail_parts.append(
            f"Stress worst-case {stress_summary.get('worst_scenario')} 손실 "
            f"{_safe_float(stress_summary.get('worst_case_loss')):.2%}."
        )
        stress_scale = 0.6 if stress_severity == "critical" else 0.8
        for t in tickers_in_play:
            if ticker_decisions.get(t) == "reject_local":
                continue
            ticker_weights[t] = round(float(ticker_weights.get(t, 0.0)) * stress_scale, 4)
            ticker_decisions[t] = "reduce"
            ticker_flags.setdefault(t, []).append("stress_test_breach")

    liquidity_severity = str(liquidity_risk.get("severity", "")).strip().lower()
    if liquidity_severity in {"high", "critical"}:
        feedback_required = True
        feedback_reasons.append("liquidity_stress")
        feedback_detail_parts.append(
            f"Stressed liquidation days max {liquidity_risk.get('max_stressed_days')} "
            f"(ticker={liquidity_risk.get('worst_ticker')})."
        )
        stressed_days = liquidity_risk.get("stressed_days_by_ticker", {}) or {}
        warning_days = _safe_float(
            liquidity_risk.get("warning_days_threshold"),
            float(lim.get("liquidity_days_warning", 5)),
        )
        liq_scale = 0.5 if liquidity_severity == "critical" else 0.7
        for t in tickers_in_play:
            if ticker_decisions.get(t) == "reject_local":
                continue
            if _safe_float(stressed_days.get(t), 0.0) < warning_days:
                continue
            ticker_weights[t] = round(float(ticker_weights.get(t, 0.0)) * liq_scale, 4)
            ticker_decisions[t] = "reduce"
            ticker_flags.setdefault(t, []).append("liquidity_stress")

    # ── Gate 5: 데이터/모델 이상 ──────────────────────────────────────────
    abs_weight = abs(first_not_none({target: ticker_weights.get(target, 0.0)}, [target], 0.0))
    if abs(raw_alloc or 0) > lim["max_quant_weight_anomaly"]:
        feedback_required = True
        feedback_reasons.append("quant_weight_anomaly")
        feedback_detail_parts.append(
            f"Quant 추천 비중 {raw_alloc:.0%} > 이상 임계 {lim['max_quant_weight_anomaly']:.0%}. "
            f"보수적 기본 비중 {lim['conservative_fallback_weight']:.0%}으로 제한."
        )
        sign = -1.0 if signed_weight < 0 else 1.0
        ticker_weights[target] = sign * lim["conservative_fallback_weight"]
        ticker_decisions[target] = "reduce" if ticker_decisions[target] != "reject_local" else "reject_local"
        ticker_flags.setdefault(target, []).append("quant_anomaly")

    # ── Portfolio mandate enforcement (hard constraints from PM/CIO) ──────
    mandate = _extract_portfolio_mandate(payload)
    mandate_breached = False

    allowed = set(mandate.get("allowed_tickers", []) or [])
    blocked = set(mandate.get("blocked_tickers", []) or [])
    required = set(mandate.get("required_tickers", []) or [])

    if required:
        missing_required = [t for t in required if t not in tickers_in_play]
        if missing_required:
            mandate_breached = True
            feedback_required = True
            feedback_reasons.append("required_ticker_missing")
            feedback_detail_parts.append(
                f"필수 티커 누락: {missing_required}. 포트폴리오 mandate 재계획 필요."
            )

    for t in tickers_in_play:
        disallowed = (allowed and t not in allowed) or (t in blocked)
        if not disallowed:
            continue
        mandate_breached = True
        ticker_weights[t] = 0.0
        ticker_decisions[t] = "reject_local"
        code = "blocked_ticker" if t in blocked else "outside_allowed_universe"
        ticker_flags.setdefault(t, []).append(code)
        feedback_required = True
        feedback_reasons.append("mandate_violation")
        feedback_detail_parts.append(f"{t}: portfolio mandate 위반({code})으로 0 비중 처리.")

    single_cap = mandate.get("max_single_name_weight")
    if single_cap is not None and single_cap >= 0:
        for t in tickers_in_play:
            cur = float(ticker_weights.get(t, 0.0))
            if abs(cur) <= float(single_cap) + 1e-12:
                continue
            mandate_breached = True
            sign = -1.0 if cur < 0 else 1.0
            ticker_weights[t] = round(sign * float(single_cap), 4)
            ticker_decisions[t] = "reduce" if ticker_decisions[t] != "reject_local" else "reject_local"
            ticker_flags.setdefault(t, []).append("max_single_name_weight_breach")
            feedback_required = True
            feedback_reasons.append("mandate_violation")
            feedback_detail_parts.append(
                f"{t}: 단일 비중 {abs(cur):.2%} > mandate 한도 {float(single_cap):.2%}."
            )

    def _scale_weights(scale: float, flag: str, reason: str) -> None:
        nonlocal mandate_breached, feedback_required, gna
        if scale >= 1.0:
            return
        mandate_breached = True
        feedback_required = True
        feedback_reasons.append("mandate_violation")
        feedback_detail_parts.append(reason)
        for t in tickers_in_play:
            if ticker_decisions.get(t) == "reject_local":
                continue
            ticker_weights[t] = round(float(ticker_weights.get(t, 0.0)) * scale, 4)
            ticker_decisions[t] = "reduce"
            ticker_flags.setdefault(t, []).append(flag)
        gross_after = sum(abs(float(ticker_weights.get(t, 0.0))) for t in tickers_in_play)
        net_after = sum(float(ticker_weights.get(t, 0.0)) for t in tickers_in_play)
        gna = {
            "target_gross_exposure": round(gross_after, 4),
            "target_net_exposure": round(net_after, 4),
            "reason": reason,
        }

    target_gross = mandate.get("target_gross_exposure")
    if target_gross is not None and target_gross >= 0:
        current_gross = sum(abs(float(ticker_weights.get(t, 0.0))) for t in tickers_in_play)
        if current_gross > float(target_gross) + 1e-12:
            _scale_weights(
                float(target_gross) / max(current_gross, 1e-8),
                "target_gross_exposure_cap",
                f"총 그로스 노출 {current_gross:.2%} > mandate 한도 {float(target_gross):.2%}.",
            )

    target_net = mandate.get("target_net_exposure")
    if target_net is not None and target_net >= 0:
        current_net = sum(float(ticker_weights.get(t, 0.0)) for t in tickers_in_play)
        if abs(current_net) > float(target_net) + 1e-12:
            _scale_weights(
                float(target_net) / max(abs(current_net), 1e-8),
                "target_net_exposure_cap",
                f"순 노출 {abs(current_net):.2%} > mandate 한도 {float(target_net):.2%}.",
            )

    kill_switch = _build_kill_switch(
        summary,
        lim,
        stress_summary,
        liquidity_risk,
        event_risk,
        feedback_reasons,
    )
    if kill_switch.get("active"):
        feedback_required = True
        feedback_reasons.append("kill_switch_active")
        feedback_detail_parts.append(f"Kill switch 활성화: {kill_switch.get('reason')}.")
        current_gross = sum(abs(float(ticker_weights.get(t, 0.0))) for t in tickers_in_play)
        target_gross = _coerce_float(kill_switch.get("target_gross_exposure"))
        if target_gross is not None and current_gross > target_gross + 1e-12:
            scale = target_gross / max(current_gross, 1e-8)
            for t in tickers_in_play:
                if ticker_decisions.get(t) == "reject_local":
                    continue
                ticker_weights[t] = round(float(ticker_weights.get(t, 0.0)) * scale, 4)
                ticker_decisions[t] = "reduce"
                ticker_flags.setdefault(t, []).append("kill_switch_active")
            gna = {
                "target_gross_exposure": round(target_gross, 4),
                "target_net_exposure": round(_safe_float(kill_switch.get("target_net_exposure"), 0.0), 4),
                "reason": f"Kill switch: {kill_switch.get('reason')}",
            }
    elif str(kill_switch.get("severity", "")).strip().lower() == "medium":
        feedback_required = True
        feedback_reasons.append("kill_switch_watch")

    escalation = _build_escalation_plan(
        event_risk,
        stress_summary,
        liquidity_risk,
        kill_switch,
        monitoring_actions if isinstance(monitoring_actions, dict) else {},
    )
    if escalation.get("severity") in {"high", "critical"}:
        feedback_required = True
    if escalation.get("freeze_new_risk"):
        feedback_reasons.append("freeze_new_risk")

    # ── 최종 조립 ─────────────────────────────────────────────────────────
    per_ticker = {}
    for t in tickers_in_play:
        per_ticker[t] = {
            "final_weight": ticker_weights.get(t, 0.0),  # signed! (negative = short)
            "decision": ticker_decisions.get(t, "approve"),
            "flags": ticker_flags.get(t, []),
            "rationale_short": _rationale(ticker_decisions.get(t, "approve"), ticker_flags.get(t, [])),
        }

    return {
        "per_ticker_decisions": per_ticker,
        "portfolio_actions": {
            "hedge_recommendations": hedges,
            "gross_net_adjustment": gna,
            "kill_switch": kill_switch,
            "escalation": escalation,
        },
        "orchestrator_feedback": {
            "required": feedback_required,
            "reasons": list(dict.fromkeys(feedback_reasons)),
            "detail": " ".join(feedback_detail_parts) if feedback_detail_parts else "모든 Gate 통과.",
        },
        "event_risk": event_risk,
        "stress_test_summary": stress_summary,
        "liquidity_risk": liquidity_risk,
        "_disagreement_score": disagreement,
        "_canonical_regime": canonical_regime,
    }


# Legacy alias for backward compat
_mock_risk_decision = compute_risk_decision


def _rationale(decision: str, flags: list) -> str:
    if decision == "reject_local":
        return f"구조적 리스크 플래그({', '.join(flags)})로 인해 비중 0 처리."
    if decision == "reduce":
        return f"리스크 플래그({', '.join(flags)})로 인해 비중 감축."
    return "모든 리스크 게이트 통과 — 원안 승인."


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 2-D.  LangGraph 노드 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def risk_manager_node(state: InvestmentState) -> dict:
    """
    ⑥ Risk Manager LangGraph 노드.

    실행 흐름:
      1. InvestmentState 에서 4명 애널리스트 결과를 읽는다.
      2. calculate_portfolio_risk_summary 로 포트폴리오 리스크를 계산한다.
      3. aggregate_risk_payload 로 마스터 JSON을 조립한다.
      4. LLM에 5-Gate 프롬프트 + 페이로드를 주입하여 RiskManagerOutput을 얻는다.
      5. risk_assessment 필드에 결과를 저장한다.

    Returns:
        {"risk_assessment": dict}
    """
    iteration = state.get("iteration_count", 1)
    ticker = str(state.get("target_ticker", "AAPL")).strip().upper()

    print(f"\n{'=' * 60}")
    print(f"⑥ RISK MANAGER  (iteration #{iteration})")
    print(f"{'=' * 60}")
    print("   [입력] 4개 애널리스트 리포트 취합 중...")

    macro = state.get("macro_analysis", {})
    funda = state.get("fundamental_analysis", {})
    senti = state.get("sentiment_analysis", {})
    quant = state.get("technical_analysis", {})

    # ── Bug fix #1+2: first_not_none + signed weight ──────────────────────
    quant_decision = quant.get("decision", "HOLD")
    raw_alloc = first_not_none(quant, ["final_allocation_pct", "final_weight"], default=0.0)
    signed_weight = compute_signed_weight(quant_decision, raw_alloc)

    raw_proposed = state.get("positions_proposed", {})
    positions_proposed = _normalize_weight_map(raw_proposed)
    frontdoor = state.get("question_understanding", {}) if isinstance(state.get("question_understanding"), dict) else {}
    frontdoor_intent = str(frontdoor.get("intent", "")).strip().lower()
    normalized_snapshot = state.get("normalized_portfolio_snapshot", {}) if isinstance(state.get("normalized_portfolio_snapshot"), dict) else {}
    snapshot_weights = _normalize_weight_map(normalized_snapshot.get("weights", {}))
    if not positions_proposed and frontdoor_intent == "position_review" and snapshot_weights:
        positions_proposed = snapshot_weights

    def _sector_for_ticker(t: str) -> str:
        tt = str(t).strip().upper()
        if tt == str(ticker).strip().upper():
            return str(funda.get("sector", "Unknown"))
        if tt in {"GLD", "SLV"}:
            return "Commodities"
        if tt in {"TLT", "IEF", "SHY", "BND"}:
            return "Fixed Income"
        if tt in {"XLE"}:
            return "Energy"
        if tt in {"QQQ", "XLK"}:
            return "Technology"
        if tt in {"SPY", "DIA", "IWM"}:
            return "Broad Market"
        return "Unknown"

    def _adv_for_ticker(t: str) -> float:
        tt = str(t).strip().upper()
        if tt in {"SPY", "QQQ", "GLD", "TLT", "XLE", "IWM", "DIA"}:
            return 8_000_000_000.0
        return 2_000_000_000.0

    if positions_proposed:
        positions = {
            t: {
                "weight": float(w),
                "sector": _sector_for_ticker(t),
                "avg_daily_volume_usd": _adv_for_ticker(t),
                "position_notional_usd": abs(float(w)) * 50_000_000,
            }
            for t, w in positions_proposed.items()
        }
    else:
        positions = {
            ticker: {
                "weight": signed_weight,
                "sector": funda.get("sector", "Technology"),
                "avg_daily_volume_usd": funda.get("avg_daily_volume_usd", 5_000_000_000),
                "position_notional_usd": funda.get("position_notional_usd", 50_000_000),
            }
        }

    # ── Check for stale desks ─────────────────────────────────────────────
    stale_desks = []
    for desk_name, desk_key in [("macro", "macro_analysis"), ("fundamental", "fundamental_analysis"),
                                 ("sentiment", "sentiment_analysis"), ("quant", "technical_analysis")]:
        desk_out = state.get(desk_key, {})
        if desk_out.get("status") == "skipped":
            stale_desks.append(desk_name)
        elif desk_out.get("iteration_generated") is not None and desk_out.get("iteration_generated") != iteration:
            stale_desks.append(desk_name)

    print("   [도구 호출] calculate_portfolio_risk_summary...")
    risk_summary = calculate_portfolio_risk_summary(positions)

    analyst_reports = {
        "macro": macro,
        "fundamental": funda,
        "sentiment": senti,
        "quant": quant,
        "_target_ticker": ticker,
    }

    print("   [도구 호출] aggregate_risk_payload...")
    payload = aggregate_risk_payload(risk_summary, analyst_reports)
    if positions_proposed:
        payload["positions_proposed"] = positions_proposed
    payload["portfolio_context"] = state.get("portfolio_context", {}) or {}
    payload["portfolio_mandate"] = (
        ((state.get("orchestrator_directives", {}) or {}).get("portfolio_mandate", {}))
        if isinstance(state.get("orchestrator_directives", {}), dict)
        else {}
    )
    payload["allocator_guidance"] = (
        ((state.get("orchestrator_directives", {}) or {}).get("allocator_guidance", {}))
        if isinstance(state.get("orchestrator_directives", {}), dict)
        else {}
    )
    payload["positions_metadata"] = positions
    payload["event_calendar"] = state.get("event_calendar", []) or []
    payload["monitoring_actions"] = state.get("monitoring_actions", {}) or {}
    payload["decision_quality_scorecard"] = state.get("decision_quality_scorecard", {}) or {}
    payload["portfolio_construction_analysis"] = state.get("portfolio_construction_analysis", {}) or {}

    print("   [LLM] 5-Gate CRO 의사결정 요청 중...")
    decision = _call_llm(payload)

    monitoring_actions = state.get("monitoring_actions", {}) if isinstance(state.get("monitoring_actions"), dict) else {}
    if monitoring_actions.get("risk_refresh_required"):
        portfolio_actions = decision.setdefault("portfolio_actions", {})
        portfolio_actions["monitoring_risk_refresh"] = True
        for t_data in (decision.get("per_ticker_decisions", {}) or {}).values():
            t_data.setdefault("flags", []).append("monitoring_risk_refresh")

    # Add stale desk flags after decision
    if stale_desks:
        for desk in stale_desks:
            flag = f"stale_{desk}"
            for t_data in decision.get("per_ticker_decisions", {}).values():
                t_data.setdefault("flags", []).append(flag)
        decision.setdefault("orchestrator_feedback", {})["required"] = True
        decision["orchestrator_feedback"].setdefault("reasons", []).append("stale_desk_data")

    if positions_proposed:
        per = decision.get("per_ticker_decisions", {}) or {}
        for t, proposed_w in positions_proposed.items():
            if t not in per:
                per[t] = {
                    "final_weight": 0.0,
                    "decision": "approve",
                    "flags": [],
                    "rationale_short": "초기 제안 누락으로 보수적으로 0 비중 처리.",
                }
            if abs(float(proposed_w)) <= 1e-12:
                per[t].setdefault("flags", []).append("proposed_zero_weight")
                rationale = str(per[t].get("rationale_short", "")).strip()
                if "0 비중" not in rationale:
                    per[t]["rationale_short"] = (rationale + " 제안 비중이 0이라 유지.") if rationale else "제안 비중이 0이라 0 비중 유지."
        decision["per_ticker_decisions"] = per

    # 피드백 루프용 grade 결정
    fb = decision.get("orchestrator_feedback", {})
    grade = "High" if fb.get("required", False) else "Low"

    risk_assessment = {
        "grade": grade,
        "risk_decision": decision,
        "risk_payload": payload,
    }
    positions_final = {
        t: float((d or {}).get("final_weight", 0.0))
        for t, d in (decision.get("per_ticker_decisions", {}) or {}).items()
    }

    print(f"   [결과] 리스크 등급: {grade}")
    if stale_desks:
        print(f"   [결과] Stale desks: {stale_desks}")
    if fb.get("required"):
        print(f"   [결과] 피드백 사유: {fb.get('reasons')}")
    per = decision.get("per_ticker_decisions", {})
    for t, d in per.items():
        print(f"   [결과] {t}: {d.get('decision')} → {d.get('final_weight')}")
    kill_switch = (decision.get("portfolio_actions", {}) or {}).get("kill_switch", {}) or {}
    if kill_switch.get("active"):
        print(f"   [결과] Kill switch: {kill_switch.get('severity')} gross->{kill_switch.get('target_gross_exposure')}")
    print(f"   [CoT] {fb.get('detail', '')[:120]}...")

    return {"risk_assessment": risk_assessment, "positions_final": positions_final}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# __main__ — Mock 시뮬레이션
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    SEP = "=" * 65
    DASH = "─" * 65

    print(SEP)
    print("⑥ RISK MANAGER — Mock 시뮬레이션")
    print(SEP)

    # 시나리오: 단일 종목(AAPL), Macro recession, Funda에 structural risk
    TICKER = "AAPL"

    mock_positions = {
        TICKER: {
            "weight": 0.12,
            "sector": "Technology",
            "avg_daily_volume_usd": 8_000_000_000,
            "position_notional_usd": 60_000_000,
        },
        "MSFT": {
            "weight": 0.08,
            "sector": "Technology",
            "avg_daily_volume_usd": 6_000_000_000,
            "position_notional_usd": 40_000_000,
        },
        "JPM": {
            "weight": -0.05,
            "sector": "Financials",
            "avg_daily_volume_usd": 3_000_000_000,
            "position_notional_usd": 25_000_000,
        },
    }

    # 합성 수익률 매트릭스 (250일 × 3종목)
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.015, (250, 3))
    returns[:, 0] *= 1.3   # AAPL 더 변동적
    ticker_order = [TICKER, "MSFT", "JPM"]

    mock_analyst_reports = {
        "macro": {
            "regime": "recession",
            "gdp_growth": -0.5,
            "interest_rate": 5.25,
        },
        "fundamental": {
            "sector": "Technology",
            "risk_flags": ["regulatory_action"],
            "revenue_growth": 5.2,
            "debt_to_equity": 1.8,
        },
        "sentiment": {
            "overall_sentiment": "negative",
            "sentiment_score": -0.35,
        },
        "quant": {
            "decision": "LONG",
            "final_allocation_pct": 0.12,
            "z_score": -2.5,
            "regime_2_high_vol": 0.15,
            "asset_cvar_99_daily": 0.028,
        },
        "_target_ticker": TICKER,
    }

    # ── Part 1: Portfolio Risk Summary ────────────────────────────────────
    print(f"\n{DASH}\nPart 1: calculate_portfolio_risk_summary\n{DASH}")
    risk_summary = calculate_portfolio_risk_summary(
        mock_positions, returns_matrix=returns, ticker_order=ticker_order
    )
    print(json.dumps(risk_summary, indent=2, ensure_ascii=False))

    # ── Part 1B: Aggregate Payload ────────────────────────────────────────
    print(f"\n{DASH}\nPart 1B: aggregate_risk_payload\n{DASH}")
    payload = aggregate_risk_payload(risk_summary, mock_analyst_reports)
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))

    # ── Part 2: LLM 5-Gate Decision ───────────────────────────────────────
    print(f"\n{DASH}\nPart 2: LLM 5-Gate CRO 의사결정\n{DASH}")
    decision = _call_llm(payload)

    print(f"\n{SEP}\n✅ 최종 RiskManagerOutput JSON\n{SEP}")
    print(json.dumps(decision, indent=2, ensure_ascii=False))

    # ── LangGraph 노드 테스트 ─────────────────────────────────────────────
    print(f"\n{SEP}\nLangGraph Node 통합 테스트: risk_manager_node(state)\n{SEP}")
    mock_state: InvestmentState = {
        "user_request": "AAPL 분석해줘",
        "target_ticker": TICKER,
        "analysis_tasks": [],
        "macro_analysis": mock_analyst_reports["macro"],
        "fundamental_analysis": mock_analyst_reports["fundamental"],
        "sentiment_analysis": mock_analyst_reports["sentiment"],
        "technical_analysis": mock_analyst_reports["quant"],
        "risk_assessment": {},
        "final_report": "",
        "iteration_count": 1,
    }
    node_out = risk_manager_node(mock_state)
    print(f"\n[노드 반환값 risk_assessment.grade]    → {node_out['risk_assessment']['grade']}")
    fb = node_out["risk_assessment"]["risk_decision"].get("orchestrator_feedback", {})
    print(f"[feedback.required]                   → {fb.get('required')}")
    print(f"[feedback.reasons]                    → {fb.get('reasons')}")
