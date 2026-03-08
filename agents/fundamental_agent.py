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


def _parse_date(value: Any):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return datetime.strptime(text[:10], "%Y-%m-%d").date()
        except ValueError:
            return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _annualize_return(target_price: Any, current_price: Any, horizon_days: int) -> float | None:
    target = _safe_float(target_price)
    current = _safe_float(current_price)
    if target in (None, 0) or current in (None, 0) or horizon_days <= 0:
        return None
    total = target / current
    if total <= 0:
        return None
    return round((total ** (365 / max(horizon_days, 1)) - 1) * 100, 2)


def _build_valuation_history_proxy(financials: dict) -> dict | None:
    current_pe = _safe_float(financials.get("pe_ratio"))
    pe_50 = _safe_float(financials.get("pe_ratio_50d_proxy"))
    pe_200 = _safe_float(financials.get("pe_ratio_200d_proxy"))
    values = [x for x in [pe_200, pe_50, current_pe] if x not in (None, 0)]
    if len(values) < 3:
        return None
    return {"pe_ratios": values, "source": "price_average_proxy"}


def _sector_metric_name(financials: dict) -> tuple[str, str]:
    sector = str(financials.get("sector", "")).lower()
    if any(x in sector for x in ("bank", "financial", "insurance")):
        return "roe", "ROE"
    if any(x in sector for x in ("utility", "real estate", "reit")):
        return "dividend_cash_yield_pct", "Dividend Yield"
    return "operating_margin", "Operating Margin"


def _build_peer_comp_engine(financials: dict, peers: Optional[list]) -> dict:
    if not peers:
        return {"status": "insufficient_data", "peer_count": 0, "peers": []}
    peer_rows = [p for p in peers if isinstance(p, dict) and p.get("symbol")]
    if not peer_rows:
        return {"status": "insufficient_data", "peer_count": 0, "peers": []}
    current_pe = _safe_float(financials.get("pe_ratio"))
    current_ps = _safe_float(financials.get("ps_ratio"))
    peer_pes = sorted(_safe_float(p.get("pe_ratio")) for p in peer_rows if _safe_float(p.get("pe_ratio")) not in (None, 0))
    peer_pss = sorted(_safe_float(p.get("ps_ratio")) for p in peer_rows if _safe_float(p.get("ps_ratio")) not in (None, 0))
    sector_key, sector_label = _sector_metric_name(financials)
    peer_sector_vals = sorted(
        _safe_float(p.get(sector_key)) for p in peer_rows if _safe_float(p.get(sector_key)) is not None
    )
    peer_median_pe = peer_pes[len(peer_pes) // 2] if peer_pes else None
    peer_median_ps = peer_pss[len(peer_pss) // 2] if peer_pss else None
    peer_median_sector = peer_sector_vals[len(peer_sector_vals) // 2] if peer_sector_vals else None
    pe_percentile = None
    pe_premium = None
    if current_pe not in (None, 0) and peer_pes:
        pe_percentile = round(sum(1 for x in peer_pes if x < current_pe) / len(peer_pes) * 100, 1)
        if peer_median_pe not in (None, 0):
            pe_premium = round((current_pe / peer_median_pe - 1) * 100, 2)
    ps_premium = None
    if current_ps not in (None, 0) and peer_median_ps not in (None, 0):
        ps_premium = round((current_ps / peer_median_ps - 1) * 100, 2)
    current_sector_metric = _safe_float(financials.get(sector_key))
    sector_premium = None
    if current_sector_metric is not None and peer_median_sector not in (None, 0):
        sector_premium = round((current_sector_metric / peer_median_sector - 1) * 100, 2)
    status = "ok" if peer_median_pe is not None or peer_median_ps is not None else "insufficient_data"
    return {
        "status": status,
        "peer_count": len(peer_rows),
        "peer_symbols": [str(p.get("symbol")) for p in peer_rows[:5]],
        "peers": peer_rows[:5],
        "peer_median_pe": peer_median_pe,
        "peer_median_ps": peer_median_ps,
        "pe_percentile_rank": pe_percentile,
        "pe_premium_discount_pct": pe_premium,
        "ps_premium_discount_pct": ps_premium,
        "sector_specific_metric": {
            "name": sector_label,
            "field": sector_key,
            "current": current_sector_metric,
            "peer_median": peer_median_sector,
            "premium_discount_pct": sector_premium,
        },
    }


def _build_consensus_revision_layer(financials: dict, as_of: str) -> dict:
    consensus = _build_consensus_snapshot(financials, as_of)
    estimate_periods = financials.get("estimate_periods") or {}
    q0 = estimate_periods.get("0q") or {}
    y0 = estimate_periods.get("0y") or {}
    y1 = estimate_periods.get("+1y") or {}
    score_30d = _safe_float(financials.get("rating_revision_score_30d"))
    score_90d = _safe_float(financials.get("rating_revision_score_90d"))
    upside = _safe_float(financials.get("price_target_upside_pct"))
    revision_proxy = str(financials.get("street_revision_proxy", "unknown")).strip().lower() or "unknown"
    q_rev_30 = _safe_float(q0.get("eps_revision_30d_pct"))
    q_rev_90 = _safe_float(q0.get("eps_revision_90d_pct"))
    y_rev_30 = _safe_float(y0.get("eps_revision_30d_pct"))
    y_rev_90 = _safe_float(y0.get("eps_revision_90d_pct"))
    if q0 or y0:
        revision_mode = "yfinance_eps_trend"
        revision_state = str(q0.get("revision_state") or y0.get("revision_state") or "unknown").lower()
        if upside is None and revision_state == "unknown":
            expectation = "insufficient_data"
        elif revision_state == "improving" and (upside is None or upside >= 10):
            expectation = "street constructive"
        elif revision_state == "deteriorating" or (upside is not None and upside < 0):
            expectation = "street expectations compressing"
        else:
            expectation = "street balanced"
    else:
        revision_mode = "rating_actions_proxy"
        revision_state = revision_proxy
        if upside is None and revision_proxy == "unknown":
            expectation = "insufficient_data"
        elif revision_proxy == "improving" and (upside is None or upside >= 10):
            expectation = "street constructive"
        elif revision_proxy == "deteriorating" or (upside is not None and upside < 0):
            expectation = "street expectations compressing"
        else:
            expectation = "street balanced"
    return {
        "status": "ok" if consensus.get("next_eps_estimate") is not None or consensus.get("next_fy_eps_estimate") is not None else "insufficient_data",
        "next_quarter": {
            "eps_estimate": q0.get("eps_estimate", consensus.get("next_eps_estimate")),
            "revenue_estimate": q0.get("revenue_estimate", consensus.get("next_revenue_estimate")),
            "eps_growth": q0.get("eps_growth"),
            "revenue_growth": q0.get("revenue_growth"),
            "earnings_date": financials.get("next_earnings_date_yahoo") or consensus.get("next_earnings_date"),
        },
        "next_fy": {
            "fiscal_year": consensus.get("analyst_estimate_fy"),
            "eps_estimate": y0.get("eps_estimate", consensus.get("next_fy_eps_estimate")),
            "revenue_estimate": y0.get("revenue_estimate", consensus.get("next_fy_revenue_estimate")),
            "eps_growth": y0.get("eps_growth"),
            "revenue_growth": y0.get("revenue_growth"),
        },
        "next_fy_plus1": {
            "eps_estimate": y1.get("eps_estimate"),
            "revenue_estimate": y1.get("revenue_estimate"),
            "eps_growth": y1.get("eps_growth"),
            "revenue_growth": y1.get("revenue_growth"),
        },
        "street_rating_consensus": {
            "label": financials.get("analyst_consensus_label") or consensus.get("rating"),
            "strong_buy": financials.get("grades_consensus_strong_buy"),
            "buy": financials.get("grades_consensus_buy"),
            "hold": financials.get("grades_consensus_hold"),
            "sell": financials.get("grades_consensus_sell"),
            "strong_sell": financials.get("grades_consensus_strong_sell"),
        },
        "estimate_revision_proxy": {
            "mode": revision_mode,
            "revision_30d_score": score_30d if revision_mode == "rating_actions_proxy" else None,
            "revision_90d_score": score_90d if revision_mode == "rating_actions_proxy" else None,
            "revision_30d_pct_next_q": q_rev_30,
            "revision_90d_pct_next_q": q_rev_90,
            "revision_30d_pct_fy": y_rev_30,
            "revision_90d_pct_fy": y_rev_90,
            "up_last_30d_next_q": q0.get("up_last_30d"),
            "down_last_30d_next_q": q0.get("down_last_30d"),
            "up_last_30d_fy": y0.get("up_last_30d"),
            "down_last_30d_fy": y0.get("down_last_30d"),
            "state": revision_state,
        },
        "market_expectation_view": {
            "target_upside_pct": upside,
            "judgement": expectation,
        },
    }


def _build_model_pack(financials: dict, peer_comp: dict, horizon_days: int) -> dict:
    estimate_periods = financials.get("estimate_periods") or {}
    y0 = estimate_periods.get("0y") or {}
    hist = (financials.get("valuation_history_real") or {}).get("valuation_points") or []
    current_price = _safe_float(financials.get("current_price"))
    current_rev = _safe_float(financials.get("ttm_revenue_real")) or _safe_float(financials.get("revenue"))
    revenue_base = _safe_float(financials.get("analyst_revenue_estimate_next_fy")) or _safe_float(y0.get("revenue_estimate")) or current_rev
    eps_base = _safe_float(financials.get("analyst_eps_estimate_next_fy")) or _safe_float(y0.get("eps_estimate")) or _safe_float(financials.get("eps_ttm_real")) or _safe_float(financials.get("eps_ttm_proxy"))
    current_margin = _safe_float(financials.get("ttm_operating_margin_real")) or _safe_float(financials.get("operating_margin")) or 15.0
    current_fcf = _safe_float(financials.get("ttm_fcf_real")) or _safe_float(financials.get("free_cash_flow"))
    current_fcf_margin = _safe_float(financials.get("ttm_fcf_margin_real"))
    if current_fcf_margin is None and current_rev not in (None, 0) and current_fcf is not None:
        current_fcf_margin = round(current_fcf / current_rev * 100, 2)
    peer_median_pe = _safe_float(peer_comp.get("peer_median_pe"))
    current_pe = _safe_float(financials.get("pe_ratio"))
    if eps_base is None or current_price in (None, 0):
        return {"status": "insufficient_data"}

    hist_pes = [float(x.get("pe_ratio")) for x in hist if _safe_float(x.get("pe_ratio")) not in (None, 0)]
    hist_median_pe = None
    if hist_pes:
        hist_pes = sorted(hist_pes)
        hist_median_pe = hist_pes[len(hist_pes) // 2]
    base_multiple = peer_median_pe or hist_median_pe or current_pe or 18.0
    bull_multiple = round(base_multiple * 1.08, 2)
    bear_multiple = round(base_multiple * 0.85, 2)
    q0 = estimate_periods.get("0q") or {}
    q1 = estimate_periods.get("+1q") or {}
    next_q_growth = _safe_float(q0.get("revenue_growth")) or 0.0
    next_fy_growth = _safe_float(y0.get("revenue_growth")) or 0.0
    bull_eps = round(eps_base * (1 + max(next_fy_growth, 0) / 100 * 0.7 + 0.04), 2)
    bear_eps = round(eps_base * max(0.7, 1 + min(next_fy_growth, 0) / 100 * 0.7 - 0.05), 2)
    price_target_consensus = _safe_float(financials.get("price_target_consensus"))
    price_target_high = _safe_float(financials.get("price_target_high"))
    price_target_low = _safe_float(financials.get("price_target_low"))
    model_base_target = round(eps_base * base_multiple, 2)
    base_target = round(price_target_consensus or model_base_target, 2)
    bull_target = round(price_target_high or max(base_target, bull_eps * bull_multiple), 2)
    bear_target = round(price_target_low or min(base_target, bear_eps * bear_multiple), 2)

    rev_growth_base = None
    if current_rev not in (None, 0) and revenue_base is not None:
        rev_growth_base = round((revenue_base / current_rev - 1) * 100, 2)

    def _fcf_forecast(revenue_val: float | None, margin_pct: float | None) -> float | None:
        if revenue_val is None or margin_pct is None:
            return None
        return round(revenue_val * margin_pct / 100, 2)

    bull_rev = round(revenue_base * (1 + max(next_fy_growth, 0) / 100 + 0.03), 2) if revenue_base is not None else None
    bear_rev = round(revenue_base * max(0.8, 1 + min(next_q_growth, 0) / 100 - 0.03), 2) if revenue_base is not None else None
    implied_margin_base = None
    shares = _safe_float(financials.get("shares_outstanding_real")) or _safe_float(financials.get("shares_outstanding_est"))
    if revenue_base not in (None, 0) and shares not in (None, 0):
        implied_margin_base = round((eps_base * shares) / revenue_base * 100, 2)
    base_margin = round(implied_margin_base if implied_margin_base is not None else current_margin, 2)
    bull_margin = round(max(base_margin, current_margin) + 1.0, 2)
    bear_margin = round(max(min(base_margin, current_margin) - 1.5, 0.0), 2)
    base_fcf = _fcf_forecast(revenue_base, current_fcf_margin)
    bull_fcf = _fcf_forecast(bull_rev, None if current_fcf_margin is None else current_fcf_margin + 0.75)
    bear_fcf = _fcf_forecast(bear_rev, None if current_fcf_margin is None else max(current_fcf_margin - 1.0, 0.0))

    return {
        "status": "ok",
        "basis": "real TTM financials + street FY estimates + market target / peer multiple anchor",
        "revenue_build": {
            "current_revenue": current_rev,
            "base_revenue": revenue_base,
            "base_growth_pct": rev_growth_base,
            "bull_revenue": bull_rev,
            "bear_revenue": bear_rev,
            "next_quarter_growth_pct": next_q_growth,
            "next_fy_growth_pct": next_fy_growth,
            "history_quarters": [
                {"date": x.get("date"), "revenue": x.get("revenue")}
                for x in (financials.get("quarterly_history_real") or [])[-4:]
            ],
        },
        "margin_bridge": {
            "current_operating_margin_pct": current_margin,
            "base_operating_margin_pct": base_margin,
            "bull_operating_margin_pct": bull_margin,
            "bear_operating_margin_pct": bear_margin,
            "ttm_fcf_margin_pct": current_fcf_margin,
            "implied_base_net_margin_pct": implied_margin_base,
            "history_quarters": [
                {"date": x.get("date"), "operating_margin_pct": x.get("operating_margin_pct")}
                for x in (financials.get("quarterly_history_real") or [])[-4:]
            ],
        },
        "eps_fcf": {
            "base_eps": round(eps_base, 2),
            "bull_eps": bull_eps,
            "bear_eps": bear_eps,
            "base_fcf": base_fcf,
            "bull_fcf": bull_fcf,
            "bear_fcf": bear_fcf,
            "eps_ttm_real": _safe_float(financials.get("eps_ttm_real")),
            "fcf_ttm_real": current_fcf,
        },
        "scenario_targets": {
            "bull": {
                "target_price": bull_target,
                "upside_pct": round((bull_target / current_price - 1) * 100, 2),
                "irr_pct": _annualize_return(bull_target, current_price, horizon_days),
            },
            "base": {
                "target_price": base_target,
                "upside_pct": round((base_target / current_price - 1) * 100, 2),
                "irr_pct": _annualize_return(base_target, current_price, horizon_days),
            },
            "bear": {
                "target_price": bear_target,
                "downside_pct": round((bear_target / current_price - 1) * 100, 2),
                "irr_pct": _annualize_return(bear_target, current_price, horizon_days),
            },
        },
    }


def _build_street_disagreement_layer(financials: dict, consensus_revision_layer: dict, model_pack: dict) -> dict:
    target_mean = _safe_float(financials.get("price_target_consensus"))
    target_high = _safe_float(financials.get("price_target_high"))
    target_low = _safe_float(financials.get("price_target_low"))
    street_upside = _safe_float(financials.get("price_target_upside_pct"))
    base_upside = _safe_float((((model_pack.get("scenario_targets") or {}).get("base") or {}).get("upside_pct")))

    total_ratings = sum(
        int(_safe_float(financials.get(k)) or 0)
        for k in (
            "grades_consensus_strong_buy",
            "grades_consensus_buy",
            "grades_consensus_hold",
            "grades_consensus_sell",
            "grades_consensus_strong_sell",
        )
    )
    positive = sum(int(_safe_float(financials.get(k)) or 0) for k in ("grades_consensus_strong_buy", "grades_consensus_buy"))
    neutral = int(_safe_float(financials.get("grades_consensus_hold")) or 0)
    negative = sum(int(_safe_float(financials.get(k)) or 0) for k in ("grades_consensus_sell", "grades_consensus_strong_sell"))
    coverage_depth = max(
        int(_safe_float(financials.get("analyst_num_eps")) or 0),
        int(_safe_float(financials.get("analyst_num_revenue")) or 0),
        total_ratings,
    )

    dispersion_pct = None
    if target_mean not in (None, 0) and target_high is not None and target_low is not None:
        dispersion_pct = round((target_high - target_low) / target_mean * 100, 2)
    model_gap_pct = None
    if street_upside is not None and base_upside is not None:
        model_gap_pct = round(base_upside - street_upside, 2)

    positive_ratio = round(positive / total_ratings, 3) if total_ratings else None
    hold_sell_ratio = round((neutral + negative) / total_ratings, 3) if total_ratings else None
    uncertainty = "medium"
    if (coverage_depth and coverage_depth < 8) or (dispersion_pct is not None and dispersion_pct >= 30) or (hold_sell_ratio is not None and hold_sell_ratio >= 0.55):
        uncertainty = "high"
    elif (coverage_depth and coverage_depth >= 18) and (dispersion_pct is None or dispersion_pct <= 15) and (hold_sell_ratio is None or hold_sell_ratio <= 0.35):
        uncertainty = "low"

    if positive_ratio is not None and positive_ratio >= 0.7 and uncertainty != "high":
        consensus_strength = "strong"
    elif hold_sell_ratio is not None and hold_sell_ratio >= 0.5:
        consensus_strength = "fragile"
    else:
        consensus_strength = "balanced"

    revision_state = str(((consensus_revision_layer.get("estimate_revision_proxy") or {}).get("state")) or "unknown").lower()
    disagreement = "aligned"
    if model_gap_pct is not None and abs(model_gap_pct) >= 12:
        disagreement = "model_above_street" if model_gap_pct > 0 else "street_above_model"
    elif revision_state == "deteriorating" and street_upside is not None and street_upside > 10:
        disagreement = "street_too_optimistic"

    return {
        "coverage_depth": coverage_depth or None,
        "consensus_strength": consensus_strength,
        "uncertainty_level": uncertainty,
        "target_dispersion_pct": dispersion_pct,
        "rating_balance": {
            "positive": positive,
            "neutral": neutral,
            "negative": negative,
            "positive_ratio": positive_ratio,
            "hold_sell_ratio": hold_sell_ratio,
        },
        "street_target_upside_pct": street_upside,
        "model_base_upside_pct": base_upside,
        "model_vs_street_gap_pct": model_gap_pct,
        "disagreement_signal": disagreement,
    }


def _build_expected_return_profile(financials: dict, model_pack: dict, street_disagreement: dict) -> dict:
    street_upside = _safe_float(street_disagreement.get("street_target_upside_pct"))
    base_upside = _safe_float(street_disagreement.get("model_base_upside_pct"))
    bull_upside = _safe_float((((model_pack.get("scenario_targets") or {}).get("bull") or {}).get("upside_pct")))
    bear_downside = _safe_float((((model_pack.get("scenario_targets") or {}).get("bear") or {}).get("downside_pct")))
    downside_abs = abs(bear_downside) if bear_downside is not None else None

    anchors = [x for x in (street_upside, base_upside) if x is not None]
    anchor_upside = round(sum(anchors) / len(anchors), 2) if anchors else None
    reward_risk = None
    if anchor_upside is not None and downside_abs not in (None, 0):
        reward_risk = round(anchor_upside / downside_abs, 2)

    quality = "balanced"
    if anchor_upside is None:
        quality = "insufficient_data"
    elif anchor_upside < 5 or (reward_risk is not None and reward_risk < 1.0):
        quality = "thin"
    elif anchor_upside >= 15 and (reward_risk is None or reward_risk >= 1.5):
        quality = "attractive"

    if str(street_disagreement.get("uncertainty_level", "")).lower() == "high" and quality == "attractive":
        quality = "balanced"
    elif str(street_disagreement.get("uncertainty_level", "")).lower() == "high" and quality == "balanced":
        quality = "thin"

    return {
        "anchor_upside_pct": anchor_upside,
        "street_upside_pct": street_upside,
        "base_upside_pct": base_upside,
        "bull_upside_pct": bull_upside,
        "bear_downside_pct": bear_downside,
        "reward_risk_ratio": reward_risk,
        "quality": quality,
    }


def _infer_expected_scenario(cat_type: str) -> str:
    mapping = {
        "earnings": "실적/가이던스가 현재 multiple을 지지하는지 확인",
        "investor_day": "중기 성장 로드맵과 capital allocation update 점검",
        "product_cycle": "신제품/출시 주기가 성장 가속으로 이어지는지 확인",
        "legal_reg": "규제/소송 이슈가 valuation 할인 요인이 되는지 확인",
        "contract_renewal": "주요 계약 유지/갱신 여부가 매출 가시성에 미치는 영향 점검",
        "pricing_reset": "가격 인상/인하가 마진 지속성에 미치는 영향 점검",
        "street_rating": "sell-side 태도 변화가 기대수익률 앵커를 바꾸는지 확인",
    }
    return mapping.get(cat_type, "thesis relevant catalyst")


def _infer_thesis_trigger(cat_type: str) -> str:
    mapping = {
        "earnings": "EPS/매출 또는 가이던스가 컨센서스를 하회하면 thesis 약화",
        "investor_day": "중기 성장률/마진 목표가 기존 기대를 하회하면 thesis 약화",
        "product_cycle": "출시 지연 또는 초기 수요 부진 시 thesis 약화",
        "legal_reg": "제재/판결/조사 범위 확대 시 thesis 약화",
        "contract_renewal": "핵심 계약 미갱신 또는 가격조건 악화 시 thesis 약화",
        "pricing_reset": "가격 인하로 마진 훼손 시 thesis 약화",
        "street_rating": "downgrade 연속 및 target 하향이 겹치면 conviction 약화",
    }
    return mapping.get(cat_type, "material negative update")


_CONFIRMED_CATALYST_HOSTS = (
    "sec.gov",
    "prnewswire.com",
    "businesswire.com",
    "globenewswire.com",
)


def _detect_catalyst_type(text: str) -> str | None:
    lowered = text.lower()
    if any(k in lowered for k in ("investor day", "analyst day", "capital markets day")):
        return "investor_day"
    if any(k in lowered for k in ("launch", "shipment", "release", "product", "iphone", "gpu", "drug", "approval")):
        return "product_cycle"
    if any(k in lowered for k in ("regulation", "antitrust", "lawsuit", "court", "sec enforcement", "sec probe", "sec investigation", "doj", "probe", "investigation")):
        return "legal_reg"
    if any(k in lowered for k in ("renewal", "contract", "customer", "supplier", "deal")):
        return "contract_renewal"
    if any(k in lowered for k in ("pricing", "price hike", "price cut", "discount")):
        return "pricing_reset"
    return None


def _is_confirmed_catalyst_evidence(item: dict) -> bool:
    resolver = str(item.get("resolver_path", "")).strip().lower()
    kind = str(item.get("kind", "")).strip().lower()
    url = str(item.get("url", "")).strip().lower()
    source = str(item.get("source", "")).strip().lower()
    if kind == "sec_filing":
        return True
    if resolver.startswith("sec_") or resolver == "official_release":
        return True
    if kind == "press_release_or_ir" and ("ir" in resolver or "press_release" in resolver):
        return True
    if any(host in url for host in _CONFIRMED_CATALYST_HOSTS):
        return True
    return source == "sec.gov"


def _merge_catalyst_event(out: list[dict], event: dict) -> None:
    new_type = str(event.get("type", "")).strip()
    new_class = str(event.get("source_classification", "inferred")).strip().lower()
    for idx, existing in enumerate(out):
        if str(existing.get("type", "")).strip() != new_type:
            continue
        old_class = str(existing.get("source_classification", "inferred")).strip().lower()
        if old_class == "confirmed":
            return
        if new_class == "confirmed":
            out[idx] = event
        return
    out.append(event)


def _build_catalyst_engine(
    financials: dict,
    evidence_digest: list[dict],
    as_of: str,
    catalyst_items: Optional[list[dict]] = None,
) -> list[dict]:
    out = _build_catalyst_calendar(financials, as_of)
    recent_actions = financials.get("recent_rating_actions") or []
    for row in recent_actions[:3]:
        _merge_catalyst_event(
            out,
            {
                "type": "street_rating",
                "date": row.get("date"),
                "days_to_event": None,
                "importance": "medium",
                "expected_scenario": _infer_expected_scenario("street_rating"),
                "thesis_change_trigger": _infer_thesis_trigger("street_rating"),
                "firm": row.get("firm"),
                "action": row.get("action"),
                "status": "observed",
                "source_classification": "confirmed",
                "source_title": f"{row.get('firm') or 'sell-side'} {row.get('action') or 'rating update'}",
                "resolver_path": "fmp_ratings",
                "evidence_kind": "street_rating",
            },
        )
    seen_urls: set[str] = set()
    source_items = []
    for item in (catalyst_items or []) + list(evidence_digest):
        if not isinstance(item, dict):
            continue
        key = str(item.get("url", "")).strip() or str(item.get("title", "")).strip()
        if key and key in seen_urls:
            continue
        if key:
            seen_urls.add(key)
        source_items.append(item)
    for item in source_items:
        cat = str(item.get("catalyst_type", "")).strip() or None
        if not cat:
            text = " ".join([str(item.get("title", "")), str(item.get("snippet", ""))]).strip()
            cat = _detect_catalyst_type(text)
        if not cat:
            cat = _detect_catalyst_type(str(item.get("url", "")))
        if not cat:
            continue
        classification = str(item.get("source_classification", "")).strip().lower()
        if classification not in {"confirmed", "inferred"}:
            classification = "confirmed" if _is_confirmed_catalyst_evidence(item) else "inferred"
        _merge_catalyst_event(
            out,
            {
                "type": cat,
                "date": item.get("published_at"),
                "days_to_event": None,
                "importance": "medium",
                "expected_scenario": _infer_expected_scenario(cat),
                "thesis_change_trigger": _infer_thesis_trigger(cat),
                "status": classification,
                "source_classification": classification,
                "source_title": item.get("title"),
                "source_url": item.get("url"),
                "resolver_path": item.get("resolver_path", ""),
                "evidence_kind": item.get("kind", ""),
                "source": item.get("source", ""),
                "filing_items": item.get("filing_items", []),
                "exhibit_codes": item.get("exhibit_codes", []),
            },
        )
        if len(out) >= 6:
            break
    return out[:6]


def _build_management_capital_allocation(financials: dict, sec_data: Optional[dict]) -> dict:
    eps_surprise = _safe_float(financials.get("last_eps_surprise_pct"))
    beat_rate = _safe_float(financials.get("earnings_beat_rate_4q"))
    eps_surprise_avg = _safe_float(financials.get("eps_surprise_avg_4q"))
    revenue_beat_rate = _safe_float(financials.get("revenue_beat_rate_4q"))
    governance_scores = [
        _safe_float(financials.get("audit_risk_yahoo")),
        _safe_float(financials.get("board_risk_yahoo")),
        _safe_float(financials.get("compensation_risk_yahoo")),
        _safe_float(financials.get("shareholder_rights_risk_yahoo")),
    ]
    governance_scores = [x for x in governance_scores if x is not None]
    governance_risk_avg = round(sum(governance_scores) / len(governance_scores), 2) if governance_scores else None
    sec_flags = {
        "going_concern": bool((sec_data or {}).get("has_going_concern_language")),
        "restatement": bool((sec_data or {}).get("has_recent_restatement") or (sec_data or {}).get("has_restatement")),
        "material_weakness": bool((sec_data or {}).get("has_material_weakness_icfr") or (sec_data or {}).get("has_material_weakness")),
        "regulatory_action": bool((sec_data or {}).get("regulatory_investigation_flag") or (sec_data or {}).get("has_regulatory_action")),
    }
    if any(sec_flags.values()):
        guidance = "low"
        governance = "weak"
    elif beat_rate is not None and beat_rate >= 75 and (eps_surprise_avg or 0) >= 2:
        guidance = "high"
        governance = "sound"
    elif beat_rate is not None and beat_rate <= 25:
        guidance = "low"
        governance = "average"
    elif eps_surprise is not None and eps_surprise <= -3:
        guidance = "low"
        governance = "average"
    else:
        guidance = "medium"
        governance = "sound"
    if governance_risk_avg is not None:
        if governance_risk_avg >= 7:
            governance = "weak"
        elif governance_risk_avg >= 4:
            governance = "average"

    buyback_yield = _safe_float(financials.get("buyback_yield_pct")) or 0.0
    dividend_yield = _safe_float(financials.get("dividend_cash_yield_pct")) or 0.0
    acq = _safe_float(financials.get("acquisitions_last_fy")) or 0.0
    debt_funded = bool(financials.get("debt_funded_buyback_flag"))
    capital_style = "balanced"
    if debt_funded and buyback_yield > 1:
        capital_style = "aggressive_financial_engineering"
    elif acq > 0 and acq > (_safe_float(financials.get("free_cash_flow")) or 0) * 0.5:
        capital_style = "acquisition_led"
    elif buyback_yield + dividend_yield > 3:
        capital_style = "shareholder_return_focused"
    return {
        "guidance_credibility": guidance,
        "earnings_beat_rate_4q": beat_rate,
        "revenue_beat_rate_4q": revenue_beat_rate,
        "eps_surprise_avg_4q": eps_surprise_avg,
        "capital_allocation_style": capital_style,
        "buyback_yield_pct": round(buyback_yield, 2),
        "dividend_cash_yield_pct": round(dividend_yield, 2),
        "capital_return_yield_pct": round((_safe_float(financials.get("capital_return_yield_pct")) or 0.0), 2),
        "acquisitions_last_fy": acq,
        "debt_funded_buyback_flag": debt_funded,
        "governance_accounting_quality": governance,
        "governance_risk_avg_yahoo": governance_risk_avg,
        "governance_flags": sec_flags,
    }


def _build_ownership_crowding(financials: dict, ownership_items: Optional[list], ownership_snapshot: Optional[dict], as_of: str) -> dict:
    if isinstance(ownership_snapshot, dict) and ownership_snapshot.get("status") == "ok":
        buyers = (((ownership_snapshot.get("incremental_buyer_seller_map") or {}).get("buyers")) or [])[:5]
        sellers = (((ownership_snapshot.get("incremental_buyer_seller_map") or {}).get("sellers")) or [])[:5]
        return {
            "status": "structured",
            "institutional_concentration": {
                "institutions_percent_held": ownership_snapshot.get("institutions_percent_held"),
                "institutions_float_percent_held": ownership_snapshot.get("institutions_float_percent_held"),
                "institutions_count": ownership_snapshot.get("institutions_count"),
                "top_holder_pct": ownership_snapshot.get("top_holder_pct"),
                "top10_pct": ownership_snapshot.get("institutional_top10_pct"),
                "hhi_top10": ownership_snapshot.get("institutional_hhi_top10"),
                "level": ownership_snapshot.get("institutional_concentration_level"),
            },
            "insider_activity": ownership_snapshot.get("insider_net_activity") or "unknown",
            "insider_net_shares_6m": ownership_snapshot.get("insider_net_shares_6m"),
            "crowding_risk": ownership_snapshot.get("crowding_risk") or "unknown",
            "incremental_buyer_seller_map": {
                "buyers": buyers,
                "sellers": sellers,
            },
            "ownership_report_date": ownership_snapshot.get("ownership_report_date"),
        }
    items = [x for x in (ownership_items or []) if isinstance(x, dict)]
    if not items:
        return {
            "status": "insufficient_data",
            "institutional_concentration": None,
            "insider_activity": "unknown",
            "crowding_risk": "unknown",
            "incremental_buyer_seller_map": [],
        }
    today = _parse_date(as_of)
    form4 = 0
    filings_13f = 0
    filings_13dg = 0
    for item in items:
        title = str(item.get("title", "")).lower()
        published = _parse_date(item.get("published_at"))
        if today is not None and published is not None and (today - published).days > 90:
            continue
        if " 4" in title or title.startswith("4 ") or "form 4" in title:
            form4 += 1
        if "13f" in title:
            filings_13f += 1
        if "13d" in title or "13g" in title:
            filings_13dg += 1
    crowding = "unknown"
    if filings_13dg > 0:
        crowding = "elevated"
    elif filings_13f > 10:
        crowding = "normal"
    return {
        "status": "signal_only",
        "institutional_concentration": None,
        "insider_activity": "active" if form4 > 0 else "unknown",
        "crowding_risk": crowding,
        "ownership_event_counts_90d": {
            "form4": form4,
            "thirteen_f": filings_13f,
            "thirteen_dg": filings_13dg,
        },
        "incremental_buyer_seller_map": [],
        "note": "structured holder concentration unavailable; filing density only",
    }


def _build_data_provenance(
    financials: dict,
    peer_comp_engine: dict,
    consensus_revision_layer: dict,
    catalyst_calendar: list[dict],
    ownership_crowding: dict,
    management_capital_allocation: dict,
) -> dict:
    history_source = str(financials.get("fundamental_history_source", "")).strip() or "proxy"
    estimate_mode = str(((consensus_revision_layer.get("estimate_revision_proxy") or {}).get("mode")) or "unknown")
    ownership_status = str(ownership_crowding.get("status", "insufficient_data"))
    peer_status = str(peer_comp_engine.get("status", "insufficient_data"))
    confirmed_catalysts = sum(1 for x in catalyst_calendar if str(x.get("source_classification", "")).lower() == "confirmed")
    inferred_catalysts = sum(1 for x in catalyst_calendar if str(x.get("source_classification", "")).lower() == "inferred")
    mgmt_raw_signals = sum(
        1
        for value in (
            management_capital_allocation.get("earnings_beat_rate_4q"),
            management_capital_allocation.get("revenue_beat_rate_4q"),
            management_capital_allocation.get("governance_risk_avg_yahoo"),
        )
        if value is not None
    )

    layers = {
        "history": {"source": history_source, "raw_backed": history_source == "yfinance_financials"},
        "estimates": {"source": estimate_mode, "raw_backed": estimate_mode == "yfinance_eps_trend"},
        "ownership": {"source": ownership_status, "raw_backed": ownership_status == "structured"},
        "peers": {"source": peer_status, "raw_backed": peer_status == "ok" and int(peer_comp_engine.get("peer_count") or 0) >= 3},
        "catalysts": {"source": "confirmed" if confirmed_catalysts else "inferred", "raw_backed": confirmed_catalysts > 0},
        "management": {"source": "mixed_raw" if mgmt_raw_signals >= 2 else "limited", "raw_backed": mgmt_raw_signals >= 2},
    }
    raw_count = sum(1 for layer in layers.values() if layer["raw_backed"])
    total_layers = len(layers)
    score = round(raw_count / total_layers, 3)
    quality = "high" if score >= 0.8 else "medium" if score >= 0.6 else "low"
    confidence_haircut = 0.0 if quality == "high" else 0.07 if quality == "medium" else 0.15
    weak_spots = []
    if history_source != "yfinance_financials":
        weak_spots.append("historical valuation anchor")
    if estimate_mode != "yfinance_eps_trend":
        weak_spots.append("estimate revision")
    if ownership_status != "structured":
        weak_spots.append("holder concentration")
    if peer_status != "ok":
        weak_spots.append("peer comp")
    if confirmed_catalysts == 0:
        weak_spots.append("confirmed catalysts")

    return {
        "quality": quality,
        "raw_backed_score": score,
        "raw_backed_layers": raw_count,
        "total_layers": total_layers,
        "confidence_haircut": confidence_haircut,
        "confirmed_catalyst_count": confirmed_catalysts,
        "inferred_catalyst_count": inferred_catalysts,
        "weak_spots": weak_spots,
        "layers": layers,
    }


def _build_consensus_snapshot(financials: dict, as_of: str) -> dict:
    current_price = financials.get("current_price")
    return {
        "current_price": current_price,
        "next_earnings_date": financials.get("next_earnings_date_yahoo") or financials.get("next_earnings_date"),
        "earnings_in_days": financials.get("earnings_in_days"),
        "next_eps_estimate": ((financials.get("estimate_periods") or {}).get("0q") or {}).get("eps_estimate", financials.get("next_eps_estimate")),
        "next_revenue_estimate": ((financials.get("estimate_periods") or {}).get("0q") or {}).get("revenue_estimate", financials.get("next_revenue_estimate")),
        "last_eps_surprise_pct": financials.get("last_eps_surprise_pct"),
        "analyst_estimate_fy": financials.get("analyst_estimate_fy"),
        "next_fy_eps_estimate": ((financials.get("estimate_periods") or {}).get("0y") or {}).get("eps_estimate", financials.get("analyst_eps_estimate_next_fy")),
        "next_fy_revenue_estimate": ((financials.get("estimate_periods") or {}).get("0y") or {}).get("revenue_estimate", financials.get("analyst_revenue_estimate_next_fy")),
        "num_analysts_eps": financials.get("analyst_num_eps"),
        "num_analysts_revenue": financials.get("analyst_num_revenue"),
        "price_target_consensus": financials.get("price_target_consensus"),
        "price_target_upside_pct": financials.get("price_target_upside_pct"),
        "rating": financials.get("rating"),
        "rating_overall_score": financials.get("rating_overall_score"),
        "rating_dcf_score": financials.get("rating_dcf_score"),
        "as_of": as_of,
    }


def _build_catalyst_calendar(financials: dict, as_of: str) -> list[dict]:
    out: list[dict] = []
    today = _parse_date(as_of)
    next_earnings = _parse_date(financials.get("next_earnings_date"))
    if next_earnings is not None:
        days_to = financials.get("earnings_in_days")
        if days_to is None and today is not None:
            days_to = (next_earnings - today).days
        out.append(
            {
                "type": "earnings",
                "date": next_earnings.isoformat(),
                "days_to_event": days_to,
                "importance": "high",
                "eps_estimate": financials.get("next_eps_estimate"),
                "revenue_estimate": financials.get("next_revenue_estimate"),
                "expected_scenario": _infer_expected_scenario("earnings"),
                "thesis_change_trigger": _infer_thesis_trigger("earnings"),
                "status": "upcoming" if days_to is None or days_to >= 0 else "stale",
                "source_classification": "confirmed",
                "source_title": "Structured earnings calendar",
                "resolver_path": "structured_calendar",
                "evidence_kind": "earnings_calendar",
            }
        )
    return out


def _build_valuation_anchor(financials: dict, vs: dict) -> dict:
    upside = financials.get("price_target_upside_pct")
    confidence = "medium" if upside is not None and financials.get("pe_ratio") is not None else "low"
    return {
        "current_price": financials.get("current_price"),
        "pe_ratio": financials.get("pe_ratio"),
        "ps_ratio": financials.get("ps_ratio"),
        "fcf_yield": financials.get("fcf_yield"),
        "price_target_consensus": financials.get("price_target_consensus"),
        "price_target_median": financials.get("price_target_median"),
        "price_target_upside_pct": upside,
        "stretch_level": vs.get("stretch_level"),
        "valuation_stretch_flag": vs.get("valuation_stretch_flag"),
        "confidence": confidence,
        "basis": "current multiple + sell-side target consensus" if upside is not None else "current multiple only",
    }


def _build_variant_view(
    financials: dict,
    fs: dict,
    vs: dict,
    consensus: dict,
    evidence_digest: list[dict],
    structural_flag: bool,
) -> dict:
    upside = consensus.get("price_target_upside_pct")
    days_to = consensus.get("earnings_in_days")
    stretch = vs.get("stretch_level", "unknown")
    quality = fs.get("quality_score", 0.5)
    growth = fs.get("growth_score", 0.5)
    evidence_kinds = sorted({str(item.get("kind", "")).strip() for item in evidence_digest if isinstance(item, dict)})

    market_bits = []
    if consensus.get("next_eps_estimate") is not None or consensus.get("next_revenue_estimate") is not None:
        market_bits.append("다음 실적에서 이익/매출 유지가 이미 기대됨")
    if upside is not None:
        market_bits.append(f"sell-side target 기준 upside {upside:.1f}%")
    if stretch in {"medium", "high"}:
        market_bits.append(f"현재 multiple은 이미 {stretch} valuation을 일부 반영")

    analyst_bits = []
    if structural_flag:
        analyst_bits.append("재무/공시 리스크가 내러티브보다 우선한다")
    else:
        if quality >= 0.7:
            analyst_bits.append("퀄리티는 강하지만 실행 미스 허용폭은 valuation이 결정")
        if growth >= 0.7:
            analyst_bits.append("성장 지속 여부가 valuation 유지의 핵심 조건")
        if upside is not None and upside <= 5:
            analyst_bits.append("현재 가격과 컨센서스 target 간 괴리가 작아 차별적 기대수익은 제한적")
        elif upside is not None and upside >= 15:
            analyst_bits.append("컨센서스 기준 upside는 남아 있지만 earnings execution 확인이 선행돼야 한다")
        elif upside is None:
            analyst_bits.append("target consensus 부재로 기대수익률 앵커가 약하다")

    if structural_flag:
        key_gap = "구조 리스크 여부가 가장 큰 판단 차이 변수"
    elif stretch == "high" and upside is not None and upside <= 5:
        key_gap = "높은 valuation 대비 기대수익률이 얕아 downside asymmetry가 커질 수 있음"
    elif days_to is not None and days_to <= 14:
        key_gap = "near-term earnings catalyst가 thesis validation의 핵심"
    else:
        key_gap = "실적/가이던스가 현재 multiple을 정당화하는지 여부"

    differentiation = "medium" if evidence_kinds else "low"
    return {
        "market_expectation": "; ".join(market_bits) if market_bits else "시장 기대치 데이터가 제한적임",
        "analyst_view": "; ".join(analyst_bits) if analyst_bits else "현재 펀더멘털 정보만으로는 차별적 뷰가 약함",
        "key_gap": key_gap,
        "differentiation_level": differentiation,
        "evidence_kinds": evidence_kinds,
    }


def _build_monitoring_triggers(
    consensus_revision_layer: dict,
    catalyst_calendar: list[dict],
    peer_comp_engine: dict,
    management_capital_allocation: dict,
    ownership_crowding: dict,
    expected_return_profile: dict,
    street_disagreement: dict,
) -> list[dict]:
    triggers: list[dict] = []
    first_earnings = next((x for x in catalyst_calendar if x.get("type") == "earnings"), None)
    if first_earnings:
        triggers.append(
            {
                "name": "Earnings catalyst",
                "metric": "earnings_in_days",
                "current_value": first_earnings.get("days_to_event"),
                "trigger": "<= 7d / miss vs consensus",
                "action": "모델·variant view·position sizing을 즉시 재평가",
                "priority": 1,
            }
        )

    revision = consensus_revision_layer.get("estimate_revision_proxy") or {}
    rev30 = _safe_float(revision.get("revision_30d_pct_next_q"))
    if rev30 is not None:
        triggers.append(
            {
                "name": "Estimate revision drift",
                "metric": "next_q_eps_revision_30d_pct",
                "current_value": rev30,
                "trigger": "< -2% or state=deteriorating",
                "action": "실적 기대치와 target price anchor를 다시 계산",
                "priority": 1,
            }
        )

    anchor_upside = _safe_float(expected_return_profile.get("anchor_upside_pct"))
    if anchor_upside is not None:
        triggers.append(
            {
                "name": "Expected return compression",
                "metric": "anchor_upside_pct",
                "current_value": anchor_upside,
                "trigger": "< 5%",
                "action": "신규 매수 중단 및 conviction 하향 여부 점검",
                "priority": 2,
            }
        )

    peer_premium = _safe_float(peer_comp_engine.get("pe_premium_discount_pct"))
    if peer_premium is not None:
        triggers.append(
            {
                "name": "Peer premium stress",
                "metric": "peer_pe_premium_discount_pct",
                "current_value": peer_premium,
                "trigger": "> 20%",
                "action": "multiple compression downside를 재평가",
                "priority": 2,
            }
        )

    crowding = str(ownership_crowding.get("crowding_risk", "")).lower()
    insider = str(ownership_crowding.get("insider_activity", "")).lower()
    if crowding in {"elevated", "high"} or insider == "selling":
        triggers.append(
            {
                "name": "Ownership pressure",
                "metric": "crowding_or_insider",
                "current_value": {"crowding_risk": crowding, "insider_activity": insider},
                "trigger": "crowding=elevated or insider=selling",
                "action": "수급 기반 downside와 exit liquidity를 재점검",
                "priority": 2,
            }
        )

    if management_capital_allocation.get("governance_accounting_quality") == "weak" or management_capital_allocation.get("capital_allocation_style") == "aggressive_financial_engineering":
        triggers.append(
            {
                "name": "Governance / capital allocation",
                "metric": "management_quality",
                "current_value": {
                    "guidance_credibility": management_capital_allocation.get("guidance_credibility"),
                    "capital_allocation_style": management_capital_allocation.get("capital_allocation_style"),
                    "governance_accounting_quality": management_capital_allocation.get("governance_accounting_quality"),
                },
                "trigger": "weak governance or aggressive financial engineering",
                "action": "thesis durability와 downside 시나리오를 재평가",
                "priority": 1,
            }
        )

    if str(street_disagreement.get("disagreement_signal", "")).strip().lower() not in {"", "aligned"}:
        triggers.append(
            {
                "name": "Street vs model divergence",
                "metric": "model_vs_street_gap_pct",
                "current_value": street_disagreement.get("model_vs_street_gap_pct"),
                "trigger": "|gap| >= 12%",
                "action": "variant view를 다시 쓰고 sizing 근거를 재검토",
                "priority": 2,
            }
        )

    return triggers[:7]


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
                "source": item.get("source", ""),
                "snippet": item.get("snippet", ""),
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


def _default_decision_sensitivity(stretch: str, structural_flag: bool, earnings_in_days: Any = None) -> list[dict]:
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
    if earnings_in_days is not None and earnings_in_days <= 14:
        out.append(
            {
                "if": "실적 발표에서 EPS/매출이 추정치 하회",
                "then_change": "near-term catalyst miss로 conviction 및 sizing 재평가",
                "impact": "high",
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
    ownership_items: Optional[list] = None,
    ownership_snapshot: Optional[dict] = None,
    catalyst_items: Optional[list[dict]] = None,
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
    history = history or financials.get("valuation_history_real") or _build_valuation_history_proxy(financials)

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
        "current_price",
        "revenue_growth",
        "earnings_growth",
        "ttm_revenue_real",
        "ttm_operating_margin_real",
        "ttm_fcf_real",
        "eps_ttm_real",
        "operating_margin",
        "roe",
        "debt_to_equity",
        "pe_ratio",
        "ps_ratio",
        "fcf_yield",
        "free_cash_flow",
        "next_eps_estimate",
        "next_revenue_estimate",
        "analyst_eps_estimate_next_fy",
        "analyst_revenue_estimate_next_fy",
        "price_target_consensus",
        "price_target_upside_pct",
        "rating_overall_score",
        "rating_revision_score_30d",
        "rating_revision_score_90d",
        "earnings_beat_rate_4q",
        "revenue_beat_rate_4q",
        "eps_surprise_avg_4q",
        "institutions_percent_held",
        "institutional_top10_pct",
        "insider_net_shares_6m",
        "buyback_yield_pct",
        "dividend_cash_yield_pct",
        "capital_return_yield_pct",
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
    earnings_in_days = financials.get("earnings_in_days")
    price_target_upside = financials.get("price_target_upside_pct")
    peer_comp_engine = _build_peer_comp_engine(financials, peers)
    consensus_revision_layer = _build_consensus_revision_layer(financials, as_of)
    model_pack = _build_model_pack(financials, peer_comp_engine, horizon_days)
    management_capital_allocation = _build_management_capital_allocation(financials, sec_data)
    ownership_crowding = _build_ownership_crowding(financials, ownership_items, ownership_snapshot, as_of)
    revision_state = str((consensus_revision_layer.get("estimate_revision_proxy") or {}).get("state", "unknown")).lower()
    base_upside = _safe_float((((model_pack.get("scenario_targets") or {}).get("base") or {}).get("upside_pct")))
    peer_premium = _safe_float(peer_comp_engine.get("pe_premium_discount_pct"))

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
    elif earnings_in_days is not None and earnings_in_days <= 14 and recommendation == "allow":
        recommendation = "allow_with_limits"
        confidence = min(confidence, 0.58)
        notes = notes + f" Earnings catalyst {earnings_in_days}일 이내로 sizing discipline 필요."
    elif price_target_upside is not None and price_target_upside <= -10 and not structural_flag:
        primary_decision = "bearish"
        recommendation = "allow_with_limits"
        confidence = min(confidence, 0.50)
        notes = notes + " Street target 기준 downside가 커 valuation support가 제한적."
    if revision_state == "deteriorating" and base_upside is not None and base_upside <= 5 and not structural_flag:
        primary_decision = "bearish"
        recommendation = "allow_with_limits"
        confidence = min(confidence, 0.48)
        notes = notes + " Street revision proxy 약화 + 기대수익률 제한."
    if peer_premium is not None and peer_premium >= 25 and base_upside is not None and base_upside <= 10 and not structural_flag:
        recommendation = "allow_with_limits"
        confidence = min(confidence, 0.52)
        notes = notes + " Peer premium이 높아 multiple compression 리스크 존재."
    if management_capital_allocation.get("capital_allocation_style") == "aggressive_financial_engineering":
        recommendation = "allow_with_limits"
        confidence = min(confidence, 0.52)
        notes = notes + " Debt-funded capital return 가능성으로 자본배분 질 점검 필요."

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
    consensus_snapshot = _build_consensus_snapshot(financials, as_of)
    catalyst_calendar = _build_catalyst_engine(financials, evidence_digest, as_of, catalyst_items=catalyst_items)
    valuation_anchor = _build_valuation_anchor(financials, vs)
    variant_view = _build_variant_view(financials, fs, vs, consensus_snapshot, evidence_digest, structural_flag)
    street_disagreement_layer = _build_street_disagreement_layer(financials, consensus_revision_layer, model_pack)
    expected_return_profile = _build_expected_return_profile(financials, model_pack, street_disagreement_layer)
    data_provenance = _build_data_provenance(
        financials,
        peer_comp_engine,
        consensus_revision_layer,
        catalyst_calendar,
        ownership_crowding,
        management_capital_allocation,
    )
    monitoring_triggers = _build_monitoring_triggers(
        consensus_revision_layer,
        catalyst_calendar,
        peer_comp_engine,
        management_capital_allocation,
        ownership_crowding,
        expected_return_profile,
        street_disagreement_layer,
    )
    if catalyst_calendar:
        first = catalyst_calendar[0]
        cat_type = str(first.get("type", "catalyst"))
        source_class = str(first.get("source_classification", "inferred"))
        if cat_type == "earnings":
            key_drivers = (
                key_drivers
                + [
                    f"Catalyst: earnings {first.get('days_to_event')}d ({first.get('date')})",
                ]
            )[:6]
            what_to_watch = (
                [
                    f"실적 촉매: {first.get('date')} / EPS est {first.get('eps_estimate')} / Revenue est {first.get('revenue_estimate')}",
                ]
                + what_to_watch
            )[:5]
        else:
            key_drivers = (
                key_drivers
                + [
                    f"Catalyst: {cat_type} ({source_class})",
                ]
            )[:6]
            what_to_watch = (
                [
                    f"촉매 확인: {first.get('source_title') or cat_type} / source={source_class}",
                ]
                + what_to_watch
            )[:5]
    if price_target_upside is not None:
        key_drivers = (key_drivers + [f"Street target upside={price_target_upside:.1f}%"])[:6]
    if revision_state in {"improving", "deteriorating"}:
        key_drivers = (key_drivers + [f"Street revision proxy={revision_state}"])[:6]
    if peer_premium is not None:
        key_drivers = (key_drivers + [f"Peer PE premium={peer_premium:.1f}%"])[:6]
    if management_capital_allocation.get("capital_allocation_style"):
        key_drivers = (
            key_drivers
            + [f"Capital allocation={management_capital_allocation.get('capital_allocation_style')}"]
        )[:6]
    target_dispersion = _safe_float(street_disagreement_layer.get("target_dispersion_pct"))
    if target_dispersion is not None:
        key_drivers = (key_drivers + [f"Street target dispersion={target_dispersion:.1f}%"])[:6]
    rr_ratio = _safe_float(expected_return_profile.get("reward_risk_ratio"))
    if rr_ratio is not None:
        key_drivers = (key_drivers + [f"Reward/risk={rr_ratio:.2f}x"])[:6]
    if data_provenance.get("quality") != "high":
        key_drivers = (
            key_drivers
            + [f"Provenance quality={data_provenance.get('quality')} ({data_provenance.get('raw_backed_layers')}/{data_provenance.get('total_layers')} raw-backed)"]
        )[:6]
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
    if monitoring_triggers:
        what_to_watch = (
            what_to_watch
            + [f"모니터링 트리거: {monitoring_triggers[0].get('name')} / {monitoring_triggers[0].get('trigger')}"]
        )[:5]

    open_questions = _default_open_questions(ticker, focus_areas, stretch_level)
    if consensus_snapshot.get("price_target_consensus") is None:
        open_questions = (
            open_questions
            + [
                {
                    "q": f"{ticker} sell-side target/consensus는 현재 가격과 얼마나 괴리되는가?",
                    "why": "expected return anchor 보강",
                    "kind": "valuation_context",
                    "priority": 2,
                    "recency_days": 30,
                }
            ]
        )[:5]
    if ownership_crowding.get("status") == "insufficient_data":
        open_questions = (
            open_questions
            + [
                {
                    "q": f"{ticker} 보유자 집중도/실제 incremental buyer는 누구인가?",
                    "why": "ownership/crowding 공백 보강",
                    "kind": "ownership_identity",
                    "priority": 2,
                    "recency_days": 30,
                }
            ]
        )[:5]
    decision_sensitivity = _default_decision_sensitivity(stretch_level, structural_flag, earnings_in_days)
    if base_upside is not None:
        decision_sensitivity = (
            decision_sensitivity
            + [
                {
                    "if": f"base target upside가 {base_upside:.1f}% 대비 추가 축소",
                    "then_change": "expected return 재산정 후 conviction 하향",
                    "impact": "medium",
                }
            ]
        )[:5]
    followups = _default_followups()
    react_trace = _default_react_trace(bool(evidence_digest))

    confidence = round(max(0.25, confidence - float(data_provenance.get("confidence_haircut") or 0.0)), 2)
    uncertainty = str(street_disagreement_layer.get("uncertainty_level", "medium")).lower()
    expected_quality = str(expected_return_profile.get("quality", "balanced")).lower()
    disagreement_signal = str(street_disagreement_layer.get("disagreement_signal", "aligned")).lower()
    if data_provenance.get("quality") == "low" and recommendation == "allow":
        recommendation = "allow_with_limits"
        notes = notes + " Raw-backed coverage가 약해 sizing을 보수적으로 제한."
    if uncertainty == "high":
        recommendation = "allow_with_limits"
        confidence = min(confidence, 0.52)
        notes = notes + " Street disagreement/coverage uncertainty가 높아 conviction을 보수적으로 반영."
    if expected_quality == "thin" and not structural_flag:
        recommendation = "allow_with_limits"
        confidence = min(confidence, 0.5)
        if primary_decision == "bullish":
            primary_decision = "neutral"
        notes = notes + " 기대수익 대비 하방 비대칭이 얕아 공격적 sizing을 제한."
    if disagreement_signal in {"street_too_optimistic", "street_above_model"} and not structural_flag:
        recommendation = "allow_with_limits"
        confidence = min(confidence, 0.5)
        notes = notes + " Street expectation과 내부 anchor 괴리가 커서 variant risk가 존재."

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
            "current_price": financials.get("current_price"),
            "pe_ratio": financials.get("pe_ratio"),
            "ps_ratio": financials.get("ps_ratio"),
            "revenue_growth_pct": financials.get("revenue_growth"),
            "roe_pct": financials.get("roe"),
            "debt_to_equity": financials.get("debt_to_equity"),
            "price_target_consensus": financials.get("price_target_consensus"),
            "price_target_upside_pct": financials.get("price_target_upside_pct"),
        },
        "quality_block": {"altman": altman, "coverage": coverage, "fcf_quality": fcf},
        "structural_risk_flag": structural_flag,
        "hard_red_flags": hard,
        "soft_flags": soft,
        "key_drivers": key_drivers,
        "what_to_watch": what_to_watch,
        "scenario_notes": scenario_notes,
        "evidence_digest": evidence_digest,
        "consensus_snapshot": consensus_snapshot,
        "consensus_revision_layer": consensus_revision_layer,
        "catalyst_calendar": catalyst_calendar,
        "valuation_anchor": valuation_anchor,
        "peer_comp_engine": peer_comp_engine,
        "street_disagreement_layer": street_disagreement_layer,
        "expected_return_profile": expected_return_profile,
        "model_pack": model_pack,
        "management_capital_allocation": management_capital_allocation,
        "ownership_crowding": ownership_crowding,
        "data_provenance": data_provenance,
        "monitoring_triggers": monitoring_triggers,
        "variant_view": variant_view,
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
