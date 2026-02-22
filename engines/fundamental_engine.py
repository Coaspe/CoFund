"""
engines/fundamental_engine.py — Fundamental 순수 연산 엔진
==========================================================
CHANGELOG:
  v1.0 (2026-02-22) — 신규 생성.

Altman Z-Score, Interest Coverage, Debt/EBITDA, FCF quality, structural risk flag.
데이터 수집 금지: 입력=재무제표 dict, 출력=분석 JSON.
"""

from __future__ import annotations

from typing import Any, Dict, List


def compute_altman_z(financials: dict) -> dict:
    """
    Altman Z-Score (제조업 기준, 비제조업 대체 지표 포함).

    입력 필드 (있는 것만 사용, 없으면 None):
      total_assets, current_assets, current_liabilities,
      retained_earnings, ebit, market_cap, total_liabilities,
      revenue

    Returns:
        {"z_score": float, "bucket": str, "altman_red_flag": bool, "error": str|None}
    """
    ta = financials.get("total_assets")
    ca = financials.get("current_assets")
    cl = financials.get("current_liabilities")
    re = financials.get("retained_earnings")
    ebit = financials.get("ebit")
    mc = financials.get("market_cap")
    tl = financials.get("total_liabilities")
    rev = financials.get("revenue")

    if ta is None or ta <= 0:
        return {"z_score": None, "bucket": "unknown", "altman_red_flag": False, "error": "total_assets 미제공"}

    try:
        x1 = ((ca or 0) - (cl or 0)) / ta
        x2 = (re or 0) / ta
        x3 = (ebit or 0) / ta
        x4 = (mc or 0) / max(tl or 1, 1)
        x5 = (rev or 0) / ta

        z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

        if z < 1.81:
            bucket = "distress"
        elif z < 2.99:
            bucket = "grey"
        else:
            bucket = "safe"

        return {
            "z_score": round(z, 4),
            "bucket": bucket,
            "altman_red_flag": z < 1.81,
            "error": None,
        }
    except Exception as exc:
        return {"z_score": None, "bucket": "unknown", "altman_red_flag": False, "error": str(exc)}


def compute_coverage_ratios(financials: dict) -> dict:
    """
    Interest Coverage + Debt/EBITDA.

    Returns:
        {
            "interest_coverage": float, "icr_bucket": str, "interest_red_flag": bool,
            "debt_to_ebitda": float, "leverage_bucket": str, "leverage_red_flag": bool,
        }
    """
    ebit = financials.get("ebit")
    interest = financials.get("interest_expense")
    net_debt = financials.get("net_debt")
    ebitda = financials.get("ebitda")

    # Interest Coverage
    if ebit is not None and interest is not None and abs(interest) > 0:
        icr = ebit / abs(interest)
        icr_bucket = "strong" if icr > 5 else ("adequate" if icr > 2 else ("weak" if icr > 1 else "distress"))
        interest_flag = icr < 1.5
    else:
        icr, icr_bucket, interest_flag = None, "unknown", False

    # Debt / EBITDA
    if net_debt is not None and ebitda is not None and ebitda > 0:
        d_e = net_debt / ebitda
        lev_bucket = "low" if d_e < 2 else ("moderate" if d_e < 4 else ("high" if d_e < 6 else "extreme"))
        lev_flag = d_e > 5
    else:
        d_e, lev_bucket, lev_flag = None, "unknown", False

    return {
        "interest_coverage": round(icr, 2) if icr is not None else None,
        "icr_bucket": icr_bucket,
        "interest_red_flag": interest_flag,
        "debt_to_ebitda": round(d_e, 2) if d_e is not None else None,
        "leverage_bucket": lev_bucket,
        "leverage_red_flag": lev_flag,
    }


def compute_fcf_quality(financials: dict) -> dict:
    """FCF margin + negative streak."""
    fcf = financials.get("free_cash_flow")
    rev = financials.get("revenue")
    fcf_history = financials.get("fcf_history", [])  # list of annual FCF

    fcf_margin = None
    if fcf is not None and rev is not None and rev > 0:
        fcf_margin = round(fcf / rev * 100, 2)

    neg_streak = sum(1 for x in fcf_history if x is not None and x < 0)

    return {
        "fcf_margin_pct": fcf_margin,
        "fcf_negative_streak_years": neg_streak,
        "fcf_quality_flag": neg_streak >= 3 or (fcf_margin is not None and fcf_margin < -5),
    }


def compute_sec_text_flags(sec_data: dict) -> dict:
    """SEC 공시 텍스트 플래그 (데이터 있으면 파싱, 없으면 False)."""
    return {
        "has_going_concern_language": sec_data.get("has_going_concern_language", False),
        "has_material_weakness_icfr": sec_data.get("has_material_weakness_icfr", False),
        "has_recent_restatement": sec_data.get("has_recent_restatement", False),
        "regulatory_investigation_flag": sec_data.get("regulatory_investigation_flag", False),
    }


def compute_structural_risk(financials: dict, sec_data: dict | None = None) -> dict:
    """
    모든 펀더멘털 플래그를 종합하여 structural_risk_flag 판별.

    Risk Manager Gate3에 직접 공급 가능한 형태로 반환.
    """
    altman = compute_altman_z(financials)
    coverage = compute_coverage_ratios(financials)
    fcf = compute_fcf_quality(financials)
    sec_flags = compute_sec_text_flags(sec_data or {})

    hard_flags: List[dict] = []
    soft_flags: List[dict] = []

    # Hard flags → structural_risk_flag = True
    if sec_flags["has_going_concern_language"]:
        hard_flags.append({"code": "going_concern", "severity": "critical", "description": "Going concern opinion in audit"})
    if sec_flags["has_material_weakness_icfr"]:
        hard_flags.append({"code": "material_weakness", "severity": "critical", "description": "Material weakness in internal controls"})
    if sec_flags["has_recent_restatement"]:
        hard_flags.append({"code": "restatement", "severity": "high", "description": "Recent financial restatement"})
    if sec_flags["regulatory_investigation_flag"]:
        hard_flags.append({"code": "regulatory_action", "severity": "high", "description": "Regulatory investigation flag"})
    if altman["altman_red_flag"]:
        hard_flags.append({"code": "default_risk", "severity": "high", "description": f"Altman Z={altman['z_score']} < 1.81 (distress zone)"})
    if coverage["interest_red_flag"] and coverage["leverage_red_flag"]:
        hard_flags.append({"code": "default_risk", "severity": "high", "description": f"ICR={coverage['interest_coverage']}, D/EBITDA={coverage['debt_to_ebitda']}"})

    # Soft flags
    if coverage["interest_red_flag"] and not coverage["leverage_red_flag"]:
        soft_flags.append({"code": "low_coverage", "severity": "medium", "description": f"ICR={coverage['interest_coverage']}"})
    if coverage["leverage_red_flag"] and not coverage["interest_red_flag"]:
        soft_flags.append({"code": "high_leverage", "severity": "medium", "description": f"D/EBITDA={coverage['debt_to_ebitda']}"})
    if altman["bucket"] == "grey":
        soft_flags.append({"code": "grey_zone_altman", "severity": "medium", "description": f"Altman Z={altman['z_score']} grey zone"})
    if fcf["fcf_quality_flag"]:
        soft_flags.append({"code": "fcf_quality_concern", "severity": "medium", "description": f"FCF streak={fcf['fcf_negative_streak_years']}y"})

    structural_risk_flag = len(hard_flags) > 0

    # flat risk_flags codes for Risk Manager Gate3
    risk_flag_codes = [f["code"] for f in hard_flags] + [f["code"] for f in soft_flags]

    return {
        "altman": altman,
        "coverage": coverage,
        "fcf_quality": fcf,
        "sec_text_flags": sec_flags,
        "structural_risk_flag": structural_risk_flag,
        "hard_red_flags": hard_flags,
        "soft_flags": soft_flags,
        "risk_flag_codes": risk_flag_codes,
    }
