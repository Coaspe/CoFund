"""
tests/test_structural_risk_gate3.py — structural_risk_flag=True → reject
=========================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines.fundamental_engine import compute_structural_risk
from agents.fundamental_agent import fundamental_analyst_run


def test_altman_distress_triggers_structural_flag():
    """Altman Z < 1.81 → structural_risk_flag=True."""
    financials = {
        "total_assets": 100_000,
        "current_assets": 10_000,
        "current_liabilities": 30_000,
        "retained_earnings": 5_000,
        "ebit": 2_000,
        "market_cap": 50_000,
        "total_liabilities": 80_000,
        "revenue": 50_000,
        "interest_expense": 5_000,
        "net_debt": 60_000,
        "ebitda": 8_000,
        "free_cash_flow": -1_000,
    }
    result = compute_structural_risk(financials)
    assert result["structural_risk_flag"] is True, f"Altman distress should trigger flag, got {result}"
    assert any(f["code"] == "default_risk" for f in result["hard_red_flags"]), "Must have default_risk flag"


def test_fundamental_agent_rejects_on_structural_risk():
    """structural_risk_flag=True → recommendation=reject, confidence<=0.4."""
    financials = {
        "total_assets": 100_000,
        "current_assets": 10_000,
        "current_liabilities": 30_000,
        "retained_earnings": 5_000,
        "ebit": 2_000,
        "market_cap": 50_000,
        "total_liabilities": 80_000,
        "revenue": 50_000,
        "interest_expense": 5_000,
        "net_debt": 60_000,
        "ebitda": 8_000,
        "free_cash_flow": -1_000,
        "revenue_growth": 5.0,
        "pe_ratio": 15.0,
        "roe": 10.0,
        "debt_to_equity": 4.0,
    }
    output = fundamental_analyst_run("TEST", financials, sec_data=None)
    assert output["structural_risk_flag"] is True
    assert output["recommendation"] == "reject"
    assert output["confidence"] <= 0.4
    assert output["primary_decision"] == "avoid"


def test_going_concern_triggers_reject():
    """Going concern language → structural_risk_flag → reject."""
    financials = {
        "total_assets": 200_000, "revenue": 100_000,
        "revenue_growth": 15.0, "pe_ratio": 12.0,
    }
    sec_data = {"has_going_concern_language": True}
    output = fundamental_analyst_run("TEST", financials, sec_data=sec_data)
    assert output["structural_risk_flag"] is True
    assert output["recommendation"] == "reject"


def test_healthy_company_allows():
    """No hard flags → allow or allow_with_limits."""
    financials = {
        "total_assets": 500_000,
        "current_assets": 150_000,
        "current_liabilities": 50_000,
        "retained_earnings": 200_000,
        "ebit": 75_000,
        "market_cap": 2_000_000,
        "total_liabilities": 200_000,
        "revenue": 400_000,
        "interest_expense": 5_000,
        "net_debt": 50_000,
        "ebitda": 100_000,
        "free_cash_flow": 70_000,
        "revenue_growth": 15.0,
        "pe_ratio": 20.0,
        "roe": 25.0,
        "debt_to_equity": 0.7,
    }
    output = fundamental_analyst_run("TEST", financials)
    assert output["structural_risk_flag"] is False
    assert output["recommendation"] in ("allow", "allow_with_limits")


if __name__ == "__main__":
    test_altman_distress_triggers_structural_flag()
    print("✅ test_altman_distress_triggers_structural_flag PASSED")
    test_fundamental_agent_rejects_on_structural_risk()
    print("✅ test_fundamental_agent_rejects_on_structural_risk PASSED")
    test_going_concern_triggers_reject()
    print("✅ test_going_concern_triggers_reject PASSED")
    test_healthy_company_allows()
    print("✅ test_healthy_company_allows PASSED")
