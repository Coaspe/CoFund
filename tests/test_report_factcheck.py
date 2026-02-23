"""
tests/test_report_factcheck.py — T3: Report fact-check validation
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from validators.factcheck import (
    FactCheckError,
    validate_risk_narrative,
    validate_orchestrator_output,
    validate_report_markdown,
)


# ── validate_risk_narrative ───────────────────────────────────────────────────

def test_narrative_rejects_digits():
    state = {"universe": ["AAPL"], "target_ticker": "AAPL"}
    bad = "포트폴리오 CVaR 5% 초과로 비중을 0.08에서 축소했습니다."
    with pytest.raises(FactCheckError, match="숫자"):
        validate_risk_narrative(state, bad)


def test_narrative_rejects_percent_sign():
    state = {"universe": ["AAPL"], "target_ticker": "AAPL"}
    # Only % with no digits — digit check must NOT fire first
    bad = "리스크 초과로 비중을 축소했습니다 (%)." 
    with pytest.raises(FactCheckError, match="%"):
        validate_risk_narrative(state, bad)


def test_narrative_rejects_ticker():
    state = {"universe": ["AAPL", "MSFT"], "target_ticker": "AAPL"}
    bad = "해당 종목의 구조적 리스크로 인해 AAPL의 비중을 제외했습니다."
    with pytest.raises(FactCheckError):  # any FactCheckError (digit or ticker)
        validate_risk_narrative(state, bad)


def test_narrative_passes_clean_text():
    state = {"universe": ["AAPL"], "target_ticker": "AAPL"}
    clean = "포트폴리오 리스크 한도를 초과하여 신규 롱 포지션을 방어적으로 축소했습니다."
    # should not raise
    validate_risk_narrative(state, clean)


# ── validate_orchestrator_output ──────────────────────────────────────────────

def test_orchestrator_rejects_unauthorized_ticker():
    state = {"universe": ["AAPL", "MSFT"]}
    orch = {"investment_brief": {"target_universe": ["AAPL", "GOOGL"]}}
    with pytest.raises(FactCheckError, match="GOOGL"):
        validate_orchestrator_output(state, orch)


def test_orchestrator_allows_known_tickers():
    state = {"universe": ["AAPL", "MSFT"]}
    orch = {"investment_brief": {"target_universe": ["AAPL"]}}
    validate_orchestrator_output(state, orch)  # no raise


def test_orchestrator_skips_empty_universe():
    state = {"universe": []}
    orch = {"investment_brief": {"target_universe": ["ANYTHING"]}}
    validate_orchestrator_output(state, orch)  # no raise — universe empty


# ── validate_report_markdown (fallback) ──────────────────────────────────────

def test_report_fallback_on_wrong_weight():
    """보고서에 잘못된 비중이 있을 때 템플릿 fallback 반환."""
    state = {
        "positions_final": {"AAPL": 0.08},
        "as_of": "2024-01-01T00:00:00Z",
        "risk_manager_decision": {
            "per_ticker_decisions": {"AAPL": {"decision": "approve", "final_weight": 0.08}},
            "orchestrator_feedback": {"required": False, "detail": "All gates passed."},
        },
        "target_ticker": "AAPL",
        "universe": ["AAPL"],
    }
    # 보고서에 잘못된 비중(50%)이 포함됨
    bad_report = "AAPL의 최종 비중은 50.00%로 결정되었습니다."
    result = validate_report_markdown(state, bad_report, llm=None)
    # 템플릿으로 fallback되어야 하고, 올바른 비중이 포함되어야 함
    assert "AAPL" in result
    assert "8.00%" in result or "0.08" in result or "템플릿" in result


def test_report_passes_correct_weight():
    """보고서에 올바른 비중이 있으면 원본 반환."""
    state = {"positions_final": {"AAPL": 0.08}}
    good_report = "AAPL 최종 비중: 8.0% (approve)"
    result = validate_report_markdown(state, good_report, llm=None)
    assert result == good_report
