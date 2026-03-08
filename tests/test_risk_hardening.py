"""
tests/test_risk_hardening.py — Phase F: Critical Bug Fix Coverage
=================================================================
Tests the 9 hardening items applied to the risk pipeline.
"""

import hashlib
import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Item 1: first_not_none — 0.0 weight bug
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_first_not_none_zero_is_valid():
    """0.0 must be returned as-is, not trigger fallback."""
    from schemas.common import first_not_none

    d = {"final_allocation_pct": 0.0, "final_weight": 0.05}
    assert first_not_none(d, ["final_allocation_pct", "final_weight"]) == 0.0


def test_first_not_none_none_skips():
    """None should fallback to next key."""
    from schemas.common import first_not_none

    d = {"final_allocation_pct": None, "final_weight": 0.07}
    assert first_not_none(d, ["final_allocation_pct", "final_weight"]) == 0.07


def test_first_not_none_all_none_returns_default():
    from schemas.common import first_not_none

    d = {"a": None}
    assert first_not_none(d, ["a", "b"], default=0.03) == 0.03


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Item 2: compute_signed_weight — SHORT exposure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_signed_weight_short():
    """SHORT + positive alloc → negative weight."""
    from schemas.common import compute_signed_weight

    assert compute_signed_weight("SHORT", 0.10) == -0.10


def test_signed_weight_long():
    """LONG → positive weight."""
    from schemas.common import compute_signed_weight

    assert compute_signed_weight("LONG", 0.15) == 0.15


def test_signed_weight_hold():
    """HOLD → 0.0 regardless of alloc."""
    from schemas.common import compute_signed_weight

    assert compute_signed_weight("HOLD", 0.15) == 0.0


def test_signed_weight_none_alloc():
    """None allocation → 0.0."""
    from schemas.common import compute_signed_weight

    assert compute_signed_weight("LONG", None) == 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Item 3: sha256 seed reproducibility
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_make_seed_stable():
    """_make_seed must return identical value for same run_id across calls."""
    from investment_team import _make_seed

    s1 = _make_seed({"run_id": "test-run-123"})
    s2 = _make_seed({"run_id": "test-run-123"})
    assert s1 == s2

    # Cross-check: manually compute sha256
    h = hashlib.sha256(b"test-run-123").digest()
    expected = int.from_bytes(h[:4], "big") % (2**31)
    assert s1 == expected


def test_make_seed_explicit_seed():
    """Explicit seed in state should override run_id."""
    from investment_team import _make_seed

    s = _make_seed({"run_id": "abc", "seed": 42})
    assert s == 42


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Item 5: Macro regime taxonomy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_canonical_regime_mapping():
    """Contraction → recession, expansion → expansion, unknown → normal."""
    from schemas.taxonomy import map_macro_regime_to_canonical

    assert map_macro_regime_to_canonical("contraction") == "recession"
    assert map_macro_regime_to_canonical("expansion") == "expansion"
    assert map_macro_regime_to_canonical("GOLDILOCKS") == "goldilocks"
    assert map_macro_regime_to_canonical("unknown_regime") == "normal"
    assert map_macro_regime_to_canonical("stagflation") == "stagflation"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Item 7: Risk determinism — Python decides, LLM doesn't override
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_compute_risk_decision_gate3_structural():
    """Gate3: structural risk flag → reject_local with weight=0.0."""
    from agents.risk_agent import compute_risk_decision

    payload = {
        "risk_limits": {"max_portfolio_cvar_1d": 0.015, "max_leverage": 2.0,
                        "max_net_exposure": 0.8, "max_gross_exposure": 2.0,
                        "max_single_name_weight": 0.15, "max_sector_weight": 0.35,
                        "max_hhi": 0.25, "max_quant_weight_anomaly": 0.20,
                        "conservative_fallback_weight": 0.03, "liquidity_days_warning": 5},
        "portfolio_risk_summary": {
            "component_var_by_ticker": {"AAPL": 0.003},
            "portfolio_cvar_1d": 0.005,
            "leverage_ratio": 0.1,
            "herfindahl_index": 1.0,
            "sector_exposure": {"Technology": 0.1},
            "total_net_exposure": 0.1,
        },
        "analyst_reports": {
            "macro": {"macro_regime": "expansion", "regime": "expansion"},
            "fundamental": {"risk_flags": [{"code": "default_risk", "severity": "critical"}]},
            "sentiment": {},
            "quant": {"decision": "LONG", "final_allocation_pct": 0.10},
            "_target_ticker": "AAPL",
        },
    }

    decision = compute_risk_decision(payload)
    aapl = decision["per_ticker_decisions"]["AAPL"]
    assert aapl["decision"] == "reject_local"
    assert aapl["final_weight"] == 0.0
    assert "default_risk" in aapl["flags"]


def test_compute_risk_decision_gate4_short_no_trigger():
    """Gate4: SHORT during recession should NOT trigger macro_headwind (it's a hedge)."""
    from agents.risk_agent import compute_risk_decision

    payload = {
        "risk_limits": {"max_portfolio_cvar_1d": 0.015, "max_leverage": 2.0,
                        "max_net_exposure": 0.8, "max_gross_exposure": 2.0,
                        "max_single_name_weight": 0.15, "max_sector_weight": 0.35,
                        "max_hhi": 0.25, "max_quant_weight_anomaly": 0.20,
                        "conservative_fallback_weight": 0.03, "liquidity_days_warning": 5},
        "portfolio_risk_summary": {
            "component_var_by_ticker": {"AAPL": 0.003},
            "portfolio_cvar_1d": 0.005,
            "leverage_ratio": 0.1,
            "herfindahl_index": 1.0,
            "sector_exposure": {"Technology": -0.12},
            "total_net_exposure": -0.12,
        },
        "analyst_reports": {
            "macro": {"macro_regime": "recession", "regime": "recession"},
            "fundamental": {"risk_flags": []},
            "sentiment": {},
            "quant": {"decision": "SHORT", "final_allocation_pct": 0.12},
            "_target_ticker": "AAPL",
        },
    }

    decision = compute_risk_decision(payload)
    aapl = decision["per_ticker_decisions"]["AAPL"]
    # SHORT during recession → no macro_headwind flag
    assert "macro_headwind" not in aapl["flags"]
    assert "strategy_regime_mismatch" not in decision["orchestrator_feedback"]["reasons"]
    # Weight should be negative (short)
    assert aapl["final_weight"] < 0


def test_compute_risk_decision_signed_weight_preserved():
    """Risk decision must preserve negative weight for SHORT positions."""
    from agents.risk_agent import compute_risk_decision

    payload = {
        "risk_limits": {"max_portfolio_cvar_1d": 0.015, "max_leverage": 2.0,
                        "max_net_exposure": 0.8, "max_gross_exposure": 2.0,
                        "max_single_name_weight": 0.15, "max_sector_weight": 0.35,
                        "max_hhi": 0.25, "max_quant_weight_anomaly": 0.20,
                        "conservative_fallback_weight": 0.03, "liquidity_days_warning": 5},
        "portfolio_risk_summary": {
            "component_var_by_ticker": {"TSLA": 0.003},
            "portfolio_cvar_1d": 0.005,
            "leverage_ratio": 0.08,
            "herfindahl_index": 1.0,
            "sector_exposure": {"Consumer Cyclical": -0.08},
            "total_net_exposure": -0.08,
        },
        "analyst_reports": {
            "macro": {"macro_regime": "expansion"},
            "fundamental": {"risk_flags": []},
            "sentiment": {},
            "quant": {"decision": "SHORT", "final_allocation_pct": 0.08},
            "_target_ticker": "TSLA",
        },
    }

    decision = compute_risk_decision(payload)
    # Single-ticker → 100% of component_var → Gate2 applies 0.7x → -0.08 * 0.7 = -0.056
    assert decision["per_ticker_decisions"]["TSLA"]["final_weight"] < 0
    assert "component_var_dominant" in decision["per_ticker_decisions"]["TSLA"]["flags"]


def test_compute_risk_decision_enforces_portfolio_mandate_caps():
    from agents.risk_agent import compute_risk_decision

    payload = {
        "risk_limits": {"max_portfolio_cvar_1d": 0.015, "max_leverage": 2.0,
                        "max_net_exposure": 0.8, "max_gross_exposure": 2.0,
                        "max_single_name_weight": 0.15, "max_sector_weight": 0.35,
                        "max_hhi": 0.25, "max_quant_weight_anomaly": 0.20,
                        "conservative_fallback_weight": 0.03, "liquidity_days_warning": 5},
        "portfolio_risk_summary": {
            "component_var_by_ticker": {"SPY": 0.01, "QQQ": 0.006, "TLT": 0.004},
            "portfolio_cvar_1d": 0.005,
            "leverage_ratio": 1.0,
            "herfindahl_index": 0.20,
            "sector_exposure": {"Broad Market": 0.8, "Fixed Income": 0.2},
            "total_net_exposure": 1.0,
        },
        "positions_proposed": {"SPY": 0.7, "QQQ": 0.2, "TLT": 0.1},
        "portfolio_mandate": {
            "constraints": {
                "allowed_tickers": ["SPY", "TLT"],
                "blocked_tickers": ["QQQ"],
                "max_single_name_weight": 0.5,
                "target_gross_exposure": 0.6,
                "target_net_exposure": 0.6,
            }
        },
        "analyst_reports": {
            "macro": {"macro_regime": "expansion", "regime": "expansion", "evidence": [1], "data_ok": True},
            "fundamental": {"risk_flags": [], "evidence": [1], "data_ok": True},
            "sentiment": {"evidence": [1], "data_ok": True},
            "quant": {"decision": "LONG", "final_allocation_pct": 0.10, "evidence": [1], "data_ok": True},
            "_target_ticker": "SPY",
        },
    }

    decision = compute_risk_decision(payload)
    per = decision["per_ticker_decisions"]

    assert per["QQQ"]["final_weight"] == 0.0
    assert "blocked_ticker" in per["QQQ"]["flags"]
    assert abs(per["SPY"]["final_weight"]) <= 0.5
    total_gross = sum(abs(v["final_weight"]) for v in per.values())
    assert total_gross <= 0.6001
    assert "mandate_violation" in decision["orchestrator_feedback"]["reasons"]


def test_risk_manager_node_enforces_portfolio_context_mandate(monkeypatch):
    from agents import risk_agent as risk
    from schemas.common import create_initial_state

    monkeypatch.setattr(risk, "_call_llm", lambda payload: risk.compute_risk_decision(payload))

    state = create_initial_state(user_request="시장 전망 + 헤지", mode="mock", seed=5)
    state["iteration_count"] = 1
    state["target_ticker"] = "SPY"
    state["positions_proposed"] = {"SPY": 0.7, "QQQ": 0.2, "TLT": 0.1}
    state["portfolio_context"] = {
        "allowed_tickers": ["SPY", "TLT"],
        "blocked_tickers": ["QQQ"],
        "max_single_name_weight": 0.5,
        "target_gross_exposure": 0.6,
    }
    state["orchestrator_directives"] = {
        "portfolio_mandate": {
            "applied": True,
            "constraints": {
                "allowed_tickers": ["SPY", "TLT"],
                "blocked_tickers": ["QQQ"],
                "max_single_name_weight": 0.5,
                "target_gross_exposure": 0.6,
            },
        }
    }
    state["macro_analysis"] = {"macro_regime": "expansion", "regime": "expansion", "evidence": [1], "data_ok": True}
    state["fundamental_analysis"] = {"sector": "Broad Market", "risk_flags": [], "evidence": [1], "data_ok": True}
    state["sentiment_analysis"] = {"evidence": [1], "data_ok": True}
    state["technical_analysis"] = {"decision": "LONG", "final_allocation_pct": 0.10, "evidence": [1], "data_ok": True}

    out = risk.risk_manager_node(state)
    positions_final = out["positions_final"]
    decision = out["risk_assessment"]["risk_decision"]

    assert positions_final["QQQ"] == 0.0
    assert positions_final["SPY"] <= 0.5
    assert sum(abs(v) for v in positions_final.values()) <= 0.6001
    assert "mandate_violation" in decision["orchestrator_feedback"]["reasons"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Item 8: Disagreement score
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_disagreement_score_agreement():
    """All bullish → 0.0 disagreement."""
    from schemas.common import compute_disagreement_score

    desks = {
        "macro": {"primary_decision": "bullish", "confidence": 0.8},
        "funda": {"primary_decision": "bullish", "confidence": 0.8},
    }
    assert compute_disagreement_score(desks) == 0.0


def test_disagreement_score_disagreement():
    """Bullish vs bearish → high disagreement."""
    from schemas.common import compute_disagreement_score

    desks = {
        "macro": {"primary_decision": "bullish", "confidence": 1.0},
        "funda": {"primary_decision": "bearish", "confidence": 1.0},
    }
    score = compute_disagreement_score(desks)
    assert score > 0.5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Item 4: Barrier stale desk tracking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_barrier_skipped_desk_gets_iteration():
    """Skipped desks must have status='skipped' and iteration_generated."""
    from investment_team import barrier_node

    state = {
        "completed_tasks": {"macro": True},
        "macro_analysis": {"summary": "done", "status": "ok"},
        "fundamental_analysis": {},
        "sentiment_analysis": {},
        "technical_analysis": {},
        "iteration_count": 1,
    }
    result = barrier_node(state)
    # fundamental was not completed → should be marked skipped
    funda = result.get("fundamental_analysis", {})
    assert funda.get("status") == "skipped"
    assert funda.get("iteration_generated") == 1
