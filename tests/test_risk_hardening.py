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


def test_compute_risk_decision_reads_allocator_guidance_gross_cap():
    from agents.risk_agent import compute_risk_decision

    payload = {
        "risk_limits": {"max_portfolio_cvar_1d": 0.015, "max_leverage": 2.0,
                        "max_net_exposure": 0.8, "max_gross_exposure": 2.0,
                        "max_single_name_weight": 0.15, "max_sector_weight": 0.35,
                        "max_hhi": 0.25, "max_quant_weight_anomaly": 0.20,
                        "conservative_fallback_weight": 0.03, "liquidity_days_warning": 5},
        "portfolio_risk_summary": {
            "component_var_by_ticker": {"SPY": 0.01, "TLT": 0.006, "GLD": 0.004},
            "portfolio_cvar_1d": 0.005,
            "leverage_ratio": 1.0,
            "herfindahl_index": 0.20,
            "sector_exposure": {"Broad Market": 0.7, "Fixed Income": 0.2, "Commodities": 0.1},
            "total_net_exposure": 1.0,
        },
        "positions_proposed": {"SPY": 0.6, "TLT": 0.25, "GLD": 0.15},
        "allocator_guidance": {
            "target_gross_exposure": 0.4,
            "single_name_cap": 0.3,
            "allocator_source": "orchestrator_book_allocator_v1",
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
    total_gross = sum(abs(v["final_weight"]) for v in decision["per_ticker_decisions"].values())

    assert total_gross <= 0.4001
    assert "mandate_violation" in decision["orchestrator_feedback"]["reasons"]


def test_compute_risk_decision_stress_and_liquidity_trigger_kill_switch():
    from agents.risk_agent import compute_risk_decision

    payload = {
        "risk_limits": {"max_portfolio_cvar_1d": 0.015, "max_leverage": 2.0,
                        "max_net_exposure": 0.8, "max_gross_exposure": 2.0,
                        "max_single_name_weight": 0.15, "max_sector_weight": 0.35,
                        "max_hhi": 0.25, "max_quant_weight_anomaly": 0.20,
                        "conservative_fallback_weight": 0.03, "liquidity_days_warning": 5},
        "portfolio_risk_summary": {
            "component_var_by_ticker": {"SPY": 0.02, "QQQ": 0.015, "TLT": 0.005},
            "portfolio_cvar_1d": 0.022,
            "leverage_ratio": 1.3,
            "herfindahl_index": 0.41,
            "sector_exposure": {"Broad Market": 0.6, "Technology": 0.25, "Fixed Income": 0.15},
            "total_net_exposure": 1.0,
            "total_gross_exposure": 1.0,
            "liquidity_score_by_ticker": {"SPY": 8.0, "QQQ": 6.0, "TLT": 4.0},
        },
        "positions_proposed": {"SPY": 0.6, "QQQ": 0.25, "TLT": 0.15},
        "positions_metadata": {
            "SPY": {"sector": "Broad Market"},
            "QQQ": {"sector": "Technology"},
            "TLT": {"sector": "Fixed Income"},
        },
        "event_calendar": [
            {"ticker": "__GLOBAL__", "status": "imminent", "priority": 1, "confirmed": True, "type": "fomc"},
            {"ticker": "SPY", "status": "triggered", "priority": 1, "confirmed": True, "type": "macro_monitor"},
        ],
        "monitoring_actions": {
            "risk_refresh_required": True,
            "selected_desks": ["macro", "quant"],
            "reason": "new_triggered_events",
        },
        "analyst_reports": {
            "macro": {
                "macro_regime": "recession",
                "regime": "recession",
                "evidence": [1],
                "data_ok": True,
                "raw_market_inputs": {"vix_level": 31.0},
            },
            "fundamental": {"risk_flags": [], "evidence": [1], "data_ok": True},
            "sentiment": {"evidence": [1], "data_ok": True},
            "quant": {"decision": "LONG", "final_allocation_pct": 0.20, "evidence": [1], "data_ok": True},
            "_target_ticker": "SPY",
        },
    }

    decision = compute_risk_decision(payload)
    kill_switch = decision["portfolio_actions"]["kill_switch"]
    escalation = decision["portfolio_actions"]["escalation"]

    assert decision["stress_test_summary"]["severity"] in {"high", "critical"}
    assert decision["liquidity_risk"]["severity"] == "critical"
    assert kill_switch["active"] is True
    assert escalation["freeze_new_risk"] is True
    assert "stress_test_breach" in decision["orchestrator_feedback"]["reasons"]
    assert "liquidity_stress" in decision["orchestrator_feedback"]["reasons"]
    assert "kill_switch_active" in decision["orchestrator_feedback"]["reasons"]
    total_gross = sum(abs(v["final_weight"]) for v in decision["per_ticker_decisions"].values())
    assert total_gross <= kill_switch["target_gross_exposure"] + 1e-6


def test_risk_manager_node_includes_escalation_and_risk_objects(monkeypatch):
    from agents import risk_agent as risk
    from schemas.common import create_initial_state

    monkeypatch.setattr(risk, "_call_llm", lambda payload: risk.compute_risk_decision(payload))

    state = create_initial_state(user_request="시장 리스크 점검", mode="mock", seed=11)
    state["iteration_count"] = 1
    state["target_ticker"] = "SPY"
    state["positions_proposed"] = {"SPY": 0.55, "QQQ": 0.25, "TLT": 0.20}
    state["event_calendar"] = [
        {"ticker": "__GLOBAL__", "status": "imminent", "priority": 1, "confirmed": True, "type": "fomc"},
    ]
    state["monitoring_actions"] = {
        "risk_refresh_required": True,
        "selected_desks": ["macro"],
        "reason": "new_triggered_events",
    }
    state["macro_analysis"] = {
        "macro_regime": "recession",
        "regime": "recession",
        "evidence": [1],
        "data_ok": True,
        "raw_market_inputs": {"vix_level": 28.0},
    }
    state["fundamental_analysis"] = {"sector": "Broad Market", "risk_flags": [], "evidence": [1], "data_ok": True}
    state["sentiment_analysis"] = {"evidence": [1], "data_ok": True}
    state["technical_analysis"] = {"decision": "LONG", "final_allocation_pct": 0.12, "evidence": [1], "data_ok": True}

    out = risk.risk_manager_node(state)
    decision = out["risk_assessment"]["risk_decision"]

    assert "stress_test_summary" in decision
    assert "liquidity_risk" in decision
    assert decision["portfolio_actions"]["escalation"]["status"] == "ok"
    assert "macro" in decision["portfolio_actions"]["escalation"]["rerun_desks"]


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
