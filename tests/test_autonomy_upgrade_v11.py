"""
tests/test_autonomy_upgrade_v11.py
=================================
Autonomy Upgrade v1.1 regression tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agents.macro_agent as macro_agent
from agents.fundamental_agent import fundamental_analyst_run
from agents.macro_agent import macro_analyst_run
from agents.sentiment_agent import sentiment_analyst_run
from investment_team import (
    create_initial_state,
    macro_analyst_node,
    fundamental_analyst_node,
    sentiment_analyst_node,
    quant_analyst_node,
    research_router_node,
)


def _base_state() -> dict:
    state = create_initial_state(user_request="NVDA 과열인지 점검", mode="mock", seed=42)
    state["target_ticker"] = "NVDA"
    state["iteration_count"] = 1
    state["completed_tasks"] = {"macro": False, "fundamental": False, "sentiment": False, "quant": False}
    state["orchestrator_directives"] = {
        "desk_tasks": {
            "macro": {"horizon_days": 21, "focus_areas": ["금리", "고용"]},
            "fundamental": {"horizon_days": 120, "focus_areas": ["밸류에이션", "실적 모멘텀"]},
            "sentiment": {"horizon_days": 10, "focus_areas": ["이벤트", "포지셔닝"]},
            "quant": {"horizon_days": 14, "risk_budget": "Conservative", "focus_areas": ["CVaR", "Z-score"]},
        }
    }
    return state


def test_desk_task_propagation_to_nodes():
    state = _base_state()

    macro_out = macro_analyst_node(state)["macro_analysis"]
    funda_out = fundamental_analyst_node(state)["fundamental_analysis"]
    senti_out = sentiment_analyst_node(state)["sentiment_analysis"]
    quant_out = quant_analyst_node(state)["technical_analysis"]

    assert macro_out["horizon_days"] == 21
    assert macro_out["focus_areas"] == ["금리", "고용"]

    assert funda_out["horizon_days"] == 120
    assert funda_out["focus_areas"] == ["밸류에이션", "실적 모멘텀"]

    assert senti_out["horizon_days"] == 10
    assert senti_out["focus_areas"] == ["이벤트", "포지셔닝"]

    assert quant_out["horizon_days"] == 14
    assert quant_out["focus_areas"] == ["CVaR", "Z-score"]
    assert quant_out["risk_budget"] == "Conservative"


def test_desks_emit_autonomy_fields_without_llm():
    macro = macro_analyst_run("NVDA", {"yield_curve_spread": -0.2, "hy_oas": 450}, focus_areas=["금리"])
    funda = fundamental_analyst_run("NVDA", {"pe_ratio": 48.0, "revenue_growth": 22.0}, focus_areas=["밸류에이션"])
    senti = sentiment_analyst_run("NVDA", {"vix_level": 25, "news_sentiment_score": 0.2}, focus_areas=["이벤트"])

    for out in (macro, funda, senti):
        assert isinstance(out.get("open_questions"), list) and len(out["open_questions"]) >= 1
        assert isinstance(out.get("decision_sensitivity"), list) and len(out["decision_sensitivity"]) >= 1
        assert isinstance(out.get("followups"), list) and len(out["followups"]) >= 1
        assert "evidence_digest" in out


def test_macro_missing_checks_do_not_treat_zero_as_missing():
    reqs = macro_agent._generate_evidence_requests(
        "NVDA",
        axes={"growth": {"score": 0}},
        ron={"risk_on_off": "risk_off", "tail_risk_warning": False},
        features={"macro_regime": "expansion"},
        indicators={"yield_curve_spread": 0.0},
        focus_areas=[],
    )

    assert not any("yield curve" in str(req.get("rationale", "")).lower() for req in reqs)


def test_etf_fundamental_requests_only_when_context_is_missing():
    out = fundamental_analyst_run(
        "SPY",
        {
            "sector": "ETF",
            "holdings_top10_weight_pct": 48.0,
            "sector_weights": {"technology": 0.31},
            "factor_exposures": {"beta": 1.0},
            "index_forward_pe": 21.5,
            "net_flow_1m": 1250000,
            "tracking_error": 0.001,
            "expense_ratio": 0.0009,
            "liquidity_score": 0.95,
        },
        asset_type="ETF",
    )

    assert out.get("evidence_requests", []) == []


def test_sentiment_no_articles_fallback_avoids_etf_specific_query_for_equity():
    out = sentiment_analyst_run("NVDA", {"vix_level": 25, "news_sentiment_score": 0.2})
    queries = [str(req.get("query", "")).lower() for req in out.get("evidence_requests", [])]
    kinds = [str(req.get("kind", "")).lower() for req in out.get("evidence_requests", [])]

    assert not any("etf flow creation redemption" in query for query in queries)
    assert "press_release_or_ir" in kinds


def test_research_router_seeds_when_requests_are_sparse():
    state = _base_state()
    state["macro_analysis"] = {"open_questions": []}
    state["fundamental_analysis"] = {"open_questions": []}
    state["sentiment_analysis"] = {"open_questions": []}
    state["technical_analysis"] = {}
    state["evidence_requests"] = []
    state["research_round"] = 0
    state["evidence_store"] = {}
    state["audit"] = {"research": {"web_queries_total": 0, "web_queries_by_ticker": {}}}

    out = research_router_node(state)
    kinds = {req.get("kind") for req in out.get("evidence_requests", [])}

    assert "press_release_or_ir" in kinds
    assert "macro_headline_context" in kinds
    assert "ownership_identity" in kinds
    assert "valuation_context" in kinds
    assert len(out.get("evidence_requests", [])) >= 4


def test_research_router_converts_open_questions_to_requests():
    state = _base_state()
    state["macro_analysis"] = {
        "open_questions": [
            {
                "q": "최근 금리 헤드라인이 반도체 밸류에 어떤 영향인가?",
                "why": "결론 민감도 높음",
                "kind": "macro_headline_context",
                "priority": 1,
                "recency_days": 7,
            }
        ]
    }
    state["fundamental_analysis"] = {"open_questions": []}
    state["sentiment_analysis"] = {"open_questions": []}
    state["technical_analysis"] = {}
    state["evidence_requests"] = []
    state["evidence_store"] = {}
    state["research_round"] = 0
    state["audit"] = {"research": {"web_queries_total": 0, "web_queries_by_ticker": {}}}

    out = research_router_node(state)
    queries = [req.get("query", "") for req in out.get("evidence_requests", [])]

    assert any("금리" in q for q in queries)


def test_evidence_digest_is_consumed_on_desk_run(monkeypatch):
    monkeypatch.setattr(macro_agent, "apply_llm_overlay_macro", lambda *args, **kwargs: {})
    state = {
        "evidence_store": {
            "h1": {
                "title": "Fed release hints slower cuts",
                "url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260101a.htm",
                "published_at": "2026-01-01T00:00:00+00:00",
                "trust_tier": 1.0,
                "kind": "macro_headline_context",
                "ticker": "NVDA",
                "resolver_path": "official_release",
            }
        }
    }
    out = macro_analyst_run("NVDA", {"yield_curve_spread": -0.1, "hy_oas": 410}, state=state)

    assert len(out.get("evidence_digest", [])) >= 1
    assert any("Evidence update:" in d for d in out.get("key_drivers", []))
