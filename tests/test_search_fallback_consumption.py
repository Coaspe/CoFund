from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import investment_team
import agents.macro_agent as macro_agent
import agents.fundamental_agent as fundamental_agent
from schemas.common import create_initial_state


def _base_state(user_request: str, ticker: str) -> dict:
    state = create_initial_state(user_request=user_request, mode="mock", seed=42)
    state["target_ticker"] = ticker
    state["completed_tasks"] = {"macro": True, "fundamental": True, "sentiment": True, "quant": True}
    state["audit"] = {"research": {"web_queries_total": 0, "web_queries_by_ticker": {}}}
    return state


def test_tavily_research_evidence_is_consumed_by_macro_agent(monkeypatch):
    monkeypatch.setattr(macro_agent, "apply_llm_overlay_macro", lambda *args, **kwargs: {})
    state = _base_state("연준 헤드라인 영향 점검", "NVDA")
    state["_research_plan"] = [
        {
            "desk": "macro",
            "kind": "macro_headline_context",
            "ticker": "NVDA",
            "query": "NVDA fed rate cuts macro headline context",
            "priority": 1,
            "recency_days": 7,
            "max_items": 3,
        }
    ]

    def _fake_resolve(req, **kwargs):
        return (
            [
                {
                    "url": "https://www.reuters.com/world/us/fed-signals-slower-rate-cuts-2026-02-01/",
                    "title": "Fed signals slower rate cuts as inflation cools",
                    "published_at": "2026-02-01T00:00:00+00:00",
                    "snippet": "Fed signals slower rate cuts while markets reprice growth and semiconductor valuations after fresh macro data.",
                    "source": "reuters.com",
                    "hash": "raw-tavily-1",
                    "kind": req.get("kind", ""),
                    "desk": req.get("desk", ""),
                    "ticker": req.get("ticker", ""),
                    "trust_tier": 0.6,
                    "resolver_path": "tavily_fallback_macro",
                }
            ],
            "tavily_fallback_macro",
        )

    monkeypatch.setattr(investment_team, "_resolve_request_with_priority", _fake_resolve)
    research_out = investment_team.research_executor_node(state)

    rerun_state = dict(state)
    rerun_state.update(research_out)
    rerun_state["completed_tasks"] = {"macro": False, "fundamental": True, "sentiment": True, "quant": True}
    rerun_state["macro_analysis"] = {"primary_decision": "bullish", "macro_regime": "expansion"}
    rerun_state["_rerun_plan"] = {
        "selected_desks": ["macro"],
        "reasons": {"macro": ["impacted_requests=1"]},
        "executed_kinds": ["macro_headline_context"],
    }

    out = investment_team.macro_analyst_node(rerun_state)["macro_analysis"]

    assert out["evidence_digest"]
    assert out["evidence_digest"][0]["resolver_path"] == "tavily_fallback_macro"
    assert any("Evidence update:" in item for item in out.get("key_drivers", []))
    assert out.get("decision_change_log", {}).get("was_rerun") is True
    assert out.get("decision_change_log", {}).get("evidence_refs")


def test_exa_research_evidence_is_consumed_by_fundamental_agent(monkeypatch):
    monkeypatch.setattr(fundamental_agent, "apply_llm_overlay_fundamental", lambda *args, **kwargs: {})
    state = _base_state("밸류에이션 점검", "NVDA")
    state["_research_plan"] = [
        {
            "desk": "fundamental",
            "kind": "valuation_context",
            "ticker": "NVDA",
            "query": "NVDA valuation peers AI semiconductor margins",
            "priority": 1,
            "recency_days": 30,
            "max_items": 3,
        }
    ]

    def _fake_resolve(req, **kwargs):
        return (
            [
                {
                    "url": "https://www.wsj.com/finance/stocks/nvda-valuation-premium-ai-demand-2026-02-03",
                    "title": "Nvidia valuation premium holds as AI demand stays elevated",
                    "published_at": "2026-02-03T00:00:00+00:00",
                    "snippet": "Nvidia keeps a valuation premium over semiconductor peers as AI demand and margin durability remain above consensus.",
                    "source": "wsj.com",
                    "hash": "raw-exa-1",
                    "kind": req.get("kind", ""),
                    "desk": req.get("desk", ""),
                    "ticker": req.get("ticker", ""),
                    "trust_tier": 0.6,
                    "resolver_path": "exa_fallback_default",
                }
            ],
            "exa_fallback_default",
        )

    monkeypatch.setattr(investment_team, "_resolve_request_with_priority", _fake_resolve)
    research_out = investment_team.research_executor_node(state)

    rerun_state = dict(state)
    rerun_state.update(research_out)
    rerun_state["completed_tasks"] = {"macro": True, "fundamental": False, "sentiment": True, "quant": True}
    rerun_state["fundamental_analysis"] = {
        "primary_decision": "bullish",
        "recommendation": "allow",
        "analysis_mode": "equity",
    }
    rerun_state["_rerun_plan"] = {
        "selected_desks": ["fundamental"],
        "reasons": {"fundamental": ["impacted_requests=1"]},
        "executed_kinds": ["valuation_context"],
    }

    out = investment_team.fundamental_analyst_node(rerun_state)["fundamental_analysis"]

    assert out["evidence_digest"]
    assert out["evidence_digest"][0]["resolver_path"] == "exa_fallback_default"
    assert any("Evidence update:" in item for item in out.get("key_drivers", []))
    assert out.get("decision_change_log", {}).get("was_rerun") is True
    assert out.get("decision_change_log", {}).get("evidence_refs")
