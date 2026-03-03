from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import investment_team
from agents.fundamental_agent import fundamental_analyst_run
from agents.report_agent import report_writer_node
from schemas.common import create_initial_state


def test_ac1_orchestrator_universe_and_execution_mode(monkeypatch):
    state = create_initial_state(user_request="미국 시장 단기/장기 전망", mode="mock", seed=7)

    def _fake_orch(_state):
        return {
            "target_ticker": "SPY",
            "analysis_tasks": ["macro_analysis", "fundamental_analysis", "sentiment_analysis", "technical_analysis"],
            "iteration_count": 1,
            "orchestrator_directives": {
                "action_type": "initial_delegation",
                "intent": "market_outlook",
                "investment_brief": {
                    "rationale": "시장 전망 점검",
                    "target_universe": ["SPY", "QQQ", "GLD", "TLT", "XLE"],
                },
            },
        }

    monkeypatch.setattr(investment_team, "_orch_node_impl", _fake_orch)
    monkeypatch.setattr(investment_team, "HAS_ORCHESTRATOR", True)
    out = investment_team.orchestrator_node(state)

    assert len(out.get("universe", [])) >= 2
    assert out.get("analysis_execution_mode", "").startswith("B_")
    assert out.get("target_ticker") == "SPY"


def test_ac2_etf_fundamental_no_insider_form4_requests():
    out = fundamental_analyst_run(
        "SPY",
        {"sector": "ETF"},
        asset_type="ETF",
        focus_areas=["섹터 비중", "플로우"],
    )
    reqs = out.get("evidence_requests", [])
    assert reqs, "ETF 모드에서도 최소 1개 evidence request 필요"
    for req in reqs:
        kind = str(req.get("kind", "")).lower()
        query = str(req.get("query", "")).lower()
        assert kind not in {"ownership_identity", "sec_filing"}
        assert "insider" not in query
        assert "form 4" not in query


def test_ac3_evidence_store_canonical_dedupe(monkeypatch):
    state = create_initial_state(user_request="macro shock", mode="mock", seed=9)
    state["_research_plan"] = [
        {
            "desk": "macro",
            "kind": "macro_headline_context",
            "ticker": "SPY",
            "query": "US Iran airstrike impact WTI Brent SPY",
            "priority": 1,
            "recency_days": 7,
            "max_items": 5,
        }
    ]
    state["completed_tasks"] = {"macro": True, "fundamental": True, "sentiment": True, "quant": True}

    def _fake_resolve(req, **kwargs):
        return (
            [
                {
                    "url": "https://example.com/news?id=1&utm_source=x",
                    "title": "Iran airstrike lifts Brent",
                    "published_at": "2026-03-01T00:00:00+00:00",
                    "snippet": "US Iran airstrike impact on WTI Brent and SPY.",
                    "source": "example.com",
                    "hash": "raw1",
                    "kind": req.get("kind", ""),
                    "desk": req.get("desk", ""),
                    "ticker": req.get("ticker", ""),
                    "trust_tier": 0.6,
                },
                {
                    "url": "https://example.com/news?id=1",
                    "title": "Iran airstrike lifts Brent",
                    "published_at": "2026-03-01T00:00:00+00:00",
                    "snippet": "US Iran airstrike impact on WTI Brent and SPY.",
                    "source": "example.com",
                    "hash": "raw2",
                    "kind": req.get("kind", ""),
                    "desk": req.get("desk", ""),
                    "ticker": req.get("ticker", ""),
                    "trust_tier": 0.6,
                },
            ],
            "test",
        )

    monkeypatch.setattr(investment_team, "_resolve_request_with_priority", _fake_resolve)
    out = investment_team.research_executor_node(state)
    urls = [item.get("canonical_url") for item in out.get("evidence_store", {}).values()]
    assert len(urls) == 1
    assert len(set(urls)) == 1


def test_ac4_report_cvar_breach_claim_requires_flag():
    state = create_initial_state(user_request="SPY 전망", mode="mock", seed=1)
    state["target_ticker"] = "SPY"
    state["universe"] = ["SPY", "GLD"]
    state["intent"] = "market_outlook"
    state["output_language"] = "ko"
    state["technical_analysis"] = {"decision": "HOLD", "final_allocation_pct": 0.0, "llm_decision": {"cot_reasoning": "signal mixed"}}
    state["risk_assessment"] = {
        "grade": "Low",
        "risk_decision": {
            "per_ticker_decisions": {
                "SPY": {"decision": "approve", "final_weight": 0.0, "flags": [], "rationale_short": "정책상 유지"}
            }
        },
        "risk_payload": {"portfolio_risk_summary": {"portfolio_cvar_1d": 0.0}, "risk_limits": {"max_portfolio_cvar_1d": 0.015}},
    }
    report = report_writer_node(state)["final_report"]
    assert "CVaR 한도 이슈: **없음**" in report
    assert "CVaR 한도 이슈: **있음**" not in report


def test_ac5_korean_request_produces_korean_report():
    state = create_initial_state(user_request="미국 증시 전망을 한국어로 정리해줘", mode="mock", seed=3)
    state["target_ticker"] = "SPY"
    state["technical_analysis"] = {"decision": "HOLD", "final_allocation_pct": 0.0, "llm_decision": {"cot_reasoning": "혼조"}}
    state["risk_assessment"] = {"risk_decision": {"per_ticker_decisions": {"SPY": {"flags": [], "decision": "approve", "final_weight": 0.0}}}}
    report = report_writer_node(state)["final_report"]
    assert re.search(r"[가-힣]", report)


def test_ac6_market_outlook_quant_uses_event_regime_indicators():
    state = create_initial_state(user_request="시장 전망 점검", mode="mock", seed=42)
    state["target_ticker"] = "SPY"
    state["intent"] = "market_outlook"
    state["asset_type_by_ticker"] = {"SPY": "ETF"}
    state["completed_tasks"] = {"macro": True, "fundamental": True, "sentiment": True, "quant": False}

    out = investment_team.quant_analyst_node(state)["technical_analysis"]
    assert out.get("analysis_mode") == "event_regime"
    assert any(k in out.get("quant_indicators", []) for k in ("event_return_5d", "trend_20d", "vol_shift_20d_vs_60d"))
    assert "Event/Regime" in str((out.get("llm_decision") or {}).get("cot_reasoning", ""))


def test_rerun_decision_change_log_records_unchanged_with_evidence_refs():
    output = {
        "primary_decision": "neutral",
        "macro_regime": "normal",
        "evidence_digest": [{"hash": "abc123", "url": "https://example.com/a"}],
    }
    prev_output = {"primary_decision": "neutral", "macro_regime": "normal"}
    state = {"evidence_store": {}}
    logged = investment_team._attach_decision_change_log(
        desk="macro",
        output=output,
        state=state,
        prev_output=prev_output,
        rerun_reason="rerun selected (kinds=macro_headline_context; top-k rerun)",
        ticker="SPY",
    )
    change_log = logged.get("decision_change_log", {})
    assert change_log.get("was_rerun") is True
    assert change_log.get("changed") is False
    assert change_log.get("status") == "unchanged_after_evidence_review"
    refs = change_log.get("evidence_refs", [])
    assert refs and refs[0].get("url") == "https://example.com/a"


def test_ac_b1_b_mode_hedge_lite_populates_candidates():
    state = create_initial_state(user_request="이벤트 리스크 헤지 점검", mode="mock", seed=42)
    state["analysis_execution_mode"] = "B_main_plus_hedge_lite"
    state["target_ticker"] = "SPY"
    state["intent"] = "event_risk"
    state["universe"] = ["SPY", "QQQ", "GLD", "TLT", "XLE"]

    out = investment_team.hedge_lite_builder_node(state)
    hedge_lite = out.get("hedge_lite", {})

    assert hedge_lite, "B 모드에서는 hedge_lite 산출물이 필요함"
    hedges = hedge_lite.get("hedges", {})
    assert isinstance(hedges, dict)
    for ticker in state["universe"][1:]:
        assert ticker in hedges
        row = hedges[ticker]
        assert isinstance(row, dict)
        assert ("score" in row) or ("status" in row)


def test_ac_b2_positions_proposed_include_main_plus_hedge_and_sum_to_one():
    state = create_initial_state(user_request="시장 전망 + 헤지", mode="mock", seed=77)
    state["analysis_execution_mode"] = "B_main_plus_hedge_lite"
    state["target_ticker"] = "SPY"
    state["intent"] = "market_outlook"
    state["universe"] = ["SPY", "QQQ", "GLD", "TLT", "XLE"]

    out = investment_team.hedge_lite_builder_node(state)
    weights = out.get("positions_proposed", {})
    hedge_lite = out.get("hedge_lite", {})

    assert "SPY" in weights
    assert any(t in weights for t in ["QQQ", "GLD", "TLT", "XLE"])
    assert abs(sum(float(v) for v in weights.values()) - 1.0) < 1e-6

    hedge_positive = [t for t in ["QQQ", "GLD", "TLT", "XLE"] if float(weights.get(t, 0.0)) > 0]
    if not hedge_positive:
        assert hedge_lite.get("status") == "insufficient_data"
        assert hedge_lite.get("reason")
