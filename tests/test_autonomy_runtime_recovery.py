"""
tests/test_autonomy_runtime_recovery.py
======================================
Runtime autonomy self-heal tests:
- dynamic issue -> recovery plan
- risk narrative 413 compact retry
- NewsAPI everything -> top-headlines fallback
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.autonomy_planner import plan_runtime_recovery
from agents import risk_agent
from data_providers.base import ProviderError
from data_providers.newsapi_provider import NewsAPIProvider


def test_runtime_planner_builds_actions_from_insight_gaps():
    state = {
        "target_ticker": "NVDA",
        "user_request": "NVDA 과열 여부와 근거 점검",
        "task_backlog": [],
        "evidence_score": 30,
    }
    desk_outputs = {
        "macro": {
            "needs_more_data": True,
            "confidence": 0.42,
            "data_quality": {"missing_pct": 0.35},
            "open_questions": [{"q": "매크로 전환 트리거는?", "kind": "macro_headline_context"}],
            "limitations": [],
        },
        "fundamental": {"limitations": []},
        "sentiment": {"limitations": []},
        "quant": {"limitations": []},
    }

    out = plan_runtime_recovery(state, desk_outputs)
    action_types = {a.get("type") for a in out.get("actions", [])}
    req_kinds = {r.get("kind") for r in out.get("evidence_requests", [])}
    codes = {i.get("code") for i in out.get("issues", [])}

    assert "insight_gap" in codes
    assert "unresolved_questions" in codes
    assert "run_research" in action_types or "rerun_desk" in action_types
    assert "macro_headline_context" in req_kinds


def test_runtime_planner_detects_provider_restriction():
    state = {
        "target_ticker": "NVDA",
        "task_backlog": [],
        "evidence_score": 65,
    }
    desk_outputs = {
        "macro": {"limitations": []},
        "fundamental": {
            "limitations": [
                "FMP endpoint unavailable (key-metrics-ttm/key-metrics): [fmp] HTTP error: 403 forbidden"
            ]
        },
        "sentiment": {"limitations": []},
        "quant": {"limitations": []},
    }

    out = plan_runtime_recovery(state, desk_outputs)
    action_types = {a.get("type") for a in out.get("actions", [])}
    req_kinds = {r.get("kind") for r in out.get("evidence_requests", [])}

    assert "provider_fallback" in action_types
    assert "sec_filing" in req_kinds


def test_risk_llm_413_compact_retry(monkeypatch):
    class _Resp:
        content = (
            '{"per_ticker_rationales":{"NVDA":"근거를 압축 반영해 보수적 유지"},'
            '"feedback_detail":"서사는 compact payload 기준으로 업데이트됨"}'
        )

    class _FakeLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, _msgs):
            self.calls += 1
            if self.calls == 1:
                raise Exception("413 Request too large: tokens per minute exceeded")
            return _Resp()

    monkeypatch.setattr(risk_agent, "HAS_LC", True)
    monkeypatch.setattr(risk_agent, "get_llm", lambda _name: _FakeLLM())

    payload = {
        "timestamp": "2026-02-24T00:00:00Z",
        "risk_limits": risk_agent.LIMITS,
        "portfolio_risk_summary": {
            "total_gross_exposure": 0.1,
            "total_net_exposure": 0.1,
            "leverage_ratio": 0.1,
            "portfolio_cvar_1d": 0.005,
            "component_var_by_ticker": {"NVDA": 0.005},
            "herfindahl_index": 0.1,
            "liquidity_score_by_ticker": {"NVDA": 0.3},
            "sector_exposure": {"Technology": 0.1},
        },
        "analyst_reports": {
            "macro": {"macro_regime": "expansion", "primary_decision": "bullish", "confidence": 0.6, "evidence": [{"metric": "m"}]},
            "fundamental": {
                "primary_decision": "bullish",
                "structural_risk_flag": False,
                "risk_flags": [],
                "confidence": 0.6,
                "evidence": [{"metric": "f"}],
            },
            "sentiment": {"primary_decision": "neutral", "tilt_factor": 1.0, "confidence": 0.6, "evidence": [{"metric": "s"}]},
            "quant": {"decision": "HOLD", "final_allocation_pct": 0.0, "asset_cvar_99_daily": 0.005, "evidence": [{"metric": "q"}]},
            "_target_ticker": "NVDA",
        },
    }

    decision = risk_agent._call_llm(payload)
    assert decision.get("_llm_enrichment_status") == "ok_compact_retry"
    assert "compact payload" in decision.get("orchestrator_feedback", {}).get("detail", "")


def test_newsapi_everything_426_falls_back_to_top_headlines(monkeypatch):
    provider = NewsAPIProvider()
    monkeypatch.setattr(provider, "_api_key", "x")

    def _fake_get_json(url, params=None, headers=None, cache_ttl=None, skip_cache=False):
        if url.endswith("/everything"):
            raise ProviderError("[newsapi] HTTP error: 426 Upgrade Required")
        if url.endswith("/top-headlines"):
            return {
                "totalResults": 1,
                "articles": [{"title": "NVDA gains on AI demand"}],
            }
        return {"totalResults": 0, "articles": []}

    monkeypatch.setattr(provider, "get_json", _fake_get_json)

    out = provider.search_ticker_news("NVDA", days=120)
    assert out["data_ok"] is True
    assert out["data"]["article_count"] >= 1
    assert any("fallback to /top-headlines" in x for x in out.get("limitations", []))
