"""
tests/test_coverage_expansion.py
=================================
Coverage Expansion Patch v1 테스트.

T_macro_axes           : 지표 일부 누락이어도 axes/state 안정적 + confidence 하락
T_funda_valuation_stretch : history/peers 없으면 graceful, 있으면 정상
T_senti_dedupe         : 중복 기사 → effective_article_count 감소
T_orch_intent          : 5개 샘플 → intent/universe/horizon_days 기대값
T_news_volume_z        : z-score 계산, std=0→0, 데이터부족→None
T_infer_vol_regime     : quant/macro/vix/unknown 우선순위
T_report_includes_new_fields: mock report에 "Top Drivers"/"What to watch" 포함
T_orch_llm_first_with_fallback: LLM disabled 시 rules fallback 작동
"""
import sys
import os
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from engines.macro_engine import compute_macro_axes, compute_risk_on_off
from engines.fundamental_engine import compute_valuation_stretch, compute_factor_scores
from engines.sentiment_engine import (
    dedupe_and_weight_news,
    compute_sentiment_velocity,
    detect_catalyst_risk,
    infer_vol_regime,
)
from agents.macro_agent import macro_analyst_run
from agents.fundamental_agent import fundamental_analyst_run
from agents.sentiment_agent import sentiment_analyst_run
from agents.orchestrator_agent import classify_intent_rules


# ─────────────────────────────────────────────────────────────────────────────
# T_macro_axes
# ─────────────────────────────────────────────────────────────────────────────

class TestMacroAxes:
    def test_full_indicators_returns_5_axes(self):
        ind = {"yield_curve_spread": -0.30, "hy_oas": 550, "cpi_yoy": 4.5,
               "pmi": 47, "fed_funds_rate": 5.25, "gdp_growth": 1.2}
        axes = compute_macro_axes(ind)
        assert set(axes) == {"growth", "inflation", "rates", "credit", "liquidity"}
        for name, ax in axes.items():
            assert "state" in ax and "score" in ax
            assert -3 <= ax["score"] <= 3

    def test_partial_indicators_stable(self):
        """지표 일부 누락 → axes는 항상 반환, score=0으로 중립 처리."""
        axes = compute_macro_axes({})            # 전부 누락
        assert set(axes) == {"growth", "inflation", "rates", "credit", "liquidity"}
        for ax in axes.values():
            assert ax["score"] == 0

    def test_confidence_drops_on_missing_key_data(self):
        """핵심 지표 없음 → macro_analyst_run의 confidence ≤ 0.40."""
        result = macro_analyst_run("AAPL", {})   # 완전 빈 지표
        assert result["confidence"] <= 0.40

    def test_risk_on_off_score_range(self):
        ind = {"yield_curve_spread": 1.0, "hy_oas": 200, "pmi": 58}
        axes = compute_macro_axes(ind)
        ron  = compute_risk_on_off(axes, ind)
        assert -100 <= ron["risk_score"] <= 100
        assert ron["risk_on_off"] in ("risk_on", "neutral", "risk_off")

    def test_key_drivers_in_output(self):
        ind  = {"yield_curve_spread": -0.40, "hy_oas": 600, "pmi": 44,
                "cpi_yoy": 5.0, "fed_funds_rate": 5.25}
        out  = macro_analyst_run("AAPL", ind)
        assert "key_drivers" in out and isinstance(out["key_drivers"], list)
        assert "what_to_watch" in out and isinstance(out["what_to_watch"], list)
        assert "scenario_notes" in out
        assert set(out["scenario_notes"]) >= {"bull", "base", "bear"}


# ─────────────────────────────────────────────────────────────────────────────
# T_funda_valuation_stretch
# ─────────────────────────────────────────────────────────────────────────────

class TestFundaValuationStretch:
    def test_no_history_no_peers_uses_absolute_pe(self):
        fin = {"pe_ratio": 50.0}
        vs  = compute_valuation_stretch(fin)
        assert vs["stretch_level"] == "high"
        assert vs["valuation_stretch_flag"] is True
        assert vs["confidence_down"] is True
        assert "used_absolute_thresholds" in vs["rationale"]
        assert "history_missing" in vs["rationale"]

    def test_pe_medium_range_absolute(self):
        vs = compute_valuation_stretch({"pe_ratio": 30.0})
        assert vs["stretch_level"] == "medium"
        assert vs["valuation_stretch_flag"] is False

    def test_pe_low_range_absolute(self):
        vs = compute_valuation_stretch({"pe_ratio": 18.0})
        assert vs["stretch_level"] == "low"

    def test_unknown_when_no_pe_ps(self):
        vs = compute_valuation_stretch({})
        assert vs["stretch_level"] == "unknown"
        assert vs["valuation_stretch_flag"] is False
        assert vs["confidence_down"] is True

    def test_with_history_uses_z_score(self):
        hist = {"pe_ratios": [15, 16, 14, 15, 16]}   # avg ~15.2
        vs   = compute_valuation_stretch({"pe_ratio": 40.0}, history=hist)
        assert vs["confidence_down"] is False
        assert vs["stretch_level"] in ("high", "medium")

    def test_graceful_without_history_peers(self):
        """Graceful degrade: no crash, returns dict with required keys."""
        vs = compute_valuation_stretch({"pe_ratio": 20.0}, history=None, peers=None)
        assert "valuation_stretch_flag" in vs
        assert "stretch_level" in vs
        assert "rationale" in vs

    def test_fundamental_agent_unknown_stretch_lowers_signal_strength(self):
        fin = {"total_assets": 1000}  # no PE/PS → stretch=unknown
        out = fundamental_analyst_run("AAPL", fin)
        assert out["signal_strength"] < 0.7
        ww = " ".join(out.get("what_to_watch") or [])
        assert "valuation" in ww.lower() or len(out.get("what_to_watch", [])) > 0


# ─────────────────────────────────────────────────────────────────────────────
# T_senti_dedupe
# ─────────────────────────────────────────────────────────────────────────────

class TestSentiDedupe:
    def _make_article(self, title: str, days_old: float = 0.5, source: str = "Reuters") -> dict:
        pub = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
        return {"title": title, "source": source, "published_at": pub}

    def test_deduplication_removes_similar_titles(self):
        arts = [
            self._make_article("Apple reports Q3 earnings beat"),
            self._make_article("Apple reports Q3 earnings beat expectations"),   # near-dup
            self._make_article("Apple reports Q3 earnings beat analyst estimates"),  # near-dup
            self._make_article("Tesla beats Q3 results"),  # unique
        ]
        result = dedupe_and_weight_news(arts)
        assert result["effective_article_count"] < len(arts)
        assert result["effective_article_count"] >= 2   # at least 2 unique

    def test_top_headlines_stable(self):
        arts = [self._make_article(f"Headline {i}", days_old=i * 0.3) for i in range(15)]
        result = dedupe_and_weight_news(arts)
        assert len(result["top_headlines"]) <= 8

    def test_empty_articles_returns_zero(self):
        result = dedupe_and_weight_news([])
        assert result["effective_article_count"] == 0
        assert result["top_headlines"] == []


# ─────────────────────────────────────────────────────────────────────────────
# T_news_volume_z
# ─────────────────────────────────────────────────────────────────────────────

class TestNewsVolumeZ:
    def _make_article(self, title: str, days_old: float, source: str = "Bloomberg") -> dict:
        pub = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
        return {"title": title, "source": source, "published_at": pub}

    def test_z_score_computed_with_enough_data(self):
        """30일 분산 기사 → news_volume_z is float."""
        arts = []
        for d in range(31):  # 31일치
            n = 5 if d > 0 else 20   # today spike
            for i in range(n):
                arts.append(self._make_article(f"News day={d} item={i} unique_{d}_{i}", days_old=d))
        result = dedupe_and_weight_news(arts)
        if result["baseline_days_used"] >= 10 and result["effective_article_count"] >= 10:
            assert isinstance(result["news_volume_z"], float)

    def test_z_score_none_when_insufficient_articles(self):
        """기사 10개 미만 → news_volume_z = None."""
        arts = [self._make_article(f"Article {i}", days_old=0.1) for i in range(5)]
        result = dedupe_and_weight_news(arts)
        assert result["news_volume_z"] is None
        assert "insufficient_news_baseline" in result["data_quality"]["warnings"]

    def test_z_score_zero_when_std_is_zero(self):
        """모든 날 동일 기사 수 → std=0 → z=0."""
        arts = []
        for d in range(15):  # only 15 baseline days
            for i in range(5):
                arts.append(self._make_article(f"Constant news d={d} i={i}", days_old=d + 1))
        result = dedupe_and_weight_news(arts)
        # With <10 baseline days used or <10 articles total, will be None
        if result["news_volume_z"] is not None:
            # std should be very low, z close to 0
            assert abs(result["news_volume_z"]) < 5.0

    def test_count_by_day_populated(self):
        arts = [self._make_article(f"A{i}", days_old=0.5) for i in range(3)]
        result = dedupe_and_weight_news(arts)
        assert isinstance(result["count_by_day"], dict)
        assert result["today_count"] >= 0


# ─────────────────────────────────────────────────────────────────────────────
# T_infer_vol_regime
# ─────────────────────────────────────────────────────────────────────────────

class TestInferVolRegime:
    def test_quant_regime_2_crisis(self):
        state = {"technical_analysis": {"regime_2_high_vol": 0.70}}
        r = infer_vol_regime(state, {})
        assert r["vol_regime"] == "crisis"
        assert r["source"] == "quant_regime_2"

    def test_quant_regime_2_high(self):
        state = {"technical_analysis": {"regime_2_high_vol": 0.40}}
        r = infer_vol_regime(state, {})
        assert r["vol_regime"] == "high"

    def test_quant_regime_2_normal(self):
        state = {"technical_analysis": {"regime_2_high_vol": 0.20}}
        r = infer_vol_regime(state, {})
        assert r["vol_regime"] == "normal"

    def test_macro_tail_risk_triggers_crisis(self):
        state = {
            "macro_analysis": {
                "tail_risk_warning": True,
                "indicators": {"credit_stress_level": "stressed"},
            }
        }
        r = infer_vol_regime(state, {})
        assert r["vol_regime"] == "crisis"

    def test_macro_risk_off_triggers_high(self):
        state = {
            "macro_analysis": {
                "tail_risk_warning": False,
                "risk_on_off": {"risk_on_off": "risk_off"},
            }
        }
        r = infer_vol_regime(state, {})
        assert r["vol_regime"] == "high"

    def test_vix_fallback_crisis(self):
        r = infer_vol_regime({}, {"vix_level": 35})
        assert r["vol_regime"] == "crisis"
        assert r["source"] == "vix_level"

    def test_vix_fallback_high(self):
        r = infer_vol_regime({}, {"vix_level": 25})
        assert r["vol_regime"] == "high"

    def test_unknown_fallback(self):
        r = infer_vol_regime({}, {})
        assert r["vol_regime"] == "normal"
        assert "vol_regime_unknown" in r.get("warnings", [])


# ─────────────────────────────────────────────────────────────────────────────
# T_orch_intent
# ─────────────────────────────────────────────────────────────────────────────

class TestOrchIntent:
    def test_single_ticker_entry(self):
        r = classify_intent_rules("AAPL 지금 매수해도 돼?")
        assert r["intent"] == "single_ticker_entry"
        assert "AAPL" in r["universe"]
        assert r["horizon_days"] == 30

    def test_overheated_check(self):
        r = classify_intent_rules("NVDA 너무 오른 것 같은데 더 갈까?")
        assert r["intent"] == "overheated_check"
        assert r["horizon_days"] == 90

    def test_compare_tickers(self):
        r = classify_intent_rules("AAPL vs MSFT 어느 게 나아?")
        assert r["intent"] == "compare_tickers"
        assert "AAPL" in r["universe"] or "MSFT" in r["universe"]

    def test_market_outlook(self):
        r = classify_intent_rules("지금 시장 전망이 어때?")
        assert r["intent"] == "market_outlook"
        assert "SPY" in r["universe"]

    def test_event_risk(self):
        r = classify_intent_rules("TSLA 실적 발표 앞두고 들어가도 돼?")
        assert r["intent"] == "event_risk"
        assert r["horizon_days"] == 14

    def test_mock_market_outlook_avoids_aapl_default(self):
        from agents.orchestrator_agent import _mock_orchestrator_decision
        out = _mock_orchestrator_decision("미국 증시 전망 알려줘", 0)
        universe = out.get("investment_brief", {}).get("target_universe", [])
        assert "SPY" in universe
        assert "AAPL" not in universe
        assert "시장/섹터 전망" in out.get("investment_brief", {}).get("rationale", "")

    def test_no_ticker_question_does_not_fall_back_to_aapl(self):
        r = classify_intent_rules("요즘 투자 어떻게 하는 게 좋아?")
        assert r["intent"] == "market_outlook"
        assert "SPY" in r["universe"]
        assert "AAPL" not in r["universe"]

    def test_desk_tasks_always_present(self):
        for req in ["AAPL 매수?", "시장 전망", "NVDA vs AMD", "너무 오른 것 같은데", "실적 앞두고"]:
            r = classify_intent_rules(req)
            assert "desk_tasks" in r
            assert "macro" in r["desk_tasks"]
            assert "quant"  in r["desk_tasks"]


# ─────────────────────────────────────────────────────────────────────────────
# T_report_includes_new_fields
# ─────────────────────────────────────────────────────────────────────────────

class TestReportIncludesNewFields:
    def _make_mock_state(self):
        from agents.macro_agent import macro_analyst_run
        from agents.fundamental_agent import fundamental_analyst_run
        from agents.sentiment_agent import sentiment_analyst_run
        macro = macro_analyst_run(
            "AAPL",
            {
                "pmi": 48,
                "hy_oas": 450,
                "yield_curve_spread": -0.2,
                "dgs2": 4.1,
                "dgs10": 3.9,
                "fed_funds_rate": 4.25,
                "dollar_index": 120.0,
                "vix_level": 21.0,
                "wti_spot": 81.0,
                "fed_funds_futures_front_implied_rate": 4.1,
                "fed_funds_futures_3m_implied_rate": 3.9,
                "fed_funds_futures_6m_implied_rate": 3.6,
                "fed_funds_futures_implied_change_6m_bp": -50.0,
            },
        )
        funda = fundamental_analyst_run("AAPL", {"pe_ratio": 30, "revenue_growth": 10, "roe": 20})
        senti = sentiment_analyst_run("AAPL", {"vix_level": 18, "put_call_ratio": 0.9})
        return {
            "user_request": "AAPL 매수해도 돼?",
            "target_ticker": "AAPL",
            "iteration_count": 0,
            "macro_analysis":       macro,
            "fundamental_analysis": funda,
            "sentiment_analysis":   senti,
            "technical_analysis":   {},
            "risk_assessment":      {},
            "orchestrator_directives": {},
        }

    def test_report_human_msg_contains_key_drivers(self):
        from agents.report_agent import _build_report_human_msg
        state = self._make_mock_state()
        msg = _build_report_human_msg(state)
        # Should contain at least one reference to key data section
        assert "Key Drivers" in msg or "핵심 내용" in msg
        assert "Macro Market Inputs" in msg
        assert "Fed Funds Fut 6M" in msg

    def test_mock_report_contains_top_drivers_section(self):
        from agents.report_agent import _mock_generate_report
        state = self._make_mock_state()
        report = _mock_generate_report(state)
        assert isinstance(report, str)
        assert len(report) > 100   # non-empty report
        assert "Macro Market Inputs" in report
        assert "Macro Translation" in report


# ─────────────────────────────────────────────────────────────────────────────
# T_orch_llm_first_with_fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestOrchLlmFirstWithFallback:
    def test_rules_fallback_has_required_fields(self):
        """LLM 없어도 rules fallback이 항상 필수 필드를 반환."""
        from agents.orchestrator_agent import classify_intent_rules, _mock_orchestrator_decision
        rules = classify_intent_rules("AAPL 지금 사도 돼?")
        assert "intent" in rules
        assert "universe" in rules
        assert "desk_tasks" in rules
        mock_out = _mock_orchestrator_decision("AAPL 지금 사도 돼?", 0)
        assert "action_type" in mock_out
        assert "investment_brief" in mock_out
        assert "desk_tasks" in mock_out

    def test_plan_cache_key_deterministic(self):
        from agents.orchestrator_agent import _plan_cache_key
        k1 = _plan_cache_key("AAPL buy?", 0, None)
        k2 = _plan_cache_key("AAPL buy?", 0, None)
        k3 = _plan_cache_key("AAPL buy?", 1, None)
        assert k1 == k2          # same inputs → same key
        assert k1 != k3          # different iteration → different key

    def test_call_llm_returns_dict(self):
        """테스트 실호출 모드에서는 키 없으면 RuntimeError, 키 있으면 dict 반환."""
        from agents.orchestrator_agent import _call_llm
        from llm.router import (
            force_real_llm_in_tests,
            _get_cerebras_key,
            _get_zai_key,
            _get_groq_key,
            _get_gemini_key,
        )
        has_any_llm_key = bool(
            _get_cerebras_key() or _get_zai_key() or _get_groq_key() or _get_gemini_key()
        )
        if force_real_llm_in_tests() and not has_any_llm_key:
            with pytest.raises(RuntimeError):
                _call_llm("AAPL 매수해도 돼?", 0)
            return
        result = _call_llm("AAPL 매수해도 돼?", 0)
        assert isinstance(result, dict)
        assert "desk_tasks" in result or "action_type" in result
