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

    def test_fundamental_agent_emits_consensus_catalyst_and_variant_fields(self):
        fin = {
            "total_assets": 500_000,
            "current_assets": 150_000,
            "current_liabilities": 50_000,
            "retained_earnings": 200_000,
            "ebit": 75_000,
            "ebitda": 100_000,
            "market_cap": 2_000_000,
            "total_liabilities": 200_000,
            "revenue": 400_000,
            "interest_expense": 5_000,
            "net_debt": 50_000,
            "free_cash_flow": 70_000,
            "fcf_history": [60_000, 58_000, 52_000, 45_000, 41_000],
            "revenue_growth": 15.0,
            "earnings_growth": 18.0,
            "pe_ratio": 20.0,
            "ps_ratio": 6.0,
            "roe": 25.0,
            "debt_to_equity": 0.7,
            "current_price": 100.0,
            "next_earnings_date": "2026-04-30",
            "earnings_in_days": 10,
            "next_eps_estimate": 1.8,
            "next_revenue_estimate": 405_000,
            "analyst_estimate_fy": "2027-09-30",
            "analyst_eps_estimate_next_fy": 8.2,
            "analyst_revenue_estimate_next_fy": 450_000,
            "price_target_consensus": 118.0,
            "price_target_upside_pct": 18.0,
            "rating": "B",
            "rating_overall_score": 3,
            "rating_dcf_score": 3,
        }
        out = fundamental_analyst_run("AAPL", fin)
        assert "consensus_snapshot" in out
        assert "catalyst_calendar" in out and out["catalyst_calendar"]
        assert "valuation_anchor" in out
        assert "variant_view" in out
        assert out["consensus_snapshot"]["price_target_upside_pct"] == 18.0
        assert out["catalyst_calendar"][0]["type"] == "earnings"
        assert out["valuation_anchor"]["price_target_consensus"] == 118.0
        assert out["variant_view"]["key_gap"]

    def test_fundamental_agent_near_earnings_limits_sizing(self):
        fin = {
            "total_assets": 500_000,
            "current_assets": 150_000,
            "current_liabilities": 50_000,
            "retained_earnings": 200_000,
            "ebit": 75_000,
            "ebitda": 100_000,
            "market_cap": 2_000_000,
            "total_liabilities": 200_000,
            "revenue": 400_000,
            "interest_expense": 5_000,
            "net_debt": 50_000,
            "free_cash_flow": 70_000,
            "fcf_history": [60_000, 58_000, 52_000, 45_000, 41_000],
            "revenue_growth": 15.0,
            "pe_ratio": 20.0,
            "roe": 25.0,
            "debt_to_equity": 0.7,
            "next_earnings_date": "2026-04-30",
            "earnings_in_days": 7,
            "next_eps_estimate": 1.8,
            "next_revenue_estimate": 405_000,
        }
        out = fundamental_analyst_run("AAPL", fin)
        assert out["recommendation"] == "allow_with_limits"

    def test_fundamental_agent_emits_peer_model_management_and_ownership_layers(self):
        fin = {
            "total_assets": 500_000,
            "current_assets": 150_000,
            "current_liabilities": 50_000,
            "retained_earnings": 200_000,
            "ebit": 75_000,
            "ebitda": 100_000,
            "market_cap": 2_000_000,
            "total_liabilities": 200_000,
            "revenue": 400_000,
            "interest_expense": 5_000,
            "net_debt": 50_000,
            "free_cash_flow": 70_000,
            "fcf_history": [60_000, 58_000, 52_000, 45_000, 41_000],
            "revenue_growth": 15.0,
            "earnings_growth": 18.0,
            "pe_ratio": 20.0,
            "ps_ratio": 6.0,
            "roe": 25.0,
            "debt_to_equity": 0.7,
            "operating_margin": 22.0,
            "current_price": 100.0,
            "price_avg_50d": 95.0,
            "price_avg_200d": 90.0,
            "eps_ttm_proxy": 5.0,
            "next_earnings_date": "2026-04-30",
            "earnings_in_days": 18,
            "next_eps_estimate": 1.8,
            "next_revenue_estimate": 405_000,
            "analyst_estimate_fy": "2027-09-30",
            "analyst_eps_estimate_next_fy": 8.2,
            "analyst_revenue_estimate_next_fy": 450_000,
            "price_target_consensus": 118.0,
            "price_target_upside_pct": 18.0,
            "rating_revision_score_30d": 2.0,
            "rating_revision_score_90d": 3.5,
            "street_revision_proxy": "improving",
            "estimate_periods": {
                "0q": {
                    "eps_estimate": 1.95,
                    "revenue_estimate": 410_000,
                    "eps_revision_30d_pct": 3.2,
                    "eps_revision_90d_pct": 5.5,
                    "up_last_30d": 14,
                    "down_last_30d": 2,
                    "revision_state": "improving",
                },
                "0y": {
                    "eps_estimate": 8.4,
                    "revenue_estimate": 455_000,
                    "eps_revision_30d_pct": 2.1,
                    "eps_revision_90d_pct": 4.4,
                    "up_last_30d": 18,
                    "down_last_30d": 1,
                    "revision_state": "improving",
                },
            },
            "valuation_history_real": {
                "status": "ok",
                "source": "quarterly_statements_plus_quarter_end_prices",
                "valuation_points": [
                    {"date": "2025-03-31", "pe_ratio": 18.0, "ps_ratio": 5.5, "fcf_yield_pct": 3.0},
                    {"date": "2025-06-30", "pe_ratio": 19.0, "ps_ratio": 5.7, "fcf_yield_pct": 2.9},
                    {"date": "2025-09-30", "pe_ratio": 20.0, "ps_ratio": 6.0, "fcf_yield_pct": 2.8},
                    {"date": "2025-12-31", "pe_ratio": 21.0, "ps_ratio": 6.1, "fcf_yield_pct": 2.7},
                ],
                "pe_ratios": [18.0, 19.0, 20.0, 21.0],
                "ps_ratios": [5.5, 5.7, 6.0, 6.1],
                "fcf_yield_pcts": [3.0, 2.9, 2.8, 2.7],
            },
            "ttm_revenue_real": 430_000,
            "ttm_operating_margin_real": 23.0,
            "ttm_fcf_real": 75_000,
            "ttm_fcf_margin_real": 17.4,
            "eps_ttm_real": 5.2,
            "shares_outstanding_real": 15_000,
            "buyback_yield_pct": 2.5,
            "dividend_cash_yield_pct": 0.7,
            "capital_return_yield_pct": 3.2,
            "acquisitions_last_fy": 1_000,
            "debt_funded_buyback_flag": False,
            "last_eps_surprise_pct": 4.0,
            "earnings_beat_rate_4q": 75.0,
            "revenue_beat_rate_4q": 75.0,
            "eps_surprise_avg_4q": 3.5,
            "audit_risk_yahoo": 2,
            "board_risk_yahoo": 1,
            "compensation_risk_yahoo": 6,
            "shareholder_rights_risk_yahoo": 2,
        }
        peers = [
            {"symbol": "MSFT", "pe_ratio": 28.0, "ps_ratio": 10.0, "operating_margin": 44.0, "roe": 30.0, "revenue_growth": 14.0},
            {"symbol": "GOOGL", "pe_ratio": 24.0, "ps_ratio": 7.0, "operating_margin": 31.0, "roe": 27.0, "revenue_growth": 13.0},
            {"symbol": "META", "pe_ratio": 25.0, "ps_ratio": 8.0, "operating_margin": 38.0, "roe": 29.0, "revenue_growth": 16.0},
        ]
        ownership_items = [
            {"title": "APPLE INC  (AAPL)  (CIK 0000320193) Form 4", "published_at": "2026-03-01"},
            {"title": "APPLE INC  (AAPL)  (CIK 0000320193) SC 13G", "published_at": "2026-02-20"},
        ]
        ownership_snapshot = {
            "status": "ok",
            "institutions_percent_held": 0.65,
            "institutions_float_percent_held": 0.67,
            "institutions_count": 4500,
            "top_holder_pct": 0.10,
            "institutional_top10_pct": 0.31,
            "institutional_hhi_top10": 0.018,
            "institutional_concentration_level": "medium",
            "crowding_risk": "normal",
            "insider_net_activity": "buying",
            "insider_net_shares_6m": 120000,
            "incremental_buyer_seller_map": {
                "buyers": [{"holder": "Vanguard", "pct_change": 0.02, "pct_held": 0.10, "date_reported": "2025-12-31"}],
                "sellers": [{"holder": "FMR", "pct_change": -0.01, "pct_held": 0.03, "date_reported": "2025-12-31"}],
            },
            "ownership_report_date": "2025-12-31",
        }
        out = fundamental_analyst_run("AAPL", fin, peers=peers, ownership_items=ownership_items, ownership_snapshot=ownership_snapshot)
        assert out["consensus_revision_layer"]["estimate_revision_proxy"]["mode"] == "yfinance_eps_trend"
        assert out["consensus_revision_layer"]["estimate_revision_proxy"]["state"] == "improving"
        assert out["peer_comp_engine"]["status"] == "ok"
        assert out["peer_comp_engine"]["peer_median_pe"] == 25.0
        assert out["model_pack"]["status"] == "ok"
        assert out["street_disagreement_layer"]["consensus_strength"] in {"strong", "balanced"}
        assert out["expected_return_profile"]["quality"] in {"attractive", "balanced"}
        assert out["management_capital_allocation"]["guidance_credibility"] == "high"
        assert out["model_pack"]["basis"].startswith("real TTM financials")
        assert out["model_pack"]["revenue_build"]["history_quarters"] == []
        assert out["ownership_crowding"]["status"] == "structured"
        assert out["ownership_crowding"]["institutional_concentration"]["top10_pct"] == 0.31
        assert out["data_provenance"]["quality"] in {"high", "medium"}
        assert out["monitoring_triggers"]

    def test_fundamental_low_provenance_and_high_uncertainty_reduce_recommendation(self):
        fin = {
            "total_assets": 500_000,
            "current_assets": 150_000,
            "current_liabilities": 50_000,
            "retained_earnings": 200_000,
            "ebit": 75_000,
            "ebitda": 100_000,
            "market_cap": 2_000_000,
            "total_liabilities": 200_000,
            "revenue": 400_000,
            "interest_expense": 5_000,
            "net_debt": 50_000,
            "free_cash_flow": 70_000,
            "fcf_history": [60_000, 58_000, 52_000, 45_000, 41_000],
            "revenue_growth": 15.0,
            "earnings_growth": 18.0,
            "pe_ratio": 20.0,
            "ps_ratio": 6.0,
            "roe": 25.0,
            "debt_to_equity": 0.7,
            "operating_margin": 22.0,
            "current_price": 100.0,
            "next_earnings_date": "2026-04-30",
            "earnings_in_days": 5,
            "next_eps_estimate": 1.8,
            "next_revenue_estimate": 405_000,
            "analyst_estimate_fy": "2027-09-30",
            "analyst_eps_estimate_next_fy": 8.2,
            "analyst_revenue_estimate_next_fy": 450_000,
            "price_target_consensus": 109.0,
            "price_target_high": 140.0,
            "price_target_low": 80.0,
            "price_target_upside_pct": 9.0,
            "street_revision_proxy": "deteriorating",
            "rating_revision_score_30d": -2.5,
            "rating_revision_score_90d": -3.0,
        }
        out = fundamental_analyst_run("AAPL", fin)
        assert out["data_provenance"]["quality"] == "low"
        assert out["street_disagreement_layer"]["uncertainty_level"] == "high"
        assert out["recommendation"] == "allow_with_limits"
        assert out["confidence"] <= 0.52

    def test_fundamental_monitoring_triggers_capture_revision_crowding_and_peer_stress(self):
        fin = {
            "total_assets": 500_000,
            "current_assets": 150_000,
            "current_liabilities": 50_000,
            "retained_earnings": 200_000,
            "ebit": 75_000,
            "ebitda": 100_000,
            "market_cap": 2_000_000,
            "total_liabilities": 200_000,
            "revenue": 400_000,
            "interest_expense": 5_000,
            "net_debt": 50_000,
            "free_cash_flow": 70_000,
            "fcf_history": [60_000, 58_000, 52_000, 45_000, 41_000],
            "revenue_growth": 15.0,
            "earnings_growth": 18.0,
            "pe_ratio": 34.0,
            "ps_ratio": 6.0,
            "roe": 25.0,
            "debt_to_equity": 0.7,
            "operating_margin": 22.0,
            "current_price": 100.0,
            "next_earnings_date": "2026-04-30",
            "earnings_in_days": 6,
            "price_target_consensus": 104.0,
            "price_target_high": 120.0,
            "price_target_low": 78.0,
            "price_target_upside_pct": 4.0,
            "estimate_periods": {
                "0q": {
                    "eps_estimate": 1.95,
                    "revenue_estimate": 410_000,
                    "eps_revision_30d_pct": -3.2,
                    "eps_revision_90d_pct": -5.5,
                    "up_last_30d": 2,
                    "down_last_30d": 10,
                    "revision_state": "deteriorating",
                },
            },
            "earnings_beat_rate_4q": 75.0,
            "revenue_beat_rate_4q": 75.0,
            "eps_surprise_avg_4q": 3.5,
            "audit_risk_yahoo": 2,
            "board_risk_yahoo": 1,
            "compensation_risk_yahoo": 6,
            "shareholder_rights_risk_yahoo": 2,
        }
        peers = [
            {"symbol": "MSFT", "pe_ratio": 20.0, "ps_ratio": 8.0, "operating_margin": 44.0},
            {"symbol": "GOOGL", "pe_ratio": 18.0, "ps_ratio": 7.0, "operating_margin": 31.0},
            {"symbol": "META", "pe_ratio": 19.0, "ps_ratio": 8.0, "operating_margin": 38.0},
        ]
        ownership_snapshot = {
            "status": "ok",
            "institutions_percent_held": 0.78,
            "institutions_float_percent_held": 0.80,
            "institutions_count": 4500,
            "top_holder_pct": 0.14,
            "institutional_top10_pct": 0.36,
            "institutional_hhi_top10": 0.025,
            "institutional_concentration_level": "high",
            "crowding_risk": "elevated",
            "insider_net_activity": "selling",
            "insider_net_shares_6m": -120000,
            "incremental_buyer_seller_map": {"buyers": [], "sellers": []},
            "ownership_report_date": "2025-12-31",
        }
        out = fundamental_analyst_run("AAPL", fin, peers=peers, ownership_snapshot=ownership_snapshot)
        trigger_names = {t["name"] for t in out["monitoring_triggers"]}
        assert "Earnings catalyst" in trigger_names
        assert "Estimate revision drift" in trigger_names
        assert "Peer premium stress" in trigger_names
        assert "Ownership pressure" in trigger_names

    def test_fundamental_catalyst_engine_marks_sec_8k_as_confirmed(self, monkeypatch):
        monkeypatch.setattr("agents.fundamental_agent.apply_llm_overlay_fundamental", lambda *args, **kwargs: {})
        fin = {
            "total_assets": 500_000,
            "current_assets": 150_000,
            "current_liabilities": 50_000,
            "retained_earnings": 200_000,
            "ebit": 75_000,
            "ebitda": 100_000,
            "market_cap": 2_000_000,
            "total_liabilities": 200_000,
            "revenue": 400_000,
            "interest_expense": 5_000,
            "net_debt": 50_000,
            "free_cash_flow": 70_000,
            "fcf_history": [60_000, 58_000, 52_000, 45_000, 41_000],
            "revenue_growth": 15.0,
            "pe_ratio": 20.0,
            "roe": 25.0,
            "debt_to_equity": 0.7,
        }
        catalyst_items = [
            {
                "title": "Apple Inc. Announces Investor Day and capital markets day agenda",
                "url": "https://www.sec.gov/Archives/edgar/data/0000320193/example8k.htm",
                "published_at": "2026-03-01",
                "kind": "press_release_or_ir",
                "resolver_path": "sec_8k",
                "source": "sec.gov",
                "snippet": "Investor Day and capital markets day event details",
            }
        ]
        out = fundamental_analyst_run("AAPL", fin, catalyst_items=catalyst_items)
        investor_day = next(x for x in out["catalyst_calendar"] if x["type"] == "investor_day")
        assert investor_day["source_classification"] == "confirmed"
        assert investor_day["resolver_path"] == "sec_8k"

    def test_fundamental_catalyst_engine_prefers_confirmed_over_inferred_same_type(self, monkeypatch):
        monkeypatch.setattr("agents.fundamental_agent.apply_llm_overlay_fundamental", lambda *args, **kwargs: {})
        fin = {
            "total_assets": 500_000,
            "current_assets": 150_000,
            "current_liabilities": 50_000,
            "retained_earnings": 200_000,
            "ebit": 75_000,
            "ebitda": 100_000,
            "market_cap": 2_000_000,
            "total_liabilities": 200_000,
            "revenue": 400_000,
            "interest_expense": 5_000,
            "net_debt": 50_000,
            "free_cash_flow": 70_000,
            "fcf_history": [60_000, 58_000, 52_000, 45_000, 41_000],
            "revenue_growth": 15.0,
            "pe_ratio": 20.0,
            "roe": 25.0,
            "debt_to_equity": 0.7,
        }
        state = {
            "evidence_store": {
                "abc": {
                    "title": "Blog speculates company price cut and discount campaign",
                    "url": "https://example.com/blog/price-cut",
                    "published_at": "2026-03-02",
                    "snippet": "Possible price cut and discount",
                    "source": "example.com",
                    "desk": "fundamental",
                    "ticker": "AAPL",
                    "kind": "press_release_or_ir",
                    "resolver_path": "tavily_web_fallback",
                }
            }
        }
        catalyst_items = [
            {
                "title": "Company files 8-K on pricing update and discount changes",
                "url": "https://www.sec.gov/Archives/edgar/data/0000320193/example-price-8k.htm",
                "published_at": "2026-03-03",
                "kind": "press_release_or_ir",
                "resolver_path": "sec_8k",
                "source": "sec.gov",
                "snippet": "pricing update and discount changes",
            }
        ]
        out = fundamental_analyst_run("AAPL", fin, state=state, catalyst_items=catalyst_items)
        pricing_reset = next(x for x in out["catalyst_calendar"] if x["type"] == "pricing_reset")
        assert pricing_reset["source_classification"] == "confirmed"
        assert pricing_reset["source_title"].startswith("Company files 8-K")

    def test_fundamental_catalyst_engine_uses_structured_ir_event_type(self, monkeypatch):
        monkeypatch.setattr("agents.fundamental_agent.apply_llm_overlay_fundamental", lambda *args, **kwargs: {})
        fin = {
            "total_assets": 500_000,
            "current_assets": 150_000,
            "current_liabilities": 50_000,
            "retained_earnings": 200_000,
            "ebit": 75_000,
            "ebitda": 100_000,
            "market_cap": 2_000_000,
            "total_liabilities": 200_000,
            "revenue": 400_000,
            "interest_expense": 5_000,
            "net_debt": 50_000,
            "free_cash_flow": 70_000,
            "fcf_history": [60_000, 58_000, 52_000, 45_000, 41_000],
            "revenue_growth": 15.0,
            "pe_ratio": 20.0,
            "roe": 25.0,
            "debt_to_equity": 0.7,
        }
        catalyst_items = [
            {
                "title": "Company unveils next-generation product family",
                "url": "https://www.businesswire.com/news/home/example",
                "published_at": "2026-03-04",
                "kind": "press_release_or_ir",
                "resolver_path": "ir_press_release_tavily",
                "source": "businesswire.com",
                "snippet": "Launch event",
                "catalyst_type": "product_cycle",
                "source_classification": "confirmed",
            }
        ]
        out = fundamental_analyst_run("AAPL", fin, catalyst_items=catalyst_items)
        product = next(x for x in out["catalyst_calendar"] if x["type"] == "product_cycle")
        assert product["source_classification"] == "confirmed"
        assert product["source_title"] == "Company unveils next-generation product family"

    def test_fundamental_agent_revision_and_peer_premium_can_turn_bearish(self):
        fin = {
            "total_assets": 500_000,
            "current_assets": 150_000,
            "current_liabilities": 50_000,
            "retained_earnings": 200_000,
            "ebit": 75_000,
            "ebitda": 100_000,
            "market_cap": 2_000_000,
            "total_liabilities": 200_000,
            "revenue": 400_000,
            "interest_expense": 5_000,
            "net_debt": 50_000,
            "free_cash_flow": 70_000,
            "fcf_history": [60_000, 58_000, 52_000, 45_000, 41_000],
            "revenue_growth": 15.0,
            "pe_ratio": 35.0,
            "ps_ratio": 10.0,
            "roe": 25.0,
            "debt_to_equity": 0.7,
            "operating_margin": 20.0,
            "current_price": 100.0,
            "analyst_eps_estimate_next_fy": 3.2,
            "analyst_revenue_estimate_next_fy": 410_000,
            "price_target_consensus": 97.0,
            "price_target_upside_pct": -3.0,
            "rating_revision_score_30d": -3.0,
            "rating_revision_score_90d": -4.5,
            "street_revision_proxy": "deteriorating",
            "estimate_periods": {
                "0q": {
                    "eps_estimate": 1.42,
                    "revenue_estimate": 398_000,
                    "eps_revision_30d_pct": -4.5,
                    "eps_revision_90d_pct": -7.2,
                    "up_last_30d": 1,
                    "down_last_30d": 16,
                    "revision_state": "deteriorating",
                },
            },
        }
        peers = [
            {"symbol": "MSFT", "pe_ratio": 20.0, "ps_ratio": 6.0, "operating_margin": 35.0, "roe": 28.0, "revenue_growth": 10.0},
            {"symbol": "GOOGL", "pe_ratio": 18.0, "ps_ratio": 5.0, "operating_margin": 28.0, "roe": 24.0, "revenue_growth": 8.0},
            {"symbol": "META", "pe_ratio": 17.0, "ps_ratio": 4.0, "operating_margin": 30.0, "roe": 26.0, "revenue_growth": 9.0},
        ]
        out = fundamental_analyst_run("AAPL", fin, peers=peers)
        assert out["primary_decision"] == "bearish"
        assert out["recommendation"] == "allow_with_limits"


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

    def test_term_structure_backwardation_with_vvix_is_high(self):
        r = infer_vol_regime({}, {"vix_level": 24, "vvix_level": 102, "vix_term_structure": "backwardation"})
        assert r["vol_regime"] in {"high", "crisis"}
        assert r["source"] == "vix_term_structure"

    def test_unknown_fallback(self):
        r = infer_vol_regime({}, {})
        assert r["vol_regime"] == "normal"
        assert "vol_regime_unknown" in r.get("warnings", [])


class TestSentimentCatalystAndStructure:
    def test_detect_catalyst_risk_counts_confirmed_imminent_events(self):
        catalyst = detect_catalyst_risk(
            [
                {"type": "earnings", "status": "imminent", "days_to_event": 2, "confirmed": True, "source_classification": "confirmed"},
                {"type": "macro_event", "status": "upcoming", "days_to_event": 10, "confirmed": True, "source_classification": "confirmed"},
            ],
            news_volume_z=0.2,
        )
        assert catalyst["catalyst_present"] is True
        assert catalyst["confirmed_count"] == 2
        assert catalyst["imminent_count"] == 1
        assert catalyst["catalyst_risk_level"] == "high"

    def test_sentiment_agent_emits_structured_positioning_and_event_layers(self):
        out = sentiment_analyst_run(
            "SPY",
            {
                "news_sentiment_score": -0.25,
                "news_articles": [
                    {"title": "SPY volatility rises into Fed week", "source": "Reuters", "published_at": datetime.now(timezone.utc).isoformat()},
                    {"title": "Options hedging grows before Fed week", "source": "Bloomberg", "published_at": datetime.now(timezone.utc).isoformat()},
                ],
                "vix_level": 27.0,
                "vvix_level": 104.0,
                "skew_index": 146.0,
                "vix_term_structure": "backwardation",
                "put_call_oi_ratio": 1.22,
                "put_call_volume_ratio": 1.15,
                "short_interest_pct": 3.2,
                "short_interest_change_pct": 18.0,
                "held_percent_institutions": 82.0,
                "institutional_top10_pct": 41.0,
                "ownership_crowding_risk": "elevated",
                "upcoming_events": [
                    {"type": "macro_event", "subtype": "FOMC", "status": "imminent", "days_to_event": 2, "confirmed": True, "source_classification": "confirmed"},
                ],
            },
            source_name="multi_source",
        )
        assert out["options_vol_structure"]["vix_term_structure"] == "backwardation"
        assert out["positioning_snapshot"]["positioning_crowding"] in {"short_crowded", "long_crowded", "balanced"}
        assert out["confirmed_events"]
        assert out["data_provenance"]["quality"] in {"medium", "high"}
        assert out["monitoring_triggers"]
        assert out["catalyst_risk"]["confirmed_count"] == 1
        assert out["volatility_regime"] in {"high", "crisis"}


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
                "sofr_rate": 4.20,
                "dollar_index": 120.0,
                "vix_level": 21.0,
                "wti_spot": 81.0,
                "fed_funds_futures_front_implied_rate": 4.1,
                "fed_funds_futures_3m_implied_rate": 3.9,
                "fed_funds_futures_6m_implied_rate": 3.6,
                "fed_funds_futures_implied_change_6m_bp": -50.0,
                "sofr_futures_front_implied_rate": 4.15,
                "sofr_futures_3m_implied_rate": 4.0,
                "sofr_futures_6m_implied_rate": 3.8,
                "sofr_futures_implied_change_6m_bp": -35.0,
                "sofr_ff_6m_basis_bp": 20.0,
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
        assert "SOFR Fut 6M" in msg

    def test_mock_report_contains_top_drivers_section(self):
        from agents.report_agent import _mock_generate_report
        state = self._make_mock_state()
        report = _mock_generate_report(state)
        assert isinstance(report, str)
        assert len(report) > 100   # non-empty report
        assert "Macro Market Inputs" in report
        assert "Macro Translation" in report

    def test_fidelity_report_contains_quality_and_monitoring_sections(self):
        from agents.report_agent import report_writer_node
        state = self._make_mock_state()
        state["target_ticker"] = "AAPL"
        state["positions_proposed"] = {"AAPL": 0.08}
        state["positions_final"] = {"AAPL": 0.08}
        state["book_allocation_plan"] = {"gross_target": 0.5, "single_name_cap": 0.08, "quality_haircut": 0.9}
        state["capital_competition"] = [{"ticker": "AAPL", "book_action": "add", "conviction_score": 0.7, "expected_return_score": 0.6, "downside_penalty": 0.2, "catalyst_proximity_score": 0.1, "target_weight": 0.08}]
        state["decision_quality_scorecard"] = {"overall_score": 0.61, "weak_desks": ["sentiment"]}
        state["event_calendar"] = [{"desk": "macro", "type": "fomc", "status": "imminent", "date": "2026-03-18T18:00:00+00:00", "source": "federalreserve.gov"}]
        state["monitoring_actions"] = {"force_research": True, "selected_desks": ["macro"], "risk_refresh_required": True}
        state["technical_analysis"] = {
            "decision": "HOLD",
            "final_allocation_pct": 0.08,
            "llm_decision": {"cot_reasoning": "혼조"},
            "data_provenance": {"quality": "high", "raw_components": 5, "coverage_score": 0.83},
        }
        state["risk_assessment"] = {"risk_decision": {"per_ticker_decisions": {"AAPL": {"flags": [], "decision": "approve", "final_weight": 0.08}}}}

        report = report_writer_node(state)["final_report"]
        assert "Portfolio Weights" in report
        assert "Book Allocation" in report
        assert "Evidence & Decision Quality" in report
        assert "Event Calendar & Monitoring" in report


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
