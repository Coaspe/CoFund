import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.macro_agent import macro_analyst_run


def _build_state(universe):
    return {
        "user_request": "미국의 이란 공습에 따른 미증시 전망과 헤지 전략",
        "universe": universe,
        "asset_type_by_ticker": {
            "SPY": "ETF",
            "QQQ": "ETF",
            "GLD": "COMMODITY",
            "TLT": "BOND",
            "XLE": "ETF",
        },
    }


def test_macro_output_includes_portfolio_translation_fields():
    state = _build_state(["SPY", "QQQ", "GLD", "TLT", "XLE"])
    ind = {
        "yield_curve_spread": 0.80,
        "hy_oas": 250,
        "cpi_yoy": 2.4,
        "pmi": 56,
        "fed_funds_rate": 2.0,
        "gdp_growth": 2.5,
    }

    out = macro_analyst_run("SPY", ind, state=state, focus_areas=["연준 정책 경로"])

    assert "transmission_map" in out and isinstance(out["transmission_map"], dict)
    assert "portfolio_implications" in out and isinstance(out["portfolio_implications"], dict)
    assert "monitoring_triggers" in out and isinstance(out["monitoring_triggers"], list)
    assert set(out["transmission_map"]) >= {"growth_beta", "policy_rates", "credit", "inflation_real_assets", "liquidity"}
    assert out["portfolio_implications"]["context"]["main_ticker"] == "SPY"
    assert len(out["portfolio_implications"]["targets"]) == 5


def test_macro_portfolio_implications_distinguish_growth_vs_duration():
    state = _build_state(["SPY", "QQQ", "GLD", "TLT", "XLE"])
    ind = {
        "yield_curve_spread": 0.80,
        "hy_oas": 250,
        "cpi_yoy": 2.4,
        "pmi": 56,
        "fed_funds_rate": 2.0,
        "gdp_growth": 2.5,
    }

    out = macro_analyst_run("SPY", ind, state=state)
    targets = {item["ticker"]: item for item in out["portfolio_implications"]["targets"]}

    assert targets["QQQ"]["bucket"] == "growth_equity"
    assert targets["QQQ"]["stance"] in {"lean_overweight", "overweight"}
    assert targets["TLT"]["bucket"] == "duration"
    assert targets["TLT"]["stance"] in {"neutral", "lean_underweight", "underweight"}


def test_macro_tail_risk_prefers_defensive_hedges_and_commodity_trigger():
    state = _build_state(["SPY", "QQQ", "GLD", "TLT", "XLE"])
    ind = {
        "yield_curve_spread": -0.40,
        "hy_oas": 600,
        "cpi_yoy": 2.2,
        "pmi": 44,
        "fed_funds_rate": 3.0,
        "gdp_growth": 0.5,
    }

    out = macro_analyst_run(
        "SPY",
        ind,
        state=state,
        focus_areas=["WTI/Brent 가격 충격", "연준(Fed)의 인플레이션 우려"],
    )
    preferred_hedges = out["portfolio_implications"]["preferred_hedges"]
    trigger_names = {item["name"] for item in out["monitoring_triggers"]}

    assert out["tail_risk_warning"] is True
    assert "GLD" in preferred_hedges or "TLT" in preferred_hedges
    assert "Commodity shock" in trigger_names
    assert out["transmission_map"]["commodity_shock_watch"]["signal"] == "watch"


def test_macro_uses_market_and_rates_pricing_inputs_directly():
    state = _build_state(["SPY", "QQQ", "GLD", "TLT", "XLE"])
    ind = {
        "yield_curve_spread": -0.10,
        "dgs2": 4.2,
        "dgs10": 4.1,
        "fed_funds_rate": 4.75,
        "cuts_priced_proxy_2y_ffr_bp": 55.0,
        "hy_oas": 380,
        "cpi_yoy": 2.8,
        "pmi": 50,
        "dollar_index": 121.0,
        "vix_level": 24.5,
        "wti_spot": 82.0,
        "brent_spot": 85.0,
    }

    out = macro_analyst_run(
        "SPY",
        ind,
        state=state,
        focus_areas=["WTI/Brent 가격", "Fed 정책 경로"],
    )
    transmission = out["transmission_map"]
    trigger_names = {item["name"] for item in out["monitoring_triggers"]}

    assert transmission["rates_pricing"]["signal"] == "dovish_pricing"
    assert transmission["volatility"]["current_value"] == 24.5
    assert transmission["usd"]["current_value"] == 121.0
    assert transmission["commodity_shock_watch"]["current_value"] == 82.0
    assert "Volatility regime shift" in trigger_names
    assert "Dollar regime shift" in trigger_names


def test_macro_prefers_true_fed_funds_futures_pricing_when_available():
    state = _build_state(["SPY", "QQQ", "GLD", "TLT", "XLE"])
    ind = {
        "yield_curve_spread": 0.10,
        "dgs2": 4.2,
        "fed_funds_rate": 4.25,
        "cuts_priced_proxy_2y_ffr_bp": 5.0,
        "fed_funds_futures_front_implied_rate": 4.10,
        "fed_funds_futures_3m_implied_rate": 3.90,
        "fed_funds_futures_6m_implied_rate": 3.60,
        "fed_funds_futures_implied_change_6m_bp": -50.0,
        "hy_oas": 320,
        "cpi_yoy": 2.4,
        "pmi": 52,
    }

    out = macro_analyst_run("SPY", ind, state=state, focus_areas=["Fed policy path"])
    rates = out["transmission_map"]["rates_pricing"]
    policy_trigger = next(item for item in out["monitoring_triggers"] if item["name"] == "Policy repricing")

    assert rates["signal"] == "dovish_pricing"
    assert rates["current_value"] == -50.0
    assert policy_trigger["metric"] == "fed_funds_futures_implied_change_6m_bp"


def test_macro_dualizes_rates_pricing_with_sofr_and_basis():
    state = _build_state(["SPY", "QQQ", "GLD", "TLT", "XLE"])
    ind = {
        "yield_curve_spread": 0.10,
        "dgs2": 4.2,
        "fed_funds_rate": 4.25,
        "sofr_rate": 4.20,
        "fed_funds_futures_front_implied_rate": 4.10,
        "fed_funds_futures_3m_implied_rate": 3.90,
        "fed_funds_futures_6m_implied_rate": 3.70,
        "fed_funds_futures_implied_change_6m_bp": -40.0,
        "sofr_futures_front_implied_rate": 4.15,
        "sofr_futures_3m_implied_rate": 4.00,
        "sofr_futures_6m_implied_rate": 3.85,
        "sofr_futures_implied_change_6m_bp": -30.0,
        "hy_oas": 320,
        "cpi_yoy": 2.4,
        "pmi": 52,
    }

    out = macro_analyst_run("SPY", ind, state=state, focus_areas=["SOFR/OIS implied path", "Fed policy path"])
    rates = out["transmission_map"]["rates_pricing"]
    policy_trigger = next(item for item in out["monitoring_triggers"] if item["name"] == "Policy repricing")
    basis_trigger = next(item for item in out["monitoring_triggers"] if item["name"] == "Rates basis divergence")

    assert rates["primary_metric"] == "sofr_futures_implied_change_6m_bp"
    assert rates["secondary_metric"] == "fed_funds_futures_implied_change_6m_bp"
    assert rates["current_value"] == -30.0
    assert rates["basis_bp"] == 15.0
    assert policy_trigger["metric"] == "sofr_futures_implied_change_6m_bp"
    assert basis_trigger["metric"] == "sofr_ff_6m_basis_bp"
