"""
tests/test_provider_smoke.py — Provider smoke tests
=====================================================
Verifies every provider returns valid structure even without API keys.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_fred_no_key():
    """FRED provider without API key → mock data, data_ok=False."""
    from data_providers.fred_provider import FREDProvider
    provider = FREDProvider()
    if provider.has_key:
        print("  (FRED key present — testing live mode)")
        snapshot = provider.get_macro_snapshot()
        # CI/로컬 네트워크(DNS/방화벽) 영향으로 live 호출이 실패할 수 있으므로
        # key가 있어도 graceful fallback(data_ok=False)을 허용한다.
        if snapshot["data_ok"] is False:
            joined = " ".join(snapshot.get("limitations", [])).lower()
            assert any(k in joined for k in ("http error", "name resolution", "dns", "unavailable"))
        else:
            assert snapshot["data_ok"] is True
    else:
        snapshot = provider._mock_snapshot("2026-01-01T00:00:00Z")
        assert snapshot["data_ok"] is False
        assert len(snapshot["data"]) >= 5
    assert "evidence" in snapshot
    assert "limitations" in snapshot


def test_market_macro_provider_mock():
    """market_data_provider macro snapshot mock → dated structured fields."""
    from data_providers.market_data_provider import fetch_macro_market_indicators
    data, evidence, meta = fetch_macro_market_indicators(mode="mock", as_of="2026-01-01T00:00:00Z", seed=42)
    assert set(data) >= {"wti_front_month", "brent_front_month", "vix_index"}
    assert len(evidence) == 3
    assert meta["data_ok"] is False


def test_fed_funds_futures_provider_mock():
    from data_providers.fed_funds_futures_provider import FedFundsFuturesProvider
    provider = FedFundsFuturesProvider()
    result = provider.get_curve(months=6, as_of="2026-01-01T00:00:00Z")
    data = result["data"]
    assert "fed_funds_futures_curve" in data
    assert len(data["fed_funds_futures_curve"]) >= 3
    assert "fed_funds_futures_front_implied_rate" in data
    assert "fed_funds_futures_implied_change_6m_bp" in data
    assert "evidence" in result and len(result["evidence"]) >= 4


def test_sofr_futures_provider_mock():
    from data_providers.sofr_futures_provider import SofrFuturesProvider
    provider = SofrFuturesProvider()
    result = provider.get_curve(contracts=5, as_of="2026-01-01T00:00:00Z")
    data = result["data"]
    assert "sofr_futures_curve" in data
    assert len(data["sofr_futures_curve"]) >= 3
    assert "sofr_futures_front_implied_rate" in data
    assert "sofr_futures_implied_change_6m_bp" in data
    assert "evidence" in result and len(result["evidence"]) >= 4


def test_macro_event_calendar_provider_mock():
    from data_providers.macro_event_calendar_provider import MacroEventCalendarProvider

    result = MacroEventCalendarProvider._mock_calendar("2026-03-08T00:00:00Z")
    items = result["items"]
    assert len(items) >= 3
    assert any(item["type"] == "fomc" for item in items)
    assert any(item["type"] == "gdp" for item in items)
    assert any(item["type"] == "pce" for item in items)
    assert result["evidence"]


def test_fmp_no_key():
    """FMP provider without API key → mock fundamentals."""
    from data_providers.fmp_provider import FMPProvider
    provider = FMPProvider()
    result = provider.get_fundamentals("AAPL")
    assert "data" in result
    assert "evidence" in result
    assert isinstance(result["data"], dict)
    assert "revenue_growth" in result["data"] or "roe" in result["data"]


def test_fmp_mock_contains_consensus_and_catalyst_fields():
    from data_providers.fmp_provider import FMPProvider
    result = FMPProvider._mock_fundamentals("AAPL", "2026-03-08T00:00:00Z")
    data = result["data"]
    assert "current_price" in data
    assert "fcf_history" in data and len(data["fcf_history"]) == 5
    assert data["next_earnings_date"] == "2026-04-30"
    assert "next_eps_estimate" in data
    assert "price_target_consensus" in data
    assert "price_target_upside_pct" in data


def test_fmp_uses_income_statement_ebitda_not_balance_sheet_placeholder(monkeypatch):
    from data_providers.fmp_provider import FMPProvider

    provider = FMPProvider()
    provider._api_key = "test-key"

    payloads = {
        "profile": [{"sector": "Technology", "volAvg": 123456, "mktCap": 2_000_000, "price": 100.0}],
        "key-metrics-ttm": [{
            "peRatioTTM": 20.0,
            "priceToSalesRatioTTM": 6.0,
            "freeCashFlowYieldTTM": 0.04,
            "roeTTM": 0.25,
            "debtToEquityTTM": 0.7,
        }],
        "ratios-ttm": [{"operatingProfitMarginTTM": 0.20}],
        "income-statement": [
            {"revenue": 400_000, "operatingIncome": 80_000, "ebitda": 95_000, "interestExpense": 5_000, "netIncome": 50_000},
            {"revenue": 360_000, "operatingIncome": 70_000, "ebitda": 88_000, "interestExpense": 4_500, "netIncome": 42_000},
        ],
        "cash-flow-statement": [
            {"freeCashFlow": 70_000},
            {"freeCashFlow": 66_000},
        ],
        "balance-sheet-statement": [{
            "totalAssets": 999_999,
            "totalLiabilities": 200_000,
            "totalCurrentAssets": 150_000,
            "totalCurrentLiabilities": 50_000,
            "retainedEarnings": 200_000,
            "totalDebt": 80_000,
            "cashAndCashEquivalents": 30_000,
        }],
        "earnings": [{"date": "2026-04-30", "epsEstimated": 1.8, "revenueEstimated": 405_000, "lastUpdated": "2026-03-08"}],
        "analyst-estimates": [{
            "date": "2027-09-30",
            "revenueAvg": 450_000,
            "ebitdaAvg": 110_000,
            "epsAvg": 8.2,
            "numAnalystsRevenue": 22,
            "numAnalystsEps": 24,
        }],
        "price-target-consensus": [{"targetConsensus": 118.0, "targetMedian": 116.0, "targetHigh": 130.0, "targetLow": 95.0}],
        "ratings-snapshot": [{"rating": "B", "overallScore": 3, "discountedCashFlowScore": 3}],
    }

    def fake_get_first_available(endpoints, ticker, limitations, **extra_params):
        assert ticker == "AAPL"
        for endpoint in endpoints:
            if endpoint in payloads:
                return payloads[endpoint]
        return []

    monkeypatch.setattr(provider, "_get_first_available", fake_get_first_available)
    result = provider.get_fundamentals("AAPL", as_of="2026-03-08T00:00:00Z")
    data = result["data"]
    assert data["ebitda"] == 95_000
    assert data["ebitda"] != data["total_assets"]
    assert data["price_target_upside_pct"] == 18.0
    assert data["earnings_in_days"] == 53


def test_fmp_analyst_estimate_prefers_nearest_future_fy(monkeypatch):
    from data_providers.fmp_provider import FMPProvider

    provider = FMPProvider()
    financials = {}
    payloads = {
        "analyst-estimates": [
            {"date": "2025-09-30", "revenueAvg": 380_000, "ebitdaAvg": 90_000, "epsAvg": 7.1, "numAnalystsRevenue": 20, "numAnalystsEps": 22},
            {"date": "2026-09-30", "revenueAvg": 450_000, "ebitdaAvg": 110_000, "epsAvg": 8.2, "numAnalystsRevenue": 22, "numAnalystsEps": 24},
            {"date": "2027-09-30", "revenueAvg": 490_000, "ebitdaAvg": 120_000, "epsAvg": 9.0, "numAnalystsRevenue": 19, "numAnalystsEps": 21},
        ]
    }

    def fake_get_first_available(endpoints, ticker, limitations, **extra_params):
        for endpoint in endpoints:
            if endpoint in payloads:
                return payloads[endpoint]
        return []

    monkeypatch.setattr(provider, "_get_first_available", fake_get_first_available)
    provider._attach_estimate_context(financials, "AAPL", [], "2026-03-08T00:00:00Z")
    assert financials["analyst_estimate_fy"] == "2026-09-30"
    assert financials["analyst_eps_estimate_next_fy"] == 8.2


def test_fmp_mock_peer_context_contains_peer_medians():
    from data_providers.fmp_provider import FMPProvider
    result = FMPProvider._mock_peer_context("AAPL", "2026-03-08T00:00:00Z", max_peers=4)
    data = result["data"]
    assert data["status"] == "ok"
    assert len(data["peers"]) == 4
    assert data["peer_median_pe"] is not None
    assert data["peer_symbols"]


def test_yahoo_structured_mock_estimate_revision_contains_revision_fields():
    from data_providers.yahoo_structured_provider import YahooStructuredProvider
    data = YahooStructuredProvider._mock_estimate_revision("AAPL", "2026-03-08T00:00:00Z")
    assert data["status"] == "ok"
    assert data["estimate_source"] == "mock"
    assert "0q" in data["estimate_periods"]
    assert data["estimate_periods"]["0q"]["eps_revision_30d_pct"] is not None
    assert data["estimate_periods"]["0q"]["revision_state"] in {"improving", "stable", "deteriorating"}


def test_yahoo_structured_mock_ownership_contains_concentration_fields():
    from data_providers.yahoo_structured_provider import YahooStructuredProvider
    data = YahooStructuredProvider._mock_ownership_snapshot("AAPL", "2026-03-08T00:00:00Z")
    assert data["status"] == "ok"
    assert data["institutional_top10_pct"] is not None
    assert data["institutions_percent_held"] is not None
    assert data["incremental_buyer_seller_map"]["buyers"]
    assert data["insider_net_activity"] in {"buying", "selling", "neutral"}


def test_yahoo_structured_mock_history_contains_real_valuation_points():
    from data_providers.yahoo_structured_provider import YahooStructuredProvider
    data = YahooStructuredProvider._mock_fundamental_history_snapshot("AAPL", "2026-03-08T00:00:00Z")
    hist = data["valuation_history_real"]
    assert hist["status"] == "ok"
    assert hist["source"] == "mock"
    assert len(hist["valuation_points"]) >= 4
    assert hist["pe_ratios"]
    assert data["ttm_revenue_real"] is not None
    assert data["eps_ttm_real"] is not None


def test_ir_press_release_provider_mock_returns_structured_events():
    from data_providers.ir_press_release_provider import IRPressReleaseProvider

    provider = IRPressReleaseProvider(mode="mock")
    result = provider.get_catalyst_events("AAPL", as_of="2026-03-08T00:00:00Z")
    assert result["items"]
    event_types = {item["catalyst_type"] for item in result["items"]}
    assert {"investor_day", "product_cycle"} <= event_types
    assert all(item["source_classification"] == "confirmed" for item in result["items"])


def test_sec_ownership_identity_prefers_cik_query(monkeypatch):
    from data_providers.sec_edgar_provider import SECEdgarProvider

    provider = SECEdgarProvider()
    monkeypatch.setattr(provider, "_user_agent", "test-agent")

    calls = []

    def fake_search(ticker, forms, startdt="2023-01-01", size=5, query=None):
        calls.append(query)
        if query == '"0000320193"':
            return [{"_source": {"file_url": "/x", "display_names": ["APPLE INC  (AAPL)  (CIK 0000320193)"], "file_date": "2026-03-01"}}]
        return []

    monkeypatch.setattr(provider, "_resolve_cik", lambda ticker: "0000320193")
    monkeypatch.setattr(provider, "_search_filings", fake_search)
    out = provider.get_ownership_identity("AAPL", as_of="2026-03-08T00:00:00Z")
    assert calls and calls[0] == '"0000320193"'
    assert out["data_ok"] is True
    assert out["items"][0]["resolver_path"] == "sec_forms"


def test_sec_8k_exhibits_classify_item_and_exhibit(monkeypatch):
    from data_providers.sec_edgar_provider import SECEdgarProvider

    provider = SECEdgarProvider()
    monkeypatch.setattr(provider, "_user_agent", "test-agent")
    monkeypatch.setattr(
        provider,
        "_search_filings",
        lambda ticker, forms, startdt="2023-01-01", size=6, query=None: [
            {
                "_source": {
                    "file_url": "https://www.sec.gov/Archives/edgar/data/0000320193/example8k.htm",
                    "display_names": ["APPLE INC 8-K"],
                    "file_date": "2026-03-01",
                    "description": "Current report",
                }
            }
        ],
    )
    monkeypatch.setattr(
        provider,
        "_fetch_filing_text",
        lambda url: "Item 8.01 Other Events. Exhibit 99.1 press release discussing pricing changes and discount actions.",
    )
    out = provider.get_8k_exhibits("AAPL", as_of="2026-03-08T00:00:00Z")
    assert out["data_ok"] is True
    item = out["items"][0]
    assert item["catalyst_type"] == "pricing_reset"
    assert item["source_classification"] == "confirmed"
    assert "8.01" in item["filing_items"]
    assert "99.1" in item["exhibit_codes"]


def test_fmp_earnings_context_builds_real_surprise_history(monkeypatch):
    from data_providers.fmp_provider import FMPProvider

    provider = FMPProvider()
    financials = {}
    payloads = {
        "earnings": [
            {"date": "2026-04-30", "epsEstimated": 1.8, "revenueEstimated": 405_000, "lastUpdated": "2026-03-08"},
            {"date": "2026-01-30", "epsActual": 2.0, "epsEstimated": 1.8, "revenueActual": 410_000, "revenueEstimated": 400_000},
            {"date": "2025-10-30", "epsActual": 1.9, "epsEstimated": 1.85, "revenueActual": 402_000, "revenueEstimated": 398_000},
            {"date": "2025-07-30", "epsActual": 1.7, "epsEstimated": 1.75, "revenueActual": 390_000, "revenueEstimated": 395_000},
            {"date": "2025-04-30", "epsActual": 1.6, "epsEstimated": 1.55, "revenueActual": 385_000, "revenueEstimated": 380_000},
        ]
    }

    def fake_get_first_available(endpoints, ticker, limitations, **extra_params):
        for endpoint in endpoints:
            if endpoint in payloads:
                return payloads[endpoint]
        return []

    monkeypatch.setattr(provider, "_get_first_available", fake_get_first_available)
    provider._attach_earnings_context(financials, "AAPL", [], "2026-03-08T00:00:00Z")
    assert financials["earnings_beat_rate_4q"] == 75.0
    assert len(financials["earnings_surprise_history"]) == 4
    assert financials["eps_surprise_avg_4q"] is not None


def test_sec_no_agent():
    """SEC EDGAR without SEC_USER_AGENT → graceful None."""
    from data_providers.sec_edgar_provider import SECEdgarProvider
    provider = SECEdgarProvider()
    result = provider.get_sec_flags("AAPL")
    assert "evidence" in result
    if not provider.has_agent:
        assert result["data_ok"] is False


def test_newsapi_no_key():
    """NewsAPI without API key → mock sentiment."""
    from data_providers.newsapi_provider import NewsAPIProvider
    provider = NewsAPIProvider()
    result = provider.search_ticker_news("AAPL")
    assert "data" in result
    assert "news_sentiment_score" in result["data"]
    assert "article_count" in result["data"]


def test_sentiment_market_provider_mock_returns_structured_snapshot():
    from data_providers.sentiment_market_provider import SentimentMarketProvider

    provider = SentimentMarketProvider(mode="mock")
    result = provider.get_snapshot("SPY", as_of="2026-03-08T00:00:00Z")
    data = result["data"]
    assert data["vix_level"] is not None
    assert data["vvix_level"] is not None
    assert data["put_call_oi_ratio"] is not None
    assert data["vix_term_structure"] in {"contango", "backwardation", "flat"}
    assert result["evidence"]


def test_alphavantage_no_key():
    """Alpha Vantage without API key → graceful None."""
    from data_providers.alphavantage_provider import AlphaVantageProvider
    provider = AlphaVantageProvider()
    result = provider.get_news_sentiment("AAPL")
    if not provider.has_key:
        assert result["data_ok"] is False
    assert "evidence" in result


def test_twelvedata_no_key():
    """TwelveData without API key → mock prices."""
    from data_providers.twelvedata_provider import TwelveDataProvider
    provider = TwelveDataProvider()
    result = provider.get_price_series("AAPL")
    assert result["data"] is not None
    assert len(result["data"]) > 0
    if not provider.has_key:
        assert result["data_ok"] is False


def test_datahub_mock_mode():
    """DataHub in mock mode → all methods return valid data."""
    from data_providers.data_hub import DataHub
    hub = DataHub(run_id="test", mode="mock")

    macro, ev, meta = hub.get_macro_indicators(seed=42)
    assert "yield_curve_spread" in macro
    assert "dollar_index" in macro
    assert "vix_level" in macro
    assert "wti_spot" in macro
    assert "cuts_priced_proxy_2y_ffr_bp" in macro
    assert "sofr_rate" in macro
    assert len(ev) > 0

    funda, ev2, meta2 = hub.get_fundamentals("AAPL", seed=42)
    assert "revenue_growth" in funda or "roe" in funda

    sec, ev3, meta3 = hub.get_sec_flags("AAPL")
    assert sec is None  # Mock mode

    senti, ev4, meta4 = hub.get_news_sentiment("AAPL", seed=42)
    assert "news_sentiment_score" in senti

    import numpy as np
    prices, ev5, meta5 = hub.get_price_series("AAPL", seed=42)
    assert isinstance(prices, np.ndarray)
    assert len(prices) > 100


if __name__ == "__main__":
    test_fred_no_key()
    print("✅ test_fred_no_key PASSED")
    test_fmp_no_key()
    print("✅ test_fmp_no_key PASSED")
    test_market_macro_provider_mock()
    print("✅ test_market_macro_provider_mock PASSED")
    test_fed_funds_futures_provider_mock()
    print("✅ test_fed_funds_futures_provider_mock PASSED")
    test_sofr_futures_provider_mock()
    print("✅ test_sofr_futures_provider_mock PASSED")
    test_sec_no_agent()
    print("✅ test_sec_no_agent PASSED")
    test_newsapi_no_key()
    print("✅ test_newsapi_no_key PASSED")
    test_alphavantage_no_key()
    print("✅ test_alphavantage_no_key PASSED")
    test_twelvedata_no_key()
    print("✅ test_twelvedata_no_key PASSED")
    test_datahub_mock_mode()
    print("✅ test_datahub_mock_mode PASSED")
