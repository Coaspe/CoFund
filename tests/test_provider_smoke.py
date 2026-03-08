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


def test_fmp_no_key():
    """FMP provider without API key → mock fundamentals."""
    from data_providers.fmp_provider import FMPProvider
    provider = FMPProvider()
    result = provider.get_fundamentals("AAPL")
    assert "data" in result
    assert "evidence" in result
    assert isinstance(result["data"], dict)
    assert "revenue_growth" in result["data"] or "roe" in result["data"]


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
