"""
data_providers/alphavantage_provider.py — Alpha Vantage sentiment (optional)
=============================================================================
If ALPHAVANTAGE_API_KEY present, fetches news sentiment to augment/override
the rule-based NewsAPI score.
"""

from __future__ import annotations

from datetime import datetime, timezone

from schemas.common import make_evidence
from config.settings import get_settings
from data_providers.base import BaseProvider, ProviderError

_AV_BASE = "https://www.alphavantage.co/query"


class AlphaVantageProvider(BaseProvider):
    PROVIDER_NAME = "alphavantage"

    def __init__(self, **kwargs):
        self._api_key = get_settings().alphavantage_api_key
        super().__init__(**kwargs)

    @property
    def has_key(self) -> bool:
        return bool(self._api_key)

    def get_news_sentiment(self, ticker: str, days: int = 7) -> dict:
        as_of = datetime.now(timezone.utc).isoformat()

        if not self.has_key:
            return {"data": None, "evidence": [], "data_ok": False,
                    "limitations": ["ALPHAVANTAGE_API_KEY not set"], "as_of": as_of}

        url = _AV_BASE
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "apikey": self._api_key,
            "limit": 50,
        }

        try:
            data = self.get_json(url, params)
        except ProviderError as e:
            return {"data": None, "evidence": [], "data_ok": False,
                    "limitations": [f"Alpha Vantage error: {e}"], "as_of": as_of}

        feed = data.get("feed", [])
        if not feed:
            return {"data": None, "evidence": [], "data_ok": False,
                    "limitations": ["No sentiment data from Alpha Vantage"], "as_of": as_of}

        # Aggregate sentiment from ticker-specific sentiment scores
        scores = []
        for item in feed:
            for ts in item.get("ticker_sentiment", []):
                if ts.get("ticker", "").upper() == ticker.upper():
                    try:
                        scores.append(float(ts["ticker_sentiment_score"]))
                    except (KeyError, ValueError):
                        pass

        if not scores:
            return {"data": None, "evidence": [], "data_ok": False,
                    "limitations": ["No ticker-specific sentiment in AV feed"], "as_of": as_of}

        avg_score = round(sum(scores) / len(scores), 3)
        topics = list({t.get("topic", "") for item in feed for t in item.get("topics", []) if t.get("topic")})[:8]

        result = {
            "sentiment_score": avg_score,
            "article_count": len(feed),
            "key_topics": topics,
        }

        evidence = [
            make_evidence(metric="news_sentiment_score", value=avg_score,
                          source_name="AlphaVantage:NEWS_SENTIMENT", source_type="api",
                          quality=0.75, as_of=as_of),
            make_evidence(metric="article_count", value=len(feed),
                          source_name="AlphaVantage:NEWS_SENTIMENT", source_type="api",
                          quality=0.75, as_of=as_of),
        ]

        return {"data": result, "evidence": evidence, "data_ok": True,
                "limitations": [], "as_of": as_of}


if __name__ == "__main__":
    import json
    provider = AlphaVantageProvider()
    ticker = "AAPL"
    if provider.has_key:
        print(f"📊 Alpha Vantage — fetching {ticker} sentiment...")
    else:
        print("⚠️  No ALPHAVANTAGE_API_KEY")
    result = provider.get_news_sentiment(ticker)
    print(json.dumps(result, indent=2, default=str))
