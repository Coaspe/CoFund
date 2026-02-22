"""
data_providers/newsapi_provider.py — NewsAPI headline sentiment provider
========================================================================
Fetches ticker-related news headlines from NewsAPI.
Computes article_count, key_topics, news_sentiment_score via rule-based lexicon.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from schemas.common import make_evidence
from config.settings import get_settings
from data_providers.base import BaseProvider, ProviderError

_NEWSAPI_BASE = "https://newsapi.org/v2"

# Simple finance sentiment lexicon (positive / negative words)
_POS_WORDS = {
    "surge", "surges", "rally", "rallies", "gain", "gains", "up", "bull", "bullish",
    "beat", "beats", "record", "growth", "profit", "upgrade", "upgrades", "positive",
    "strong", "outperform", "boost", "boosts", "rise", "rises", "soar", "soars",
}
_NEG_WORDS = {
    "drop", "drops", "fall", "falls", "crash", "plunge", "loss", "losses", "down",
    "bear", "bearish", "miss", "misses", "decline", "declines", "downgrade", "downgrades",
    "negative", "weak", "warning", "risk", "debt", "bankruptcy", "lawsuit", "investigation",
    "cut", "cuts", "sell", "selloff", "slump", "slumps", "fear", "fears",
}
_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "and",
    "in", "that", "it", "for", "on", "with", "as", "at", "by", "from", "or", "but",
    "not", "this", "which", "its", "has", "have", "had", "will", "would", "can", "could",
    "s", "t", "d", "he", "she", "they", "their", "his", "her",
}


class NewsAPIProvider(BaseProvider):
    PROVIDER_NAME = "newsapi"

    def __init__(self, **kwargs):
        self._api_key = get_settings().newsapi_api_key
        super().__init__(**kwargs)

    @property
    def has_key(self) -> bool:
        return bool(self._api_key)

    def search_ticker_news(self, ticker: str, days: int = 7, language: str = "en") -> dict:
        as_of = datetime.now(timezone.utc).isoformat()

        if not self.has_key:
            return self._mock_sentiment(ticker, as_of)

        from_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        url = f"{_NEWSAPI_BASE}/everything"
        params = {
            "q": ticker,
            "from": from_date,
            "language": language,
            "sortBy": "relevancy",
            "pageSize": 50,
            "apiKey": self._api_key,
        }

        evidence: list = []
        limitations: list = []

        try:
            data = self.get_json(url, params)
            articles = data.get("articles", [])
            total = data.get("totalResults", len(articles))
        except ProviderError as e:
            limitations.append(f"NewsAPI error: {e}")
            return self._mock_sentiment(ticker, as_of)

        article_count = min(total, 100)

        # Extract titles for analysis
        titles = [a.get("title", "") for a in articles if a.get("title")]

        # Key topics from titles
        key_topics = self._extract_topics(titles, top_n=8)

        # Sentiment score from finance lexicon
        sentiment_score = self._compute_sentiment(titles)

        evidence.append(make_evidence(
            metric="article_count", value=article_count,
            source_name="NewsAPI", source_type="api", quality=0.70, as_of=as_of,
        ))
        evidence.append(make_evidence(
            metric="news_sentiment_score", value=sentiment_score,
            source_name="NewsAPI:heuristic", source_type="model", quality=0.50, as_of=as_of,
            note="Rule-based finance lexicon score",
        ))

        indicators = {
            "news_sentiment_score": sentiment_score,
            "article_count": article_count,
            "key_topics": key_topics,
            "upcoming_events": [],  # Not available from NewsAPI
        }

        return {
            "data": indicators,
            "evidence": evidence,
            "data_ok": article_count > 0,
            "limitations": limitations,
            "as_of": as_of,
        }

    def _extract_topics(self, titles: List[str], top_n: int = 8) -> List[str]:
        words: list = []
        for title in titles:
            tokens = re.findall(r"[a-zA-Z]{3,}", title.lower())
            words.extend(t for t in tokens if t not in _STOP_WORDS)
        freq = Counter(words).most_common(top_n)
        return [w for w, _ in freq]

    def _compute_sentiment(self, titles: List[str]) -> float:
        if not titles:
            return 0.0
        pos_count = 0
        neg_count = 0
        for title in titles:
            words = set(re.findall(r"[a-zA-Z]+", title.lower()))
            pos_count += len(words & _POS_WORDS)
            neg_count += len(words & _NEG_WORDS)
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return round((pos_count - neg_count) / total, 2)

    @staticmethod
    def _mock_sentiment(ticker: str, as_of: str) -> dict:
        rng = random.Random(hash(ticker) % (2**31))
        indicators = {
            "news_sentiment_score": round(rng.uniform(-0.6, 0.6), 2),
            "article_count": rng.randint(10, 200),
            "key_topics": rng.sample(["earnings", "growth", "AI", "revenue", "market", "regulation"], k=4),
            "upcoming_events": [],
        }
        evidence = [
            make_evidence(metric="article_count", value=indicators["article_count"],
                          source_name="mock", quality=0.3, as_of=as_of),
            make_evidence(metric="news_sentiment_score", value=indicators["news_sentiment_score"],
                          source_name="mock", quality=0.3, as_of=as_of),
        ]
        return {"data": indicators, "evidence": evidence, "data_ok": False,
                "limitations": ["NewsAPI unavailable; using mock sentiment"], "as_of": as_of}


if __name__ == "__main__":
    import json
    provider = NewsAPIProvider()
    ticker = "AAPL"
    if provider.has_key:
        print(f"📰 NewsAPI — fetching {ticker} headlines...")
    else:
        print(f"⚠️  No NEWSAPI_API_KEY — mock for {ticker}")
    result = provider.search_ticker_news(ticker)
    print(json.dumps(result, indent=2, default=str))
