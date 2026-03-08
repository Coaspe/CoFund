"""
data_providers/ir_press_release_provider.py
==========================================
Structured IR / press release catalyst collector.

Purpose:
  - Promote investor day / product launch catalysts from generic evidence
    into source-backed structured events.
  - Use existing search providers, but only on official wire/IR domains.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from data_providers.base import BaseProvider
from data_providers.exa_search_provider import ExaSearchProvider
from data_providers.tavily_search_provider import TavilySearchProvider


class IRPressReleaseProvider(BaseProvider):
    PROVIDER_NAME = "ir_press_release"

    WIRE_DOMAINS = ("prnewswire.com", "businesswire.com", "globenewswire.com")

    _EVENT_QUERIES = {
        "investor_day": "{ticker} investor day OR analyst day OR capital markets day press release",
        "product_cycle": "{ticker} launches OR unveils OR announces new product press release",
    }

    def __init__(self, *, mode: str = "mock", **kwargs):
        self.mode = mode
        super().__init__(**kwargs)
        self._tavily = TavilySearchProvider(mode=mode, cache=self._cache, rate_limiter=self._limiter)
        self._exa = ExaSearchProvider(mode=mode, cache=self._cache, rate_limiter=self._limiter)

    def get_catalyst_events(self, ticker: str, as_of: str = "") -> dict:
        as_of = as_of or datetime.now(timezone.utc).isoformat()
        if self.mode == "mock":
            return {
                "items": self._mock_items(ticker, as_of),
                "data_ok": False,
                "limitations": ["Mock mode — IR/press release catalyst snapshot"],
                "as_of": as_of,
            }

        items: list[dict[str, Any]] = []
        limitations: list[str] = []
        seen_urls: set[str] = set()

        for event_type, query_tmpl in self._EVENT_QUERIES.items():
            query = query_tmpl.format(ticker=ticker)
            rows = self._tavily.collect_evidence(
                kind="press_release_or_ir",
                ticker=ticker,
                query=query,
                recency_days=365,
                max_items=3,
                allowlist=self.WIRE_DOMAINS,
                desk="fundamental",
                resolver_path="ir_press_release_tavily",
            )
            if not rows:
                rows = self._exa.collect_evidence(
                    kind="press_release_or_ir",
                    ticker=ticker,
                    query=query,
                    recency_days=365,
                    max_items=3,
                    allowlist=self.WIRE_DOMAINS,
                    desk="fundamental",
                    resolver_path="ir_press_release_exa",
                )
            if not rows:
                limitations.append(f"No {event_type} wire/IR event found")
                continue
            for row in rows:
                url = str(row.get("url", "")).strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                items.append(
                    {
                        **row,
                        "catalyst_type": event_type,
                        "source_classification": "confirmed" if self._is_wire_domain(url) else "inferred",
                        "event_origin": "press_release",
                        "status": "confirmed" if self._is_wire_domain(url) else "inferred",
                    }
                )
                break

        return {
            "items": items,
            "data_ok": bool(items),
            "limitations": limitations,
            "as_of": as_of,
        }

    @classmethod
    def _is_wire_domain(cls, url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        return any(host == d or host.endswith("." + d) for d in cls.WIRE_DOMAINS)

    @staticmethod
    def _mock_items(ticker: str, as_of: str) -> list[dict[str, Any]]:
        return [
            {
                "title": f"{ticker} Investor Day announced",
                "url": f"https://www.prnewswire.com/mock/{ticker.lower()}-investor-day",
                "published_at": as_of,
                "snippet": "Investor Day event details",
                "source": "prnewswire.com",
                "kind": "press_release_or_ir",
                "desk": "fundamental",
                "ticker": ticker,
                "trust_tier": 0.8,
                "resolver_path": "ir_press_release_mock",
                "catalyst_type": "investor_day",
                "source_classification": "confirmed",
                "event_origin": "press_release",
                "status": "confirmed",
            },
            {
                "title": f"{ticker} launches new product line",
                "url": f"https://www.businesswire.com/mock/{ticker.lower()}-product-launch",
                "published_at": as_of,
                "snippet": "New product launch announcement",
                "source": "businesswire.com",
                "kind": "press_release_or_ir",
                "desk": "fundamental",
                "ticker": ticker,
                "trust_tier": 0.8,
                "resolver_path": "ir_press_release_mock",
                "catalyst_type": "product_cycle",
                "source_classification": "confirmed",
                "event_origin": "press_release",
                "status": "confirmed",
            },
        ]
