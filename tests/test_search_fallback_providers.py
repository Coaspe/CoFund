from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config.settings as settings_mod
from data_providers.exa_search_provider import ExaSearchProvider
from data_providers.tavily_search_provider import TavilySearchProvider


class _Resp:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def test_tavily_provider_maps_results_into_evidence(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(settings_mod, "_settings", None)
    provider = TavilySearchProvider(mode="live")
    provider._session.post = lambda *args, **kwargs: _Resp(
        200,
        {
            "results": [
                {
                    "url": "https://www.reuters.com/world/us/fed-signals-slower-rate-cuts-2026-02-01/",
                    "title": "Fed signals slower rate cuts",
                    "content": "Fed signaled slower rate cuts while markets repriced growth and risk assets.",
                    "published_date": "2026-02-01T00:00:00+00:00",
                }
            ]
        },
    )

    items = provider.collect_evidence(
        kind="macro_headline_context",
        ticker="NVDA",
        query="NVDA fed macro",
        max_items=2,
        resolver_path="tavily_fallback_macro",
    )

    assert len(items) == 1
    assert items[0]["resolver_path"] == "tavily_fallback_macro"
    assert items[0]["source"] == "www.reuters.com"
    assert "Fed signaled slower rate cuts" in items[0]["snippet"]


def test_exa_provider_maps_results_into_evidence(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(settings_mod, "_settings", None)
    provider = ExaSearchProvider(mode="live")
    provider._session.post = lambda *args, **kwargs: _Resp(
        200,
        {
            "results": [
                {
                    "url": "https://www.wsj.com/finance/stocks/nvda-valuation-premium-ai-demand-2026-02-03",
                    "title": "Nvidia valuation premium holds",
                    "text": "Nvidia keeps a valuation premium over peers as AI demand and margins remain resilient.",
                    "publishedDate": "2026-02-03T00:00:00+00:00",
                }
            ]
        },
    )

    items = provider.collect_evidence(
        kind="valuation_context",
        ticker="NVDA",
        query="NVDA valuation peers",
        max_items=2,
        resolver_path="exa_fallback_default",
    )

    assert len(items) == 1
    assert items[0]["resolver_path"] == "exa_fallback_default"
    assert items[0]["source"] == "www.wsj.com"
    assert "valuation premium" in items[0]["snippet"]
