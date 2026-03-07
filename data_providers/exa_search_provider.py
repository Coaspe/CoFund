"""
data_providers/exa_search_provider.py
=====================================
Final-resort web search provider using Exa Search API.

Usage policy:
  - Called only after structured providers and Tavily fail to return items.
  - Produces evidence items from Exa search results.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

import requests

from api_usage_stats import record_api_request
from config.settings import get_settings
from data_providers.base import BaseProvider, ProviderError, is_rate_limit_error


class ExaSearchProvider(BaseProvider):
    PROVIDER_NAME = "exa_search"
    API_URL = "https://api.exa.ai/search"
    DEFAULT_SEARCH_TYPE = "auto"

    OFFICIAL_DOMAINS = (
        "sec.gov",
        "federalreserve.gov",
        "treasury.gov",
        "bea.gov",
        "bls.gov",
        "fred.stlouisfed.org",
        "nyfed.org",
        "eia.gov",
    )
    WIRE_DOMAINS = ("prnewswire.com", "businesswire.com", "globenewswire.com")
    NEWS_DOMAINS = ("reuters.com", "bloomberg.com", "wsj.com", "cnbc.com")

    def __init__(self, *, mode: str = "mock", **kwargs):
        self.mode = mode
        self._settings = get_settings()
        super().__init__(**kwargs)

    def collect_evidence(
        self,
        *,
        kind: str,
        ticker: str = "",
        query: str = "",
        recency_days: int = 30,
        max_items: int = 5,
        allowlist: list[str] | tuple[str, ...] | None = None,
        desk: str = "",
        resolver_path: str = "exa_web_fallback",
    ) -> list[dict[str, Any]]:
        if self.mode == "mock":
            return []

        api_key = str(self._settings.exa_api_key or "").strip()
        if not api_key:
            return []

        search_query = (query or ticker or kind).strip()
        if not search_query:
            return []

        allow = [str(x).strip().lower() for x in (allowlist or []) if str(x).strip()]
        payload: dict[str, Any] = {
            "query": search_query,
            "numResults": max(1, min(int(max_items or 5), 10)),
            "type": str(self._settings.exa_search_type or self.DEFAULT_SEARCH_TYPE).strip() or self.DEFAULT_SEARCH_TYPE,
            "text": True,
        }
        if allow:
            payload["includeDomains"] = allow
        category = self._category_for_kind(kind)
        if category:
            payload["category"] = category
        if int(recency_days or 0) > 0:
            start_dt = datetime.now(timezone.utc) - timedelta(days=int(recency_days))
            payload["startPublishedDate"] = start_dt.isoformat()

        try:
            data = self._post_search(payload=payload, api_key=api_key)
        except ProviderError:
            return []

        results = data.get("results")
        if not isinstance(results, list):
            return []

        now_iso = datetime.now(timezone.utc).isoformat()
        out: list[dict[str, Any]] = []
        for row in results[: max(1, int(max_items or 5))]:
            if not isinstance(row, dict):
                continue
            url = str(row.get("url") or "").strip()
            if not self._url_ok(url):
                continue
            host = (urlparse(url).hostname or "").lower()
            title = str(row.get("title") or f"{ticker or 'MARKET'} {kind} evidence").strip()[:220]
            snippet = " ".join(str(row.get("text") or row.get("summary") or "").split())[:420]
            published_at = str(row.get("publishedDate") or row.get("published_at") or "").strip() or now_iso
            source = host
            out.append(
                {
                    "url": url,
                    "title": title,
                    "published_at": published_at,
                    "snippet": snippet or f"Exa search evidence for: {search_query}",
                    "source": source,
                    "retrieved_at": now_iso,
                    "hash": self._evidence_hash(url=url, title=title, published_at=published_at, source=source),
                    "kind": kind,
                    "desk": desk,
                    "ticker": ticker,
                    "trust_tier": self._trust_tier_from_host(host),
                    "resolver_path": resolver_path,
                }
            )
        return out

    def _post_search(self, *, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        target = f"api.exa.ai/search type={payload.get('type', '')}"
        self._http_seq += 1
        req_id = self._http_seq
        self._log_http(f"[HTTP][START][{self.PROVIDER_NAME}][#{req_id}] {target}", "start")
        self._limiter.wait()
        started = time.perf_counter()
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
        }

        try:
            resp = self._session.post(self.API_URL, json=payload, headers=headers, timeout=self._timeout)
        except requests.RequestException as exc:
            record_api_request(self.PROVIDER_NAME, success=False, category="data")
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            self._log_http(
                f"[HTTP][FAIL ][{self.PROVIDER_NAME}][#{req_id}] {elapsed_ms}ms status=ERR {target}",
                "fail",
            )
            if is_rate_limit_error(exc):
                raise ProviderError(f"[{self.PROVIDER_NAME}] RATE_LIMIT: {exc}") from exc
            raise ProviderError(f"[{self.PROVIDER_NAME}] HTTP error: {exc}") from exc

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        status = int(resp.status_code)
        if status >= 400:
            record_api_request(self.PROVIDER_NAME, success=False, category="data")
            self._log_http(
                f"[HTTP][FAIL ][{self.PROVIDER_NAME}][#{req_id}] {elapsed_ms}ms status={status} {target}",
                "fail",
            )
            body = resp.text[:400]
            if status == 429 or is_rate_limit_error(body):
                raise ProviderError(f"[{self.PROVIDER_NAME}] RATE_LIMIT: status={status}")
            raise ProviderError(f"[{self.PROVIDER_NAME}] HTTP status {status}")

        try:
            data = resp.json()
        except ValueError as exc:
            record_api_request(self.PROVIDER_NAME, success=False, category="data")
            self._log_http(
                f"[HTTP][FAIL ][{self.PROVIDER_NAME}][#{req_id}] {elapsed_ms}ms status={status} {target}",
                "fail",
            )
            raise ProviderError(f"[{self.PROVIDER_NAME}] invalid JSON response") from exc

        record_api_request(self.PROVIDER_NAME, success=True, category="data")
        self._log_http(
            f"[HTTP][DONE ][{self.PROVIDER_NAME}][#{req_id}] {elapsed_ms}ms status={status} {target}",
            "done",
        )
        return data

    @staticmethod
    def _category_for_kind(kind: str) -> str:
        kind_l = str(kind or "").strip().lower()
        if kind_l in {"macro_headline_context", "press_release_or_ir", "catalyst_event_detail"}:
            return "news"
        if kind_l in {"valuation_context", "sector_performance"}:
            return "financial report"
        return ""

    @staticmethod
    def _url_ok(url: str) -> bool:
        try:
            parsed = urlparse(url)
            return parsed.scheme.lower() == "https" and bool(parsed.hostname)
        except ValueError:
            return False

    @staticmethod
    def _evidence_hash(*, url: str, title: str, published_at: str, source: str) -> str:
        raw = f"{url}|{title}|{published_at}|{source}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def _trust_tier_from_host(cls, host: str) -> float:
        h = str(host or "").lower()
        if any(h == d or h.endswith("." + d) for d in cls.OFFICIAL_DOMAINS):
            return 1.0
        if any(h == d or h.endswith("." + d) for d in cls.WIRE_DOMAINS):
            return 0.8
        if any(h == d or h.endswith("." + d) for d in cls.NEWS_DOMAINS):
            return 0.6
        return 0.4
