"""
data_providers/perplexity_search_provider.py
============================================
Final-resort web search provider using Perplexity API.

Usage policy:
  - Called only after higher-trust structured providers fail to return items.
  - Produces evidence items from citations/search results in API response.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import requests

from api_usage_stats import record_api_request
from config.settings import get_settings
from data_providers.base import BaseProvider, ProviderError, is_rate_limit_error


class PerplexitySearchProvider(BaseProvider):
    PROVIDER_NAME = "perplexity_search"
    API_URL = "https://api.perplexity.ai/chat/completions"
    DEFAULT_MODEL = "sonar"

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
        resolver_path: str = "perplexity_web_fallback",
    ) -> list[dict[str, Any]]:
        if self.mode == "mock":
            return []

        api_key = str(self._settings.perplexity_api_key or "").strip()
        if not api_key:
            return []

        model = str(self._settings.perplexity_model or self.DEFAULT_MODEL).strip() or self.DEFAULT_MODEL
        search_query = (query or ticker or kind).strip()
        if not search_query:
            return []

        payload = {
            "model": model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a financial web research agent. "
                        "Return concise factual output and include web citations."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"kind={kind}\n"
                        f"ticker={ticker}\n"
                        f"query={search_query}\n"
                        f"recency_days={max(1, int(recency_days or 30))}\n"
                        "Need authoritative and recent evidence."
                    ),
                },
            ],
        }

        try:
            data = self._post_chat(payload=payload, api_key=api_key)
        except ProviderError:
            return []
        content = self._extract_content(data)
        content_for_llm = self._strip_source_sections(content)
        search_meta = self._extract_search_meta(data)
        urls = self._extract_urls(data, search_meta)

        allow = tuple(str(x).strip().lower() for x in (allowlist or []) if str(x).strip())
        ordered_urls = self._prefer_allowlisted(urls, allow)

        now_iso = datetime.now(timezone.utc).isoformat()
        snippet = " ".join((content_for_llm or "").split())[:420]
        out: list[dict[str, Any]] = []
        for u in ordered_urls[: max(1, int(max_items or 5))]:
            meta = search_meta.get(u, {})
            host = (urlparse(u).hostname or "").lower()
            title = str(meta.get("title") or f"{ticker or 'MARKET'} {kind} evidence").strip()[:220]
            published_at = str(meta.get("published_at") or now_iso).strip() or now_iso
            source = str(meta.get("source") or host).strip() or host
            out.append(
                {
                    "url": u,
                    "title": title,
                    "published_at": published_at,
                    "snippet": snippet or f"Perplexity search evidence for: {search_query}",
                    "source": source,
                    "retrieved_at": now_iso,
                    "hash": self._evidence_hash(url=u, title=title, published_at=published_at, source=source),
                    "kind": kind,
                    "desk": desk,
                    "ticker": ticker,
                    "trust_tier": self._trust_tier_from_host(host),
                    "resolver_path": resolver_path,
                }
            )
        return out

    def _post_chat(self, *, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        model = str(payload.get("model", "")).strip()
        target = f"api.perplexity.ai/chat/completions model={model}"
        self._http_seq += 1
        req_id = self._http_seq
        self._log_http(f"[HTTP][START][{self.PROVIDER_NAME}][#{req_id}] {target}", "start")
        self._limiter.wait()
        started = time.perf_counter()
        headers = {
            "Authorization": f"Bearer {api_key}",
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
    def _extract_content(data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        msg = first.get("message")
        if isinstance(msg, dict):
            return str(msg.get("content", "") or "")
        return ""

    @staticmethod
    def _strip_source_sections(content: str) -> str:
        """
        Remove provider-side source appendix sections before passing text to internal LLM prompts.
        Example sections:
          - Citations:
          - Search Results:
        """
        text = str(content or "")
        if not text.strip():
            return ""

        kept: list[str] = []
        for raw in text.splitlines():
            s = raw.strip().lower()
            if (
                s.startswith("citations:")
                or s.startswith("search results:")
                or s.startswith("## citations")
                or s.startswith("## search results")
            ):
                break
            kept.append(raw)
        cleaned = "\n".join(kept).strip()
        return cleaned or text.strip()

    @staticmethod
    def _extract_search_meta(data: dict[str, Any]) -> dict[str, dict[str, str]]:
        meta: dict[str, dict[str, str]] = {}
        for key in ("search_results", "results", "sources"):
            rows = data.get(key)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                u = str(row.get("url") or row.get("link") or "").strip()
                if not PerplexitySearchProvider._url_ok(u):
                    continue
                meta[u] = {
                    "title": str(row.get("title", "") or "").strip(),
                    "published_at": str(row.get("date") or row.get("published_at") or "").strip(),
                    "source": str(row.get("source") or "").strip(),
                }
        return meta

    @staticmethod
    def _extract_urls(data: dict[str, Any], search_meta: dict[str, dict[str, str]]) -> list[str]:
        urls: list[str] = []
        seen: set[str] = set()

        citations = data.get("citations")
        if isinstance(citations, list):
            for c in citations:
                u = ""
                if isinstance(c, str):
                    u = c.strip()
                elif isinstance(c, dict):
                    u = str(c.get("url") or c.get("link") or "").strip()
                if PerplexitySearchProvider._url_ok(u) and u not in seen:
                    seen.add(u)
                    urls.append(u)

        for u in search_meta.keys():
            if u not in seen:
                seen.add(u)
                urls.append(u)
        return urls

    @staticmethod
    def _prefer_allowlisted(urls: list[str], allowlist: tuple[str, ...]) -> list[str]:
        if not allowlist:
            return urls

        def _in_allow(url: str) -> bool:
            host = (urlparse(url).hostname or "").lower()
            return any(host == d or host.endswith("." + d) for d in allowlist)

        preferred = [u for u in urls if _in_allow(u)]
        others = [u for u in urls if not _in_allow(u)]
        return preferred + others

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
