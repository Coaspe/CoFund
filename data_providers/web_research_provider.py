"""
data_providers/web_research_provider.py — Web research evidence collector
=========================================================================
Collects search/fetch snippets without LLM.

Security hardening:
  - HTTPS only
  - redirect-chain final domain must pass allowlist
  - private/loopback/link-local/reserved IP blocked
  - timeout and max-bytes enforced
  - fetch allowed only from:
      * NewsAPI result URLs passing allowlist
      * official URLs supplied by structured providers
"""

from __future__ import annotations

import hashlib
import ipaddress
import random
import re
import socket
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urljoin, urlparse

from api_usage_stats import record_api_request
from config.settings import get_settings
from data_providers.base import BaseProvider, ProviderError


class WebResearchProvider(BaseProvider):
    PROVIDER_NAME = "web_research"
    NEWSAPI_MAX_RECENCY_DAYS = 30

    ALLOWLIST_MACRO = (
        "fred.stlouisfed.org", "nyfed.org", "bls.gov", "bea.gov",
        "federalreserve.gov", "treasury.gov", "eia.gov",
    )
    ALLOWLIST_FILINGS = ("sec.gov", "efts.sec.gov", "data.sec.gov")
    ALLOWLIST_EARNINGS = ("sec.gov", "prnewswire.com", "businesswire.com")

    def __init__(self, *, mode: str = "mock", max_bytes: int = 150_000, **kwargs):
        self.mode = mode
        self.max_bytes = max_bytes
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
        official_urls: list[str] | None = None,
        desk: str = "",
        resolver_path: str = "",
    ) -> list[dict[str, Any]]:
        allow = tuple((allowlist or []))
        if self.mode == "mock":
            return self._mock_items(
                kind=kind, ticker=ticker, query=query, max_items=max_items,
                allowlist=allow, desk=desk, resolver_path=resolver_path,
            )

        items: list[dict[str, Any]] = []
        fetched_urls = set()

        # 1) official URLs first (structured providers)
        for u in official_urls or []:
            if len(items) >= max_items:
                break
            if not self._url_allowed(u, allow):
                continue
            ev = self._fetch_url_as_evidence(
                u, kind=kind, ticker=ticker, desk=desk, resolver_path=resolver_path, allowlist=allow
            )
            if ev:
                items.append(ev)
                fetched_urls.add(ev["url"])

        # 2) NewsAPI URLs as supplementary source
        if len(items) < max_items:
            urls = self._search_newsapi_urls(query=query or ticker, recency_days=recency_days, allowlist=allow, limit=max_items * 3)
            for u, title, published_at, source in urls:
                if len(items) >= max_items:
                    break
                if u in fetched_urls:
                    continue
                ev = self._fetch_url_as_evidence(
                    u, kind=kind, ticker=ticker, desk=desk, resolver_path=resolver_path, allowlist=allow,
                    fallback_title=title, fallback_published_at=published_at, fallback_source=source,
                )
                if ev:
                    items.append(ev)
                    fetched_urls.add(ev["url"])

        return items

    def _mock_items(
        self,
        *,
        kind: str,
        ticker: str,
        query: str,
        max_items: int,
        allowlist: tuple[str, ...],
        desk: str,
        resolver_path: str,
    ) -> list[dict[str, Any]]:
        seed_raw = f"{kind}|{ticker}|{query}|{max_items}"
        seed = int(hashlib.sha256(seed_raw.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)
        host = allowlist[0] if allowlist else "example.com"
        now = datetime.now(timezone.utc)
        items: list[dict[str, Any]] = []
        for i in range(max_items):
            published = (now - timedelta(days=min(7, i))).isoformat()
            title = f"{ticker or 'MARKET'} {kind} mock evidence {i + 1}"
            url = f"https://{host}/mock/{kind}/{ticker.lower() if ticker else 'global'}/{i+1}"
            snippet = f"deterministic mock snippet {rng.randint(1000, 9999)}"
            h = self._evidence_hash(url=url, title=title, published_at=published, source=host)
            items.append({
                "url": url,
                "title": title,
                "published_at": published,
                "snippet": snippet,
                "source": host,
                "retrieved_at": now.isoformat(),
                "hash": h,
                "kind": kind,
                "desk": desk,
                "ticker": ticker,
                "trust_tier": self._trust_tier_from_host(host),
                "resolver_path": resolver_path or "mock",
            })
        return items

    def _search_newsapi_urls(
        self,
        *,
        query: str,
        recency_days: int,
        allowlist: tuple[str, ...],
        limit: int,
    ) -> list[tuple[str, str, str, str]]:
        key = self._settings.newsapi_api_key
        if not key or not query:
            return []
        capped_days = max(1, min(int(recency_days or 30), self.NEWSAPI_MAX_RECENCY_DAYS))
        from_date = (datetime.now(timezone.utc) - timedelta(days=capped_days)).strftime("%Y-%m-%d")
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": from_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(100, max(10, limit)),
            "apiKey": key,
        }
        try:
            data = self.get_json(url, params=params)
        except ProviderError:
            return []

        out: list[tuple[str, str, str, str]] = []
        for art in data.get("articles", []):
            u = str(art.get("url", "")).strip()
            if not self._url_allowed(u, allowlist):
                continue
            out.append((
                u,
                str(art.get("title", "")).strip(),
                str(art.get("publishedAt", "")).strip(),
                str((art.get("source") or {}).get("name", "")),
            ))
            if len(out) >= limit:
                break
        return out

    def _fetch_url_as_evidence(
        self,
        url: str,
        *,
        kind: str,
        ticker: str,
        desk: str,
        resolver_path: str,
        allowlist: tuple[str, ...],
        fallback_title: str = "",
        fallback_published_at: str = "",
        fallback_source: str = "",
    ) -> dict[str, Any] | None:
        try:
            result = self._safe_fetch_with_redirects(url, allowlist=allowlist, max_bytes=self.max_bytes)
        except ProviderError:
            return None
        if result is None:
            return None

        final_url = result["final_url"]
        host = (urlparse(final_url).hostname or "").lower()
        title = result.get("title") or fallback_title or final_url
        snippet = result.get("snippet", "")
        published_at = result.get("published_at") or fallback_published_at or datetime.now(timezone.utc).isoformat()
        source = fallback_source or host
        return {
            "url": final_url,
            "title": title,
            "published_at": published_at,
            "snippet": snippet,
            "source": source,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "hash": self._evidence_hash(url=final_url, title=title, published_at=published_at, source=source),
            "kind": kind,
            "desk": desk,
            "ticker": ticker,
            "trust_tier": self._trust_tier_from_host(host),
            "resolver_path": resolver_path,
        }

    def _safe_fetch_with_redirects(
        self,
        url: str,
        *,
        allowlist: tuple[str, ...],
        max_redirects: int = 5,
        max_bytes: int = 150_000,
    ) -> dict[str, Any] | None:
        current = url
        for _ in range(max_redirects + 1):
            self._validate_fetch_target(current, allowlist=allowlist)
            try:
                resp = self._session.get(current, timeout=self._timeout, allow_redirects=False, stream=True)
                record_api_request(self.PROVIDER_NAME, success=True, category="data")
            except Exception as exc:  # requests exceptions
                record_api_request(self.PROVIDER_NAME, success=False, category="data")
                raise ProviderError(f"[web_research] fetch failed: {exc}") from exc

            code = int(resp.status_code)
            if 300 <= code < 400:
                nxt = resp.headers.get("Location", "")
                if not nxt:
                    raise ProviderError("[web_research] redirect without location")
                current = urljoin(current, nxt)
                continue

            if code >= 400:
                raise ProviderError(f"[web_research] HTTP status {code}")

            final_url = resp.url
            self._validate_fetch_target(final_url, allowlist=allowlist)

            chunks = []
            size = 0
            for chunk in resp.iter_content(chunk_size=4096):
                if not chunk:
                    continue
                size += len(chunk)
                if size > max_bytes:
                    break
                chunks.append(chunk)
            raw = b"".join(chunks)
            text = raw.decode("utf-8", errors="ignore")
            return {
                "final_url": final_url,
                "title": self._extract_title(text),
                "snippet": self._extract_snippet(text),
                "published_at": self._extract_published_at(text),
            }
        raise ProviderError("[web_research] too many redirects")

    def _validate_fetch_target(self, url: str, *, allowlist: tuple[str, ...]) -> None:
        parsed = urlparse(url)
        if parsed.scheme.lower() != "https":
            raise ProviderError("[web_research] only https allowed")
        host = (parsed.hostname or "").lower()
        if not host:
            raise ProviderError("[web_research] invalid host")
        if not self._host_in_allowlist(host, allowlist):
            raise ProviderError("[web_research] host outside allowlist")
        if not self._host_public(host):
            raise ProviderError("[web_research] private/reserved host blocked")

    def _url_allowed(self, url: str, allowlist: tuple[str, ...]) -> bool:
        try:
            parsed = urlparse(url)
            host = (parsed.hostname or "").lower()
            return parsed.scheme.lower() == "https" and self._host_in_allowlist(host, allowlist)
        except ValueError:
            return False

    @staticmethod
    def _host_in_allowlist(host: str, allowlist: tuple[str, ...]) -> bool:
        return any(host == d or host.endswith("." + d) for d in allowlist)

    @staticmethod
    def _host_public(host: str) -> bool:
        try:
            infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
        except socket.gaierror:
            return False
        if not infos:
            return False
        for info in infos:
            ip_str = info[4][0]
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                return False
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
                return False
        return True

    @staticmethod
    def _extract_title(html: str) -> str:
        m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return ""
        return re.sub(r"\s+", " ", m.group(1)).strip()[:300]

    @staticmethod
    def _extract_snippet(html: str) -> str:
        text = re.sub(r"<script.*?>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:600]

    @staticmethod
    def _extract_published_at(html: str) -> str:
        meta_patterns = [
            r'property="article:published_time"\s+content="([^"]+)"',
            r'name="pubdate"\s+content="([^"]+)"',
            r'name="date"\s+content="([^"]+)"',
        ]
        for pat in meta_patterns:
            m = re.search(pat, html, flags=re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return ""

    @staticmethod
    def _trust_tier_from_host(host: str) -> float:
        host = (host or "").lower()
        if any(host == d or host.endswith("." + d) for d in (
            "sec.gov", "federalreserve.gov", "treasury.gov", "bea.gov", "bls.gov",
            "fred.stlouisfed.org", "nyfed.org", "eia.gov",
        )):
            return 1.0
        if any(host == d or host.endswith("." + d) for d in ("prnewswire.com", "businesswire.com", "globenewswire.com")):
            return 0.8
        if any(host == d or host.endswith("." + d) for d in ("reuters.com", "bloomberg.com", "wsj.com", "cnbc.com")):
            return 0.6
        return 0.4

    @staticmethod
    def _evidence_hash(*, url: str, title: str, published_at: str, source: str) -> str:
        raw = f"{url}|{title}|{published_at}|{source}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
