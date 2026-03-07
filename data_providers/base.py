"""
data_providers/base.py — BaseProvider with session, caching, rate limiting
==========================================================================
"""

from __future__ import annotations

import threading
import time
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import requests
from requests.adapters import HTTPAdapter

from api_usage_stats import record_api_request
from config.settings import get_settings
from data_providers.cache import DiskCache
from data_providers.rate_limiter import RateLimiter


class ProviderError(Exception):
    """Raised when a provider cannot fulfill a request."""


_RATE_LIMIT_TOKENS = (
    "429",
    "rate limit",
    "too many requests",
    "quota",
    "resource exhausted",
    "tokens per minute",
    "tpm",
    "rpm",
)
_RATE_LIMITED_DATA_APIS: set[str] = set()
_RATE_LIMIT_LOCK = threading.Lock()
_ALL_DATA_APIS_RATE_LIMIT_LOGGED = False


def is_rate_limit_error(exc_or_msg: Exception | str) -> bool:
    text = str(exc_or_msg).lower()
    return any(token in text for token in _RATE_LIMIT_TOKENS)


def _configured_data_api_providers() -> set[str]:
    cfg = get_settings()
    providers: set[str] = set()
    if cfg.fred_api_key:
        providers.add("fred")
    if cfg.fmp_api_key:
        providers.add("fmp")
    if cfg.newsapi_api_key:
        providers.add("newsapi")
    if cfg.alphavantage_api_key:
        providers.add("alphavantage")
    if cfg.twelvedata_api_key:
        providers.add("twelvedata")
    if cfg.sec_user_agent:
        providers.add("sec_edgar")
    if cfg.tavily_api_key:
        providers.add("tavily_search")
    if cfg.exa_api_key:
        providers.add("exa_search")
    if cfg.perplexity_api_key:
        providers.add("perplexity_search")
    return providers


def _mark_data_api_rate_limited(provider_name: str) -> None:
    global _ALL_DATA_APIS_RATE_LIMIT_LOGGED
    with _RATE_LIMIT_LOCK:
        _RATE_LIMITED_DATA_APIS.add(provider_name)
        configured = _configured_data_api_providers()
        if configured and configured.issubset(_RATE_LIMITED_DATA_APIS):
            if not _ALL_DATA_APIS_RATE_LIMIT_LOGGED:
                print("   [API Router] ⚠️ 모든 데이터 API가 rate limit 상태입니다.", flush=True)
                _ALL_DATA_APIS_RATE_LIMIT_LOGGED = True


def _mark_data_api_recovered(provider_name: str) -> None:
    global _ALL_DATA_APIS_RATE_LIMIT_LOGGED
    with _RATE_LIMIT_LOCK:
        if provider_name in _RATE_LIMITED_DATA_APIS:
            _RATE_LIMITED_DATA_APIS.discard(provider_name)
        if _ALL_DATA_APIS_RATE_LIMIT_LOGGED:
            configured = _configured_data_api_providers()
            if not configured.issubset(_RATE_LIMITED_DATA_APIS):
                _ALL_DATA_APIS_RATE_LIMIT_LOGGED = False


class BaseProvider:
    """Base class for all data providers. Manages HTTP session, caching, rate limiting."""

    PROVIDER_NAME: str = "base"

    def __init__(self, *, cache: DiskCache | None = None, rate_limiter: RateLimiter | None = None):
        cfg = get_settings()
        self._cache = cache or DiskCache(cache_dir=cfg.cache_dir, default_ttl=cfg.cache_ttl_sec)
        self._limiter = rate_limiter or RateLimiter(qps=cfg.rate_limit_qps)
        self._timeout = cfg.http_timeout_sec
        self._http_log_level = getattr(cfg, "http_log_level", "compact")
        self._session = self._build_session()
        self._http_seq = 0

    def _log_http(self, message: str, event: str) -> None:
        """
        event: cache | start | done | fail
        levels:
          all:     cache/start/done/fail
          compact: start/done/fail
          fail:    fail only
          off:     none
        """
        lvl = self._http_log_level
        if lvl == "off":
            return
        if lvl == "fail" and event != "fail":
            return
        if lvl == "compact" and event == "cache":
            return
        if lvl not in {"all", "compact", "fail", "off"}:
            # safe fallback
            if event == "cache":
                return
            print(message, flush=True)
            return
        if lvl == "all" or lvl == "compact" or (lvl == "fail" and event == "fail"):
            print(message, flush=True)

    @staticmethod
    def _sanitize_params(params: dict | None) -> dict | None:
        if params is None:
            return None
        masked = {}
        for k, v in params.items():
            key_l = str(k).lower()
            if "key" in key_l or "token" in key_l or "secret" in key_l:
                masked[k] = "***"
            else:
                masked[k] = v
        return masked

    @staticmethod
    def _compact_value(v: Any, max_len: int = 40) -> str:
        s = str(v)
        if len(s) <= max_len:
            return s
        return s[: max_len - 3] + "..."

    def _format_request_target(self, url: str, params: dict | None = None) -> str:
        """
        Readable compact target:
          host/path key1=val1 key2=val2 ... (+N)
        """
        try:
            sp = urlsplit(url)
            base = f"{sp.netloc}{sp.path}"
            query_pairs = list(parse_qsl(sp.query, keep_blank_values=True))
            clean_params = self._sanitize_params(params) or {}
            query_pairs.extend((k, str(v)) for k, v in clean_params.items())

            # Deduplicate by key preserving first seen
            seen = set()
            compact = []
            for k, v in query_pairs:
                if k in seen:
                    continue
                seen.add(k)
                key_l = str(k).lower()
                if key_l in {"apikey", "api_key", "token", "secret", "key"}:
                    vv = "***"
                else:
                    vv = self._compact_value(v)
                compact.append((k, vv))

            if not compact:
                return base

            shown = compact[:4]
            parts = [f"{k}={v}" for k, v in shown]
            extra = len(compact) - len(shown)
            if extra > 0:
                parts.append(f"+{extra}")
            return f"{base} {' '.join(parts)}"
        except Exception:
            return url

    def _build_session(self) -> requests.Session:
        s = requests.Session()
        try:
            from urllib3.util.retry import Retry
            retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
            adapter = HTTPAdapter(max_retries=retry)
            s.mount("https://", adapter)
            s.mount("http://", adapter)
        except ImportError:
            pass
        return s

    def get_json(
        self, url: str, params: dict | None = None, headers: dict | None = None,
        *, cache_ttl: int | None = None, skip_cache: bool = False,
    ) -> dict:
        cache_key = DiskCache.make_key(self.PROVIDER_NAME, url, params)
        target = self._format_request_target(url, params)
        self._http_seq += 1
        req_id = self._http_seq

        if not skip_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._log_http(f"[HTTP][CACHE][{self.PROVIDER_NAME}][#{req_id}] {target}", "cache")
                return cached

        self._log_http(f"[HTTP][START][{self.PROVIDER_NAME}][#{req_id}] {target}", "start")
        self._limiter.wait()
        started = time.perf_counter()

        try:
            resp = self._session.get(url, params=params, headers=headers, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            record_api_request(self.PROVIDER_NAME, success=False, category="data")
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            status = getattr(getattr(e, "response", None), "status_code", "ERR")
            is_rl = str(status) == "429" or is_rate_limit_error(e)
            self._log_http(
                f"[HTTP][FAIL ][{self.PROVIDER_NAME}][#{req_id}] {elapsed_ms}ms status={status} {target}",
                "fail",
            )
            if is_rl:
                print(
                    f"   [API Router] {self.PROVIDER_NAME}: rate limit 감지 (status={status})",
                    flush=True,
                )
                _mark_data_api_rate_limited(self.PROVIDER_NAME)
                raise ProviderError(f"[{self.PROVIDER_NAME}] RATE_LIMIT: {e}") from e
            raise ProviderError(f"[{self.PROVIDER_NAME}] HTTP error: {e}") from e

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        status = getattr(resp, "status_code", 200)
        self._log_http(
            f"[HTTP][DONE ][{self.PROVIDER_NAME}][#{req_id}] {elapsed_ms}ms status={status} {target}",
            "done",
        )
        record_api_request(self.PROVIDER_NAME, success=True, category="data")
        _mark_data_api_recovered(self.PROVIDER_NAME)
        self._cache.set(cache_key, data, ttl_sec=cache_ttl)
        return data

    @staticmethod
    def quality_score(*, is_live: bool, freshness_hours: float = 0, completeness: float = 1.0) -> float:
        if not is_live:
            return 0.3
        freshness_penalty = min(freshness_hours / 168.0, 0.3)  # 1 week = max penalty
        return round(max(0.1, min(1.0, 0.9 - freshness_penalty) * completeness), 2)
