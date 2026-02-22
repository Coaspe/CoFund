"""
data_providers/base.py — BaseProvider with session, caching, rate limiting
==========================================================================
"""

from __future__ import annotations

import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter

from config.settings import get_settings
from data_providers.cache import DiskCache
from data_providers.rate_limiter import RateLimiter


class ProviderError(Exception):
    """Raised when a provider cannot fulfill a request."""


class BaseProvider:
    """Base class for all data providers. Manages HTTP session, caching, rate limiting."""

    PROVIDER_NAME: str = "base"

    def __init__(self, *, cache: DiskCache | None = None, rate_limiter: RateLimiter | None = None):
        cfg = get_settings()
        self._cache = cache or DiskCache(cache_dir=cfg.cache_dir, default_ttl=cfg.cache_ttl_sec)
        self._limiter = rate_limiter or RateLimiter(qps=cfg.rate_limit_qps)
        self._timeout = cfg.http_timeout_sec
        self._session = self._build_session()

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

        if not skip_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        self._limiter.wait()

        try:
            resp = self._session.get(url, params=params, headers=headers, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            raise ProviderError(f"[{self.PROVIDER_NAME}] HTTP error: {e}") from e

        self._cache.set(cache_key, data, ttl_sec=cache_ttl)
        return data

    @staticmethod
    def quality_score(*, is_live: bool, freshness_hours: float = 0, completeness: float = 1.0) -> float:
        if not is_live:
            return 0.3
        freshness_penalty = min(freshness_hours / 168.0, 0.3)  # 1 week = max penalty
        return round(max(0.1, min(1.0, 0.9 - freshness_penalty) * completeness), 2)
