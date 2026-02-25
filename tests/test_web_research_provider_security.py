"""
tests/test_web_research_provider_security.py
============================================
Security hardening tests for direct fetch path.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_providers.base import ProviderError
from data_providers.web_research_provider import WebResearchProvider


def test_allowlist_blocks_non_allowed_domain(monkeypatch):
    p = WebResearchProvider(mode="mock")
    monkeypatch.setattr(p, "_host_public", lambda host: True)
    with pytest.raises(ProviderError):
        p._validate_fetch_target("https://evil.com/a", allowlist=("sec.gov",))


def test_http_scheme_blocked(monkeypatch):
    p = WebResearchProvider(mode="mock")
    monkeypatch.setattr(p, "_host_public", lambda host: True)
    with pytest.raises(ProviderError):
        p._validate_fetch_target("http://sec.gov/a", allowlist=("sec.gov",))


def test_private_ip_host_blocked(monkeypatch):
    p = WebResearchProvider(mode="mock")
    monkeypatch.setattr(p, "_host_public", lambda host: False)
    with pytest.raises(ProviderError):
        p._validate_fetch_target("https://sec.gov/a", allowlist=("sec.gov",))


def test_redirect_final_domain_blocked(monkeypatch):
    class _Resp:
        def __init__(self, status_code, url, headers=None, body=b""):
            self.status_code = status_code
            self.url = url
            self.headers = headers or {}
            self._body = body

        def iter_content(self, chunk_size=4096):
            yield self._body

    calls = {"n": 0}

    def _fake_get(url, timeout=10, allow_redirects=False, stream=True):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(302, "https://sec.gov/a", headers={"Location": "https://evil.com/x"})
        return _Resp(200, "https://evil.com/x", body=b"<html><title>x</title>bad</html>")

    p = WebResearchProvider(mode="mock")
    monkeypatch.setattr(p, "_host_public", lambda host: True)
    monkeypatch.setattr(p._session, "get", _fake_get)
    with pytest.raises(ProviderError):
        p._safe_fetch_with_redirects("https://sec.gov/a", allowlist=("sec.gov",), max_bytes=1024)


def test_size_limit_enforced(monkeypatch):
    class _Resp:
        def __init__(self):
            self.status_code = 200
            self.url = "https://sec.gov/a"
            self.headers = {}

        def iter_content(self, chunk_size=4096):
            yield b"a" * 1000
            yield b"b" * 1000

    p = WebResearchProvider(mode="mock")
    monkeypatch.setattr(p, "_host_public", lambda host: True)
    monkeypatch.setattr(p._session, "get", lambda *args, **kwargs: _Resp())
    out = p._safe_fetch_with_redirects("https://sec.gov/a", allowlist=("sec.gov",), max_bytes=1000)
    assert out is not None


def test_newsapi_recency_is_capped_to_30_days(monkeypatch):
    class _Settings:
        newsapi_api_key = "x"

    captured: dict = {}

    def _fake_get_json(url, params=None, headers=None):
        captured["url"] = url
        captured["params"] = params or {}
        return {"articles": []}

    p = WebResearchProvider(mode="live")
    monkeypatch.setattr(p, "_settings", _Settings())
    monkeypatch.setattr(p, "get_json", _fake_get_json)

    p._search_newsapi_urls(query="NVDA valuation context", recency_days=365, allowlist=("sec.gov",), limit=10)

    expected_from = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    assert captured["url"] == "https://newsapi.org/v2/everything"
    assert captured["params"]["from"] == expected_from
