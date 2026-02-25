"""
tests/test_research_resolver_priority.py
========================================
Kind resolver priority tests.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import investment_team


class _SecStub:
    def __init__(self, ownership_ok=False, sec8k_ok=False):
        self.ownership_ok = ownership_ok
        self.sec8k_ok = sec8k_ok
        self.calls = []

    def get_ownership_identity(self, ticker, as_of=""):
        self.calls.append(("ownership", ticker))
        if self.ownership_ok:
            return {"data_ok": True, "items": [{"hash": "sec-own", "url": "https://sec.gov/own"}]}
        return {"data_ok": False, "items": []}

    def get_8k_exhibits(self, ticker, as_of=""):
        self.calls.append(("8k", ticker))
        if self.sec8k_ok:
            return {"data_ok": True, "items": [{"hash": "sec-8k", "url": "https://sec.gov/8k"}]}
        return {"data_ok": False, "items": []}


class _WebStub:
    def __init__(self):
        self.calls = []
        self._next = []

    def queue(self, *returns):
        self._next = list(returns)

    def collect_evidence(self, **kwargs):
        self.calls.append(kwargs.get("resolver_path"))
        if self._next:
            return self._next.pop(0)
        return []


def test_ownership_identity_sec_first_then_web_fallback():
    req = {"kind": "ownership_identity", "desk": "fundamental", "ticker": "AAPL", "query": "q", "max_items": 3}
    sec = _SecStub(ownership_ok=True)
    web = _WebStub()
    items, path = investment_team._resolve_request_with_priority(req, sec=sec, web=web, as_of="2026-01-01T00:00:00+00:00")
    assert path == "sec_forms"
    assert sec.calls and sec.calls[0][0] == "ownership"
    assert web.calls == []

    sec2 = _SecStub(ownership_ok=False)
    web2 = _WebStub()
    web2.queue([{"hash": "web-own"}])
    items2, path2 = investment_team._resolve_request_with_priority(req, sec=sec2, web=web2, as_of="2026-01-01T00:00:00+00:00")
    assert path2 == "web_fallback_ownership"
    assert web2.calls == ["web_fallback_ownership"]


def test_press_release_or_ir_priority_sec8k_ir_newsapi():
    req = {"kind": "press_release_or_ir", "desk": "fundamental", "ticker": "AAPL", "query": "q", "max_items": 3}

    sec = _SecStub(sec8k_ok=True)
    web = _WebStub()
    items, path = investment_team._resolve_request_with_priority(req, sec=sec, web=web, as_of="2026-01-01T00:00:00+00:00")
    assert path == "sec_8k"
    assert web.calls == []

    sec2 = _SecStub(sec8k_ok=False)
    web2 = _WebStub()
    web2.queue([], [{"hash": "news"}])  # ir_domain empty -> newsapi success
    items2, path2 = investment_team._resolve_request_with_priority(req, sec=sec2, web=web2, as_of="2026-01-01T00:00:00+00:00")
    assert path2 == "newsapi"
    assert web2.calls == ["ir_domain", "newsapi"]


def test_macro_headline_context_official_then_newsapi():
    req = {"kind": "macro_headline_context", "desk": "macro", "ticker": "AAPL", "query": "macro", "max_items": 3}
    sec = _SecStub()
    web = _WebStub()
    web.queue([], [{"hash": "news"}])
    items, path = investment_team._resolve_request_with_priority(req, sec=sec, web=web, as_of="2026-01-01T00:00:00+00:00")
    assert path == "newsapi_supplement"
    assert web.calls == ["official_release", "newsapi_supplement"]
