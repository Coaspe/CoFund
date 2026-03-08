"""
data_providers/sec_edgar_provider.py — SEC EDGAR text flag provider
====================================================================
Resolves ticker→CIK, fetches recent 10-K/10-Q, keyword-scans for
going_concern, restatement, material_weakness, regulatory_action flags.
No API key needed. Requires SEC_USER_AGENT header.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from api_usage_stats import record_api_request
from schemas.common import make_evidence
from config.settings import get_settings
from data_providers.base import BaseProvider, ProviderError

_SEC_BASE = "https://efts.sec.gov/LATEST"
_SEC_SUBMISSIONS = "https://data.sec.gov/submissions"

_FLAG_KEYWORDS = {
    "going_concern": [
        "going concern", "substantial doubt", "ability to continue as a going concern"
    ],
    "restatement": [
        "restatement", "restated", "restate previously reported", "correction of an error"
    ],
    "material_weakness": [
        "material weakness", "significant deficiency in internal control"
    ],
    "regulatory_action": [
        "sec enforcement", "regulatory action", "consent decree",
        "investigation by the securities", "subpoena"
    ],
}

_EIGHT_K_ITEM_RE = re.compile(r"\bitem\s+([1-9]\.\d{2})\b", re.IGNORECASE)
_EIGHT_K_EXHIBIT_RE = re.compile(r"\b(?:exhibit|ex)\s+(99(?:\.\d+)?)\b", re.IGNORECASE)
_EIGHT_K_LEGAL_RE = re.compile(
    r"\b(antitrust|lawsuit|litigation|court|doj|ftc|subpoena|investigation|settlement|consent decree|regulator(?:y)?)\b",
    re.IGNORECASE,
)
_EIGHT_K_CONTRACT_RE = re.compile(
    r"\b(agreement|contract|renewal|customer|supplier|strategic partnership|commercial agreement|master services agreement)\b",
    re.IGNORECASE,
)
_EIGHT_K_PRICING_RE = re.compile(
    r"\b(pricing|price increase|price cut|discount|repricing|tariff|rate change)\b",
    re.IGNORECASE,
)


def _strip_html(raw: str) -> str:
    return re.sub(r"<[^>]+>", " ", raw or "")


class SECEdgarProvider(BaseProvider):
    PROVIDER_NAME = "sec_edgar"

    def __init__(self, **kwargs):
        self._user_agent = get_settings().sec_user_agent
        super().__init__(**kwargs)

    @property
    def has_agent(self) -> bool:
        return bool(self._user_agent)

    def _headers(self) -> dict:
        return {"User-Agent": self._user_agent, "Accept-Encoding": "gzip, deflate"}

    def _search_filings(self, ticker: str, forms: str, startdt: str = "2023-01-01", size: int = 5, query: str | None = None) -> list[dict]:
        url = f"{_SEC_BASE}/search-index"
        params = {
            "q": query or f'"{ticker}"',
            "forms": forms,
            "dateRange": "custom",
            "startdt": startdt,
            "from": 0,
            "size": size,
        }
        data = self.get_json(url, params, headers=self._headers())
        return data.get("hits", {}).get("hits", []) or []

    @staticmethod
    def _hit_to_evidence_item(hit: dict, *, kind: str, ticker: str, resolver_path: str) -> dict:
        src = hit.get("_source", {})
        filing_url = src.get("file_url", "")
        if filing_url and filing_url.startswith("/"):
            filing_url = f"https://efts.sec.gov{filing_url}"
        title = src.get("display_names", []) or []
        title_text = " | ".join(str(t) for t in title[:2]) if title else f"{ticker} {kind}"
        published_at = src.get("file_date", "")
        source = "sec.gov"
        raw = f"{filing_url}|{title_text}|{published_at}|{source}"
        return {
            "url": filing_url or "https://sec.gov",
            "title": title_text[:300],
            "published_at": published_at,
            "snippet": str(src.get("description", ""))[:600],
            "source": source,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "hash": hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16],
            "kind": kind,
            "desk": "fundamental",
            "ticker": ticker,
            "trust_tier": 1.0,
            "resolver_path": resolver_path,
        }

    def _resolve_cik(self, ticker: str) -> str | None:
        url = f"{_SEC_BASE}/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2020-01-01&forms=10-K"
        try:
            data = self.get_json(url, headers=self._headers())
            hits = data.get("hits", {}).get("hits", [])
            if hits:
                cik = hits[0].get("_source", {}).get("entity_id")
                return str(cik).zfill(10) if cik else None
        except ProviderError:
            pass
        # Fallback: direct ticker search
        try:
            url2 = "https://www.sec.gov/cgi-bin/browse-edgar"
            params = {"company": ticker, "CIK": ticker, "type": "10-K", "dateb": "",
                      "owner": "include", "count": 1, "search_text": "", "action": "getcompany", "output": "atom"}
            data2 = self.get_json(url2, params, headers=self._headers())
            return None  # XML parsing not trivial; return None
        except ProviderError:
            return None

    def get_sec_flags(self, ticker: str, as_of: str = "") -> dict:
        as_of = as_of or datetime.now(timezone.utc).isoformat()

        if not self.has_agent:
            return {"data": None, "evidence": [], "data_ok": False,
                    "limitations": ["SEC_USER_AGENT not set; SEC flags unavailable"], "as_of": as_of}

        evidence: list = []
        limitations: list = []
        flags: Dict[str, bool] = {k: False for k in _FLAG_KEYWORDS}

        try:
            # Try EFTS full-text search for the ticker's recent 10-K
            url = f"{_SEC_BASE}/search-index"
            params = {"q": f'"{ticker}"', "forms": "10-K,10-Q", "dateRange": "custom",
                      "startdt": "2023-01-01", "from": 0, "size": 3}
            data = self.get_json(url, params, headers=self._headers())

            hits = data.get("hits", {}).get("hits", [])
            if not hits:
                limitations.append(f"No recent 10-K/10-Q found for {ticker}")
            else:
                for hit in hits[:2]:
                    source = hit.get("_source", {})
                    file_date = source.get("file_date", "")
                    # Full-text search for keywords in filing text
                    filing_url = source.get("file_url")
                    if filing_url:
                        try:
                            resp = self._session.get(
                                f"https://efts.sec.gov{filing_url}" if filing_url.startswith("/") else filing_url,
                                headers=self._headers(), timeout=self._timeout,
                            )
                            record_api_request(self.PROVIDER_NAME, success=True, category="data")
                            text = resp.text.lower()[:500_000]  # limit for performance
                            for flag_name, keywords in _FLAG_KEYWORDS.items():
                                for kw in keywords:
                                    if kw in text:
                                        flags[flag_name] = True
                                        break
                        except Exception:
                            record_api_request(self.PROVIDER_NAME, success=False, category="data")
                            limitations.append("Could not fetch filing text for keyword scan")

                    for flag_name, found in flags.items():
                        evidence.append(make_evidence(
                            metric=flag_name, value=found,
                            source_name=f"SEC:{file_date}", source_type="api", quality=0.70, as_of=file_date or as_of,
                        ))

        except ProviderError as e:
            limitations.append(f"SEC EDGAR error: {e}")
            return {"data": None, "evidence": [], "data_ok": False,
                    "limitations": limitations, "as_of": as_of}

        sec_data = {
            "has_going_concern_language": flags["going_concern"],
            "has_restatement": flags["restatement"],
            "has_material_weakness": flags["material_weakness"],
            "has_regulatory_action": flags["regulatory_action"],
            "filing_date": as_of,
        }

        return {
            "data": sec_data,
            "evidence": evidence,
            "data_ok": True,
            "limitations": limitations,
            "as_of": as_of,
        }

    def get_ownership_identity(self, ticker: str, as_of: str = "") -> dict:
        """
        Ownership identity evidence (Form 4 / 13F / 13D / 13G).
        Resolver priority uses this provider first.
        """
        as_of = as_of or datetime.now(timezone.utc).isoformat()
        if not self.has_agent:
            return {"items": [], "data_ok": False, "limitations": ["SEC_USER_AGENT not set"], "as_of": as_of}
        forms = "4,13F-HR,SC 13D,SC 13G"
        limitations: list[str] = []
        try:
            cik = self._resolve_cik(ticker)
            hits = []
            if cik:
                hits = self._search_filings(ticker, forms=forms, startdt="2023-01-01", size=12, query=f'"{cik}"')
            if not hits:
                hits = self._search_filings(ticker, forms=forms, startdt="2023-01-01", size=12)
        except ProviderError as exc:
            limitations.append(f"SEC ownership search error: {exc}")
            return {"items": [], "data_ok": False, "limitations": limitations, "as_of": as_of}
        items = [
            self._hit_to_evidence_item(h, kind="ownership_identity", ticker=ticker, resolver_path="sec_forms")
            for h in hits
        ]
        return {"items": items, "data_ok": bool(items), "limitations": limitations, "as_of": as_of}

    def get_8k_exhibits(self, ticker: str, as_of: str = "") -> dict:
        """
        8-K filings as primary source for press_release_or_ir resolution.
        """
        as_of = as_of or datetime.now(timezone.utc).isoformat()
        if not self.has_agent:
            return {"items": [], "data_ok": False, "limitations": ["SEC_USER_AGENT not set"], "as_of": as_of}
        limitations: list[str] = []
        try:
            hits = self._search_filings(ticker, forms="8-K", startdt="2023-01-01", size=6)
        except ProviderError as exc:
            limitations.append(f"SEC 8-K search error: {exc}")
            return {"items": [], "data_ok": False, "limitations": limitations, "as_of": as_of}
        items = []
        for hit in hits:
            item = self._hit_to_evidence_item(hit, kind="press_release_or_ir", ticker=ticker, resolver_path="sec_8k")
            filing_text = self._fetch_filing_text(item.get("url", ""))
            item_numbers = sorted(set(_EIGHT_K_ITEM_RE.findall(filing_text)))
            exhibit_codes = sorted(set(_EIGHT_K_EXHIBIT_RE.findall(filing_text)))
            catalyst_type = self._classify_8k_catalyst(item, filing_text, item_numbers, exhibit_codes)
            if filing_text:
                item["snippet"] = " ".join(_strip_html(filing_text).split())[:600]
            item["filing_items"] = item_numbers
            item["exhibit_codes"] = exhibit_codes
            item["catalyst_type"] = catalyst_type
            item["source_classification"] = "confirmed"
            item["event_origin"] = "sec_8k"
            items.append(item)
        return {"items": items, "data_ok": bool(items), "limitations": limitations, "as_of": as_of}

    def _fetch_filing_text(self, filing_url: str) -> str:
        if not filing_url:
            return ""
        try:
            resp = self._session.get(filing_url, headers=self._headers(), timeout=self._timeout)
            record_api_request(self.PROVIDER_NAME, success=True, category="data")
            return resp.text[:200_000]
        except Exception:
            record_api_request(self.PROVIDER_NAME, success=False, category="data")
            return ""

    @staticmethod
    def _classify_8k_catalyst(item: dict, filing_text: str, item_numbers: list[str], exhibit_codes: list[str]) -> str | None:
        text = " ".join(
            [
                str(item.get("title", "")),
                str(item.get("snippet", "")),
                _strip_html(filing_text),
            ]
        )
        if _EIGHT_K_PRICING_RE.search(text):
            return "pricing_reset"
        if _EIGHT_K_LEGAL_RE.search(text):
            return "legal_reg"
        if any(x in {"1.01", "1.02"} for x in item_numbers) and _EIGHT_K_CONTRACT_RE.search(text):
            return "contract_renewal"
        if "investor day" in text.lower() or "capital markets day" in text.lower() or "analyst day" in text.lower():
            return "investor_day"
        if any(k in text.lower() for k in ("launch", "unveil", "release", "new product", "approval", "shipment")):
            return "product_cycle"
        if "99.1" in exhibit_codes and _EIGHT_K_CONTRACT_RE.search(text):
            return "contract_renewal"
        return None


if __name__ == "__main__":
    import json
    provider = SECEdgarProvider()
    ticker = "AAPL"
    if provider.has_agent:
        print(f"📄 SEC EDGAR — scanning {ticker} filings...")
    else:
        print("⚠️  No SEC_USER_AGENT set")
    result = provider.get_sec_flags(ticker)
    print(json.dumps(result, indent=2, default=str))
