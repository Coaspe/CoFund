"""
data_providers/macro_event_calendar_provider.py — official macro event calendar
===============================================================================
Fetches first-class macro event dates from official sources where possible.

Current live sources:
- Federal Reserve FOMC calendar
- BEA release schedule (GDP advance estimate / Personal Income and Outlays)

Notes:
- BLS pages are bot-restricted from this environment, so CPI/NFP are not added
  here unless an accessible structured source is introduced later.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

from schemas.common import make_evidence

try:
    from data_providers.base import BaseProvider, ProviderError
except ImportError:  # pragma: no cover
    BaseProvider = object  # type: ignore
    ProviderError = Exception


_FED_FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
_BEA_SCHEDULE_URL = "https://www.bea.gov/news/schedule"
_ET = ZoneInfo("America/New_York")
_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()


def _parse_release_datetime(year: int, date_label: str, time_label: str = "", *, end_of_range: bool = False) -> str:
    clean_date = str(date_label or "").strip()
    if not clean_date:
        return ""
    parts = clean_date.replace("*", "").split()
    if len(parts) < 2:
        return ""
    month = _MONTHS.get(parts[0].strip().lower())
    if month is None:
        return ""
    day_token = parts[1].strip()
    if "-" in day_token:
        left, right = day_token.split("-", 1)
        day = int(right if end_of_range else left)
    else:
        day = int(re.sub(r"[^\d]", "", day_token))
    hour = 8
    minute = 30
    if time_label:
        match = re.search(r"(\d{1,2}):(\d{2})\s*(AM|PM)", time_label, re.I)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            ampm = match.group(3).upper()
            if ampm == "PM" and hour != 12:
                hour += 12
            elif ampm == "AM" and hour == 12:
                hour = 0
    dt = datetime(year, month, day, hour, minute, tzinfo=_ET)
    return dt.astimezone(timezone.utc).isoformat()


class MacroEventCalendarProvider(BaseProvider if isinstance(BaseProvider, type) else object):
    PROVIDER_NAME = "macro_event_calendar"

    def get_calendar(self, as_of: str = "") -> dict:
        as_of = as_of or datetime.now(timezone.utc).isoformat()
        try:
            year = datetime.fromisoformat(as_of.replace("Z", "+00:00")).astimezone(timezone.utc).year
        except ValueError:
            year = datetime.now(timezone.utc).year

        limitations: list[str] = []
        items: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []

        try:
            fomc_items = self._fetch_fomc_events(year, as_of)
            items.extend(fomc_items)
            evidence.extend(self._event_evidence(fomc_items))
        except ProviderError as exc:
            limitations.append(f"FOMC schedule unavailable: {exc}")

        try:
            bea_items = self._fetch_bea_events(year, as_of)
            items.extend(bea_items)
            evidence.extend(self._event_evidence(bea_items))
        except ProviderError as exc:
            limitations.append(f"BEA schedule unavailable: {exc}")

        if not items:
            mock = self._mock_calendar(as_of)
            return mock

        return {
            "data": {"events": items},
            "items": items,
            "evidence": evidence,
            "data_ok": True,
            "limitations": limitations,
            "as_of": as_of,
        }

    def _fetch_text(self, url: str) -> str:
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            resp = self._session.get(url, headers=headers, timeout=self._timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:  # pragma: no cover - network variability
            raise ProviderError(f"HTTP error: {exc}") from exc

    def _fetch_fomc_events(self, year: int, as_of: str) -> list[dict]:
        text = self._fetch_text(_FED_FOMC_URL)
        section_match = re.search(
            rf"{year}\s+FOMC Meetings(.*?)(?:{year + 1}\s+FOMC Meetings|</section>|<h\d)",
            text,
            re.I | re.S,
        )
        if not section_match:
            raise ProviderError(f"could not locate {year} FOMC section")
        section = section_match.group(1)
        rows = re.findall(
            r'fomc-meeting__month[^>]*><strong>([^<]+)</strong>.*?fomc-meeting__date[^>]*>([^<]+)</div>',
            section,
            re.I | re.S,
        )
        items: list[dict] = []
        for month_label, day_label in rows:
            dt = _parse_release_datetime(year, f"{month_label} {day_label}", "2:00 PM", end_of_range=True)
            if not dt:
                continue
            items.append(
                {
                    "type": "fomc",
                    "subtype": "policy_decision",
                    "title": f"FOMC Meeting ({month_label} {day_label.strip()}, {year})",
                    "date": dt,
                    "source": "federalreserve.gov",
                    "source_url": _FED_FOMC_URL,
                    "source_classification": "confirmed",
                    "event_origin": "official_macro_calendar",
                    "notes": "Federal Reserve FOMC calendar",
                    "as_of": as_of,
                }
            )
        return items

    def _fetch_bea_events(self, year: int, as_of: str) -> list[dict]:
        text = self._fetch_text(_BEA_SCHEDULE_URL)
        rows = re.findall(
            r'<tr[^>]*class="scheduled-releases-type-press"[^>]*>.*?<div class="release-date">([^<]+)</div>\s*<small[^>]*>([^<]+)</small>.*?<td class="release-title[^"]*"[^>]*>(.*?)</td>',
            text,
            re.I | re.S,
        )
        items: list[dict] = []
        for date_label, time_label, raw_title in rows:
            title = _strip_html(raw_title)
            if str(year) not in title:
                continue
            if title.startswith("GDP (Advance Estimate)"):
                event_type = "gdp"
                subtype = "advance_estimate"
            elif title.startswith("Personal Income and Outlays"):
                event_type = "pce"
                subtype = "personal_income_and_outlays"
            else:
                continue
            dt = _parse_release_datetime(year, date_label, time_label)
            if not dt:
                continue
            items.append(
                {
                    "type": event_type,
                    "subtype": subtype,
                    "title": title,
                    "date": dt,
                    "source": "bea.gov",
                    "source_url": _BEA_SCHEDULE_URL,
                    "source_classification": "confirmed",
                    "event_origin": "official_macro_calendar",
                    "notes": "BEA release schedule",
                    "as_of": as_of,
                }
            )
        return items

    def _event_evidence(self, items: list[dict]) -> list[dict]:
        evidence: list[dict] = []
        for item in items:
            evidence.append(
                make_evidence(
                    metric=f"macro_event:{item.get('type', 'event')}",
                    value=item.get("date", ""),
                    source_name=str(item.get("source", "macro_event_calendar")),
                    source_type="api",
                    quality=0.85,
                    as_of=str(item.get("date", "")).strip() or str(item.get("as_of", "")).strip(),
                    note=str(item.get("title", "")).strip(),
                )
            )
        return evidence

    @staticmethod
    def _mock_calendar(as_of: str) -> dict:
        try:
            base = datetime.fromisoformat(as_of.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            base = datetime.now(timezone.utc)
        items = [
            {
                "type": "fomc",
                "subtype": "policy_decision",
                "title": "FOMC Meeting (March 17-18, 2026)",
                "date": datetime(2026, 3, 18, 18, 0, tzinfo=timezone.utc).isoformat(),
                "source": "mock",
                "source_url": _FED_FOMC_URL,
                "source_classification": "confirmed",
                "event_origin": "mock_macro_calendar",
                "notes": "mock",
                "as_of": as_of,
            },
            {
                "type": "pce",
                "subtype": "personal_income_and_outlays",
                "title": "Personal Income and Outlays, February 2026",
                "date": datetime(2026, 4, 9, 12, 30, tzinfo=timezone.utc).isoformat(),
                "source": "mock",
                "source_url": _BEA_SCHEDULE_URL,
                "source_classification": "confirmed",
                "event_origin": "mock_macro_calendar",
                "notes": "mock",
                "as_of": as_of,
            },
            {
                "type": "gdp",
                "subtype": "advance_estimate",
                "title": "GDP (Advance Estimate), 1st Quarter 2026",
                "date": datetime(2026, 4, 30, 12, 30, tzinfo=timezone.utc).isoformat(),
                "source": "mock",
                "source_url": _BEA_SCHEDULE_URL,
                "source_classification": "confirmed",
                "event_origin": "mock_macro_calendar",
                "notes": "mock",
                "as_of": as_of,
            },
        ]
        return {
            "data": {"events": items},
            "items": items,
            "evidence": [
                make_evidence(
                    metric=f"macro_event:{item['type']}",
                    value=item["date"],
                    source_name="mock",
                    source_type="api",
                    quality=0.3,
                    as_of=item["date"],
                    note=item["title"],
                )
                for item in items
            ],
            "data_ok": False,
            "limitations": ["Using mock macro event calendar; official BLS schedule not accessible from provider environment"],
            "as_of": as_of,
        }
