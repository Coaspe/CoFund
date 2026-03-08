"""
data_providers/sofr_futures_provider.py — SOFR futures curve provider
=====================================================================
Uses dated CME 3-Month SOFR futures contracts via yfinance.
No API key required. Dates are preserved per contract.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List

from schemas.common import make_evidence

try:
    from data_providers.base import BaseProvider
except ImportError:
    BaseProvider = object  # type: ignore


_QUARTER_MONTHS = (3, 6, 9, 12)
_MONTH_CODES = {
    3: "H",
    6: "M",
    9: "U",
    12: "Z",
}


def _next_quarter_contracts(year: int, month: int, count: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    y = year
    candidates = [m for m in _QUARTER_MONTHS if m >= month]
    if candidates:
        start_month = candidates[0]
    else:
        y += 1
        start_month = _QUARTER_MONTHS[0]
    cur_year, cur_month = y, start_month
    for _ in range(count):
        pairs.append((cur_year, cur_month))
        idx = _QUARTER_MONTHS.index(cur_month)
        if idx == len(_QUARTER_MONTHS) - 1:
            cur_year += 1
            cur_month = _QUARTER_MONTHS[0]
        else:
            cur_month = _QUARTER_MONTHS[idx + 1]
    return pairs


class SofrFuturesProvider(BaseProvider if isinstance(BaseProvider, type) else object):
    PROVIDER_NAME = "sofr_futures"

    def get_curve(self, *, contracts: int = 5, as_of: str = "") -> dict:
        as_of = as_of or datetime.now(timezone.utc).isoformat()
        try:
            curve = self._fetch_live_curve(contracts=contracts)
            evidence = self._build_evidence(curve, as_of=as_of)
            data = self._curve_to_data(curve)
            return {
                "data": data,
                "evidence": evidence,
                "data_ok": len(curve) >= 3,
                "limitations": [] if len(curve) >= 3 else ["Insufficient SOFR futures contracts fetched"],
                "as_of": as_of,
            }
        except Exception as e:
            curve = self._mock_curve(contracts=contracts, as_of=as_of)
            evidence = self._build_evidence(curve, as_of=as_of, source_name="mock", quality=0.3)
            data = self._curve_to_data(curve)
            return {
                "data": data,
                "evidence": evidence,
                "data_ok": False,
                "limitations": [f"sofr futures live error: {e}", "Using mock SOFR futures curve"],
                "as_of": as_of,
            }

    def _fetch_live_curve(self, *, contracts: int) -> list[dict]:
        import yfinance as yf  # type: ignore

        now = datetime.now(timezone.utc)
        curve: list[dict] = []
        for year, month in _next_quarter_contracts(now.year, now.month, contracts):
            symbol = f"SR3{_MONTH_CODES[month]}{str(year % 100).zfill(2)}.CME"
            df = yf.download(symbol, period="10d", interval="1d", progress=False, auto_adjust=False)
            if df is None or df.empty:
                continue
            close = df["Close"]
            if hasattr(close, "columns"):
                close = close.iloc[:, 0]
            close = close.dropna()
            if close.empty:
                continue
            last_ts = close.index[-1]
            ts = getattr(last_ts, "to_pydatetime", lambda: last_ts)()
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                else:
                    ts = ts.astimezone(timezone.utc)
                point_as_of = ts.isoformat()
            else:
                point_as_of = datetime.now(timezone.utc).isoformat()
            price = float(close.iloc[-1])
            curve.append(
                {
                    "contract": symbol,
                    "contract_month": f"{year:04d}-{month:02d}",
                    "price": round(price, 6),
                    "implied_rate": round(100.0 - price, 4),
                    "as_of": point_as_of,
                }
            )
        if not curve:
            raise ValueError("no SOFR futures contracts available")
        return curve

    def _mock_curve(self, *, contracts: int, as_of: str) -> list[dict]:
        base = [4.20, 4.05, 3.90, 3.75, 3.60]
        now = datetime.now(timezone.utc)
        curve: list[dict] = []
        for i, (year, month) in enumerate(_next_quarter_contracts(now.year, now.month, contracts)):
            implied = base[i] if i < len(base) else max(base[-1] - 0.10 * (i - len(base) + 1), 2.0)
            curve.append(
                {
                    "contract": f"SR3{_MONTH_CODES[month]}{str(year % 100).zfill(2)}.CME",
                    "contract_month": f"{year:04d}-{month:02d}",
                    "price": round(100.0 - implied, 6),
                    "implied_rate": round(implied, 4),
                    "as_of": as_of,
                }
            )
        return curve

    @staticmethod
    def _curve_to_data(curve: list[dict]) -> dict:
        out: dict[str, Any] = {
            "sofr_futures_curve": curve,
        }
        if not curve:
            return out
        front = curve[0]
        mid = curve[min(1, len(curve) - 1)]
        far = curve[min(2, len(curve) - 1)]
        out.update(
            {
                "sofr_futures_front_contract": front["contract"],
                "sofr_futures_front_implied_rate": front["implied_rate"],
                "sofr_futures_3m_implied_rate": mid["implied_rate"],
                "sofr_futures_6m_implied_rate": far["implied_rate"],
                "sofr_futures_implied_change_6m_bp": round((far["implied_rate"] - front["implied_rate"]) * 100, 1),
            }
        )
        return out

    @staticmethod
    def _build_evidence(
        curve: list[dict],
        *,
        as_of: str,
        source_name: str = "yfinance",
        quality: float = 0.78,
    ) -> List[dict]:
        evidence: List[dict] = []
        if not curve:
            return evidence
        front = curve[0]
        mid = curve[min(1, len(curve) - 1)]
        far = curve[min(2, len(curve) - 1)]
        evidence.append(
            make_evidence(
                metric="sofr_futures_front_implied_rate",
                value=front["implied_rate"],
                source_name=f"{source_name}:{front['contract']}",
                source_type="api",
                quality=quality,
                as_of=front.get("as_of") or as_of,
                note=f"front contract {front['contract_month']}",
            )
        )
        evidence.append(
            make_evidence(
                metric="sofr_futures_3m_implied_rate",
                value=mid["implied_rate"],
                source_name=f"{source_name}:{mid['contract']}",
                source_type="api",
                quality=quality,
                as_of=mid.get("as_of") or as_of,
                note=f"3m contract {mid['contract_month']}",
            )
        )
        evidence.append(
            make_evidence(
                metric="sofr_futures_6m_implied_rate",
                value=far["implied_rate"],
                source_name=f"{source_name}:{far['contract']}",
                source_type="api",
                quality=quality,
                as_of=far.get("as_of") or as_of,
                note=f"6m contract {far['contract_month']}",
            )
        )
        evidence.append(
            make_evidence(
                metric="sofr_futures_implied_change_6m_bp",
                value=round((far["implied_rate"] - front["implied_rate"]) * 100, 1),
                source_name=f"derived:{front['contract']}->{far['contract']}",
                source_type="model",
                quality=quality,
                as_of=far.get("as_of") or as_of,
                note="negative means easing priced over the next 6 months",
            )
        )
        return evidence
