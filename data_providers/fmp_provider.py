"""
data_providers/fmp_provider.py — Financial Modeling Prep provider
=================================================================
Fetches fundamental financials (ratios, profile, cash flow) from FMP.
All live requests use stable base URL:
  https://financialmodelingprep.com/stable
Mock fallback if FMP_API_KEY absent.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any, Dict

from schemas.common import make_evidence
from config.settings import get_settings
from data_providers.base import BaseProvider, ProviderError

_FMP_STABLE_BASE = "https://financialmodelingprep.com/stable"


class FMPProvider(BaseProvider):
    PROVIDER_NAME = "fmp"

    def __init__(self, **kwargs):
        self._api_key = get_settings().fmp_api_key
        super().__init__(**kwargs)

    @property
    def has_key(self) -> bool:
        return bool(self._api_key)

    def _get(self, endpoint: str, ticker: str, **extra_params) -> dict:
        """
        Stable-only request:
          GET /stable/{endpoint}?symbol={ticker}&apikey=...
        """
        url = f"{_FMP_STABLE_BASE}/{endpoint}"
        params = {"symbol": ticker, "apikey": self._api_key}
        params.update(extra_params)
        return self.get_json(url, params)

    def _get_first_available(
        self,
        endpoints: list[str],
        ticker: str,
        limitations: list[str],
        **extra_params,
    ) -> list[dict]:
        last_error = ""
        for endpoint in endpoints:
            try:
                data = self._get(endpoint, ticker, **extra_params)
            except ProviderError as exc:
                last_error = str(exc)
                continue
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                # Some stable endpoints may return {"symbol": ..., ...}
                return [data]
        if last_error:
            limitations.append(f"FMP endpoint unavailable ({'/'.join(endpoints)}): {last_error}")
        return []

    def get_fundamentals(self, ticker: str, as_of: str = "") -> dict:
        as_of = as_of or datetime.now(timezone.utc).isoformat()

        if not self.has_key:
            return self._mock_fundamentals(ticker, as_of)

        evidence: list = []
        limitations: list = []
        financials: Dict[str, Any] = {}

        try:
            # Profile → sector, avg volume
            profile_data = self._get_first_available(["profile"], ticker, limitations)
            if profile_data:
                p = profile_data[0]
                financials["sector"] = p.get("sector", "Unknown")
                financials["avg_daily_volume_usd"] = p.get("volAvg", p.get("avgVolume"))
                financials["market_cap"] = p.get("mktCap", p.get("marketCap"))

            # Key Metrics TTM
            metrics_data = self._get_first_available(
                ["key-metrics-ttm", "key-metrics"],
                ticker,
                limitations,
            )
            if metrics_data:
                m = metrics_data[0]
                financials["pe_ratio"] = m.get("peRatioTTM")
                financials["roe"] = _pct(m.get("roeTTM"))
                financials["debt_to_equity"] = m.get("debtToEquityTTM")

            # Ratios TTM
            ratios_data = self._get_first_available(
                ["ratios-ttm", "ratios"],
                ticker,
                limitations,
            )
            if ratios_data:
                r = ratios_data[0]
                financials["operating_margin"] = _pct(r.get("operatingProfitMarginTTM"))
                if financials.get("roe") is None:
                    financials["roe"] = _pct(r.get("returnOnEquityTTM"))

            # Income Statement (annual, last 2 for growth)
            income_data = self._get_first_available(
                ["income-statement"],
                ticker,
                limitations,
                limit=2,
            )
            if len(income_data) >= 2:
                rev_now = income_data[0].get("revenue", 0)
                rev_prev = income_data[1].get("revenue", 1)
                if rev_prev and rev_prev != 0:
                    financials["revenue_growth"] = round((rev_now / rev_prev - 1) * 100, 2)
                financials["revenue"] = rev_now
                financials["ebit"] = income_data[0].get("operatingIncome")
                financials["interest_expense"] = income_data[0].get("interestExpense")

            # Cash Flow (FCF)
            cf_data = self._get_first_available(
                ["cash-flow-statement"],
                ticker,
                limitations,
                limit=1,
            )
            if cf_data:
                financials["free_cash_flow"] = cf_data[0].get("freeCashFlow")

            # Balance Sheet
            bs_data = self._get_first_available(
                ["balance-sheet-statement"],
                ticker,
                limitations,
                limit=1,
            )
            if bs_data:
                b = bs_data[0]
                financials["total_assets"] = b.get("totalAssets")
                financials["total_liabilities"] = b.get("totalLiabilities")
                financials["current_assets"] = b.get("totalCurrentAssets")
                financials["current_liabilities"] = b.get("totalCurrentLiabilities")
                financials["retained_earnings"] = b.get("retainedEarnings")
                financials["ebitda"] = b.get("totalAssets")  # placeholder
                net_debt = (b.get("totalDebt") or 0) - (b.get("cashAndCashEquivalents") or 0)
                financials["net_debt"] = net_debt

            # Build evidence for every non-None numeric field
            for k, v in financials.items():
                if isinstance(v, (int, float)) and v is not None:
                    evidence.append(make_evidence(
                        metric=k, value=v,
                        source_name=f"FMP:{ticker}", source_type="api", quality=0.80, as_of=as_of,
                    ))

            # Keep compatibility when some stable endpoints are restricted by plan.
            if financials.get("revenue_growth") is None and financials.get("roe") is None:
                mock_data = self._mock_fundamentals(ticker, as_of)["data"]
                financials["revenue_growth"] = mock_data.get("revenue_growth")
                financials["roe"] = mock_data.get("roe")
                limitations.append("FMP partial access: revenue_growth/roe filled from conservative mock baseline")
                evidence.append(make_evidence(
                    metric="revenue_growth",
                    value=financials["revenue_growth"],
                    source_name="mock_blend",
                    source_type="model",
                    quality=0.3,
                    as_of=as_of,
                    note="fallback due FMP endpoint restrictions",
                ))
                evidence.append(make_evidence(
                    metric="roe",
                    value=financials["roe"],
                    source_name="mock_blend",
                    source_type="model",
                    quality=0.3,
                    as_of=as_of,
                    note="fallback due FMP endpoint restrictions",
                ))

        except ProviderError as e:
            limitations.append(f"FMP API error: {e}")
            return self._mock_fundamentals(ticker, as_of)

        data_ok = len(evidence) >= 3
        if not data_ok:
            limitations.append("Insufficient FMP data for reliable analysis")

        return {
            "data": financials,
            "evidence": evidence,
            "data_ok": data_ok,
            "limitations": limitations,
            "as_of": as_of,
        }

    @staticmethod
    def _mock_fundamentals(ticker: str, as_of: str) -> dict:
        rng = random.Random(hash(ticker) % (2**31))
        ta = rng.uniform(50_000, 500_000)
        tl = rng.uniform(20_000, ta * 0.8)
        financials = {
            "total_assets": round(ta, 0), "current_assets": round(ta * rng.uniform(0.15, 0.40), 0),
            "current_liabilities": round(tl * rng.uniform(0.10, 0.30), 0),
            "retained_earnings": round(ta * rng.uniform(0.05, 0.35), 0),
            "ebit": round(ta * rng.uniform(0.03, 0.15), 0), "ebitda": round(ta * rng.uniform(0.05, 0.18), 0),
            "market_cap": round(ta * rng.uniform(1.5, 6.0), 0), "total_liabilities": round(tl, 0),
            "revenue": round(ta * rng.uniform(0.4, 1.2), 0), "interest_expense": round(ta * rng.uniform(0.005, 0.03), 0),
            "net_debt": round(tl * rng.uniform(0.3, 0.7), 0), "free_cash_flow": round(ta * rng.uniform(-0.02, 0.10), 0),
            "revenue_growth": round(rng.uniform(-5, 25), 2), "operating_margin": round(rng.uniform(5, 35), 2),
            "roe": round(rng.uniform(5, 40), 2), "debt_to_equity": round(tl / max(ta - tl, 1), 2),
            "pe_ratio": round(rng.uniform(10, 50), 2), "sector": "Technology",
        }
        evidence = [
            make_evidence(metric=k, value=v, source_name="mock", quality=0.3, as_of=as_of)
            for k, v in financials.items() if isinstance(v, (int, float))
        ]
        return {"data": financials, "evidence": evidence, "data_ok": False,
                "limitations": ["FMP unavailable; using mock data"], "as_of": as_of}


def _pct(val) -> float | None:
    if val is None:
        return None
    return round(float(val) * 100, 2) if abs(float(val)) < 1 else round(float(val), 2)


if __name__ == "__main__":
    import json
    provider = FMPProvider()
    ticker = "AAPL"
    if provider.has_key:
        print(f"🔑 FMP API key detected — fetching {ticker} fundamentals...")
    else:
        print(f"⚠️  No FMP_API_KEY — using mock for {ticker}")
    result = provider.get_fundamentals(ticker)
    print(json.dumps(result, indent=2, default=str))
