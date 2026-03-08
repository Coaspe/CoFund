"""
data_providers/fred_provider.py — FRED API macro data provider
==============================================================
Live: FRED API (requires FRED_API_KEY)
Mock: deterministic random fallback with data_ok=False
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from schemas.common import make_evidence
from config.settings import get_settings

try:
    from data_providers.base import BaseProvider, ProviderError
except ImportError:
    BaseProvider = object  # type: ignore
    ProviderError = Exception

_FRED_BASE = "https://api.stlouisfed.org/fred"

# Series IDs → macro_agent keys
_SERIES_MAP = {
    "DGS10": "dgs10",
    "DGS2": "dgs2",
    "BAMLH0A0HYM2": "hy_oas",
    "T5YIE": "inflation_expectation",
    "CPIAUCSL": "cpi_level",
    "ISM/MAN_PMI": "pmi",
    "FEDFUNDS": "fed_funds_rate",
    "GDP": "gdp_level",
}


class FREDProvider(BaseProvider if isinstance(BaseProvider, type) else object):
    PROVIDER_NAME = "fred"

    def __init__(self, **kwargs):
        self._api_key = get_settings().fred_api_key
        if BaseProvider is not object:
            super().__init__(**kwargs)

    @property
    def has_key(self) -> bool:
        return bool(self._api_key)

    def get_latest_observation(self, series_id: str) -> Tuple[float | None, str]:
        if not self.has_key:
            raise ProviderError("FRED_API_KEY not set")
        url = f"{_FRED_BASE}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 5,
        }
        data = self.get_json(url, params)
        for obs in data.get("observations", []):
            val = obs.get("value", ".")
            if val != ".":
                return float(val), obs["date"]
        return None, ""

    def get_macro_snapshot(self, as_of: str = "") -> dict:
        as_of = as_of or datetime.now(timezone.utc).isoformat()
        result: Dict[str, Any] = {}
        evidence: list = []
        limitations: list = []

        if not self.has_key:
            return self._mock_snapshot(as_of)

        try:
            # Yield curve spread = DGS10 - DGS2
            dgs10_val, dgs10_date = self.get_latest_observation("DGS10")
            dgs2_val, dgs2_date = self.get_latest_observation("DGS2")
            if dgs10_val is not None and dgs2_val is not None:
                result["dgs10"] = dgs10_val
                result["dgs2"] = dgs2_val
                result["yield_curve_spread"] = round(dgs10_val - dgs2_val, 2)
                evidence.append(make_evidence(metric="dgs10", value=dgs10_val,
                                              source_name="FRED:DGS10", source_type="api", quality=0.85, as_of=dgs10_date))
                evidence.append(make_evidence(metric="dgs2", value=dgs2_val,
                                              source_name="FRED:DGS2", source_type="api", quality=0.85, as_of=dgs2_date))
                evidence.append(make_evidence(metric="yield_curve_spread", value=result["yield_curve_spread"],
                                              source_name="FRED:DGS10-DGS2", source_type="api", quality=0.85, as_of=dgs10_date))

            # HY OAS
            hy_val, hy_date = self.get_latest_observation("BAMLH0A0HYM2")
            if hy_val is not None:
                result["hy_oas"] = hy_val
                evidence.append(make_evidence(metric="hy_oas", value=hy_val,
                                              source_name="FRED:BAMLH0A0HYM2", source_type="api", quality=0.85, as_of=hy_date))

            # Inflation expectation (5Y breakeven)
            ie_val, ie_date = self.get_latest_observation("T5YIE")
            if ie_val is not None:
                result["inflation_expectation"] = ie_val
                evidence.append(make_evidence(metric="inflation_expectation", value=ie_val,
                                              source_name="FRED:T5YIE", source_type="api", quality=0.85, as_of=ie_date))

            # CPI YoY — fetch last 13 months of CPI level, compute YoY %
            cpi_val, cpi_date = self._compute_cpi_yoy()
            if cpi_val is not None:
                result["cpi_yoy"] = cpi_val
                evidence.append(make_evidence(metric="cpi_yoy", value=cpi_val,
                                              source_name="FRED:CPIAUCSL", source_type="api", quality=0.85, as_of=cpi_date))

            # Fed Funds Rate — prefer daily effective rate (DFF), fallback to monthly FEDFUNDS
            ff_val, ff_date, ff_series = None, "", ""
            for series_id in ("DFF", "FEDFUNDS"):
                try:
                    ff_candidate, ff_candidate_date = self.get_latest_observation(series_id)
                except ProviderError:
                    ff_candidate, ff_candidate_date = None, ""
                if ff_candidate is not None:
                    ff_val, ff_date, ff_series = ff_candidate, ff_candidate_date, series_id
                    break
            if ff_val is not None:
                result["fed_funds_rate"] = ff_val
                evidence.append(make_evidence(metric="fed_funds_rate", value=ff_val,
                                              source_name=f"FRED:{ff_series}", source_type="api", quality=0.85, as_of=ff_date))
                if dgs2_val is not None:
                    result["cuts_priced_proxy_2y_ffr_bp"] = round((ff_val - dgs2_val) * 100, 1)
                    evidence.append(make_evidence(metric="cuts_priced_proxy_2y_ffr_bp",
                                                  value=result["cuts_priced_proxy_2y_ffr_bp"],
                                                  source_name=f"derived:DGS2-{ff_series}", source_type="model",
                                                  quality=0.8, as_of=ff_date,
                                                  note="positive means 2Y below effective policy rate; market pricing easier policy"))

            # PMI (ISM Manufacturing — try MANEMP as proxy, or set None)
            try:
                pmi_val, pmi_date = self.get_latest_observation("MANEMP")
                if pmi_val is not None:
                    result["pmi"] = pmi_val
                    evidence.append(make_evidence(metric="pmi", value=pmi_val,
                                                  source_name="FRED:MANEMP", source_type="api", quality=0.6, as_of=pmi_date))
            except ProviderError:
                result["pmi"] = None
                limitations.append("PMI series unavailable from FRED")

            # GDP growth (simple QoQ annualized)
            try:
                gdp_val, gdp_date = self.get_latest_observation("A191RL1Q225SBEA")
                if gdp_val is not None:
                    result["gdp_growth"] = gdp_val
                    evidence.append(make_evidence(metric="gdp_growth", value=gdp_val,
                                                  source_name="FRED:A191RL1Q225SBEA", source_type="api", quality=0.80, as_of=gdp_date))
            except ProviderError:
                result["gdp_growth"] = None
                limitations.append("GDP series unavailable from FRED")

            extra_series = [
                ("DFII10", "real_10y_yield"),
                ("DTWEXBGS", "dollar_index"),
                ("UNRATE", "unemployment_rate"),
                ("NFCI", "financial_conditions_index"),
                ("VIXCLS", "vix_level"),
                ("DCOILWTICO", "wti_spot"),
                ("DCOILBRENTEU", "brent_spot"),
            ]
            for series_id, key in extra_series:
                try:
                    val, obs_date = self.get_latest_observation(series_id)
                    if val is None:
                        continue
                    result[key] = val
                    evidence.append(make_evidence(metric=key, value=val,
                                                  source_name=f"FRED:{series_id}", source_type="api",
                                                  quality=0.82, as_of=obs_date))
                except ProviderError:
                    limitations.append(f"{key} series unavailable from FRED")

        except ProviderError as e:
            limitations.append(f"FRED API error: {e}")
            return self._mock_snapshot(as_of)

        data_ok = len(evidence) >= 3
        return {
            "data": result,
            "evidence": evidence,
            "data_ok": data_ok,
            "limitations": limitations,
            "as_of": as_of,
        }

    def _compute_cpi_yoy(self) -> Tuple[float | None, str]:
        url = f"{_FRED_BASE}/series/observations"
        params = {
            "series_id": "CPIAUCSL",
            "api_key": self._api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 14,
        }
        data = self.get_json(url, params, skip_cache=True)
        obs = [o for o in data.get("observations", []) if o.get("value", ".") != "."]
        if len(obs) < 13:
            return None, ""
        latest = float(obs[0]["value"])
        year_ago = float(obs[12]["value"])
        yoy = round((latest / year_ago - 1) * 100, 2)
        return yoy, obs[0]["date"]

    @staticmethod
    def _mock_snapshot(as_of: str) -> dict:
        rng = random.Random(42)
        indicators = {
            "dgs10": round(rng.uniform(3.0, 5.5), 2),
            "dgs2": round(rng.uniform(2.5, 5.75), 2),
            "yield_curve_spread": round(rng.uniform(-0.5, 2.5), 2),
            "hy_oas": round(rng.uniform(200, 700), 0),
            "inflation_expectation": round(rng.uniform(1.0, 5.5), 2),
            "cpi_yoy": round(rng.uniform(1.5, 6.0), 2),
            "pmi": round(rng.uniform(42, 62), 1),
            "fed_funds_rate": round(rng.uniform(3.0, 5.75), 2),
            "gdp_growth": round(rng.uniform(-1.0, 4.5), 2),
            "real_10y_yield": round(rng.uniform(0.5, 2.5), 2),
            "dollar_index": round(rng.uniform(105, 130), 2),
            "unemployment_rate": round(rng.uniform(3.2, 6.8), 2),
            "financial_conditions_index": round(rng.uniform(-1.0, 1.5), 2),
            "vix_level": round(rng.uniform(12, 35), 2),
            "wti_spot": round(rng.uniform(65, 95), 2),
            "brent_spot": round(rng.uniform(68, 100), 2),
        }
        indicators["cuts_priced_proxy_2y_ffr_bp"] = round((indicators["fed_funds_rate"] - indicators["dgs2"]) * 100, 1)
        evidence = [
            make_evidence(metric=k, value=v, source_name="mock", quality=0.3, as_of=as_of)
            for k, v in indicators.items()
        ]
        return {
            "data": indicators,
            "evidence": evidence,
            "data_ok": False,
            "limitations": ["FRED unavailable; using mock data"],
            "as_of": as_of,
        }


# ── Legacy wrapper for backward compatibility ─────────────────────────────

def fetch_macro_indicators(*, mode: str = "mock", as_of: str = "", seed: int | None = None):
    """Legacy API: returns (indicators_dict, evidence_list)."""
    as_of = as_of or datetime.now(timezone.utc).isoformat()
    if mode == "live":
        provider = FREDProvider()
        snapshot = provider.get_macro_snapshot(as_of=as_of)
        return snapshot["data"], snapshot["evidence"]
    # Mock
    rng = random.Random(seed)
    indicators = {
        "dgs10": round(rng.uniform(3.0, 5.5), 2),
        "dgs2": round(rng.uniform(2.5, 5.75), 2),
        "yield_curve_spread": round(rng.uniform(-0.50, 2.50), 2),
        "hy_oas": round(rng.uniform(200, 700), 0),
        "inflation_expectation": round(rng.uniform(1.0, 5.5), 2),
        "cpi_yoy": round(rng.uniform(1.5, 6.0), 2),
        "pmi": round(rng.uniform(42, 62), 1),
        "fed_funds_rate": round(rng.uniform(3.0, 5.75), 2),
        "gdp_growth": round(rng.uniform(-1.0, 4.5), 2),
        "real_10y_yield": round(rng.uniform(0.5, 2.5), 2),
        "dollar_index": round(rng.uniform(105, 130), 2),
        "unemployment_rate": round(rng.uniform(3.2, 6.8), 2),
        "financial_conditions_index": round(rng.uniform(-1.0, 1.5), 2),
        "vix_level": round(rng.uniform(12, 35), 2),
        "wti_spot": round(rng.uniform(65, 95), 2),
        "brent_spot": round(rng.uniform(68, 100), 2),
    }
    indicators["cuts_priced_proxy_2y_ffr_bp"] = round((indicators["fed_funds_rate"] - indicators["dgs2"]) * 100, 1)
    evidence = [
        make_evidence(metric=k, value=v, source_name="mock", quality=0.3, as_of=as_of)
        for k, v in indicators.items()
    ]
    return indicators, evidence


def fetch_sentiment_indicators(ticker: str, *, mode: str = "mock", as_of: str = "", seed: int | None = None):
    """Legacy sentiment mock — kept for backward compat."""
    as_of = as_of or datetime.now(timezone.utc).isoformat()
    rng = random.Random(seed)
    indicators = {
        "put_call_ratio": round(rng.uniform(0.4, 1.5), 2),
        "pcr_percentile_90d": round(rng.uniform(5, 95), 1),
        "vix_level": round(rng.uniform(10, 40), 2),
        "vix_term_structure": rng.choice(["contango", "flat", "backwardation"]),
        "skew_index": round(rng.uniform(110, 160), 1),
        "short_interest_pct": round(rng.uniform(2, 25), 1),
        "insider_net_activity": rng.choice(["buying", "selling", "neutral"]),
        "news_sentiment_score": round(rng.uniform(-0.8, 0.8), 2),
        "article_count_7d": rng.randint(20, 300),
        "upcoming_events": rng.sample(
            ["2Q Earnings (7d)", "FOMC Meeting (3d)", "CPI Release (5d)", "NFP (10d)"],
            k=rng.randint(0, 2),
        ),
    }
    evidence = [
        make_evidence(metric=k, value=v, source_name="mock", quality=0.3, as_of=as_of)
        for k, v in indicators.items() if not isinstance(v, list)
    ]
    return indicators, evidence


def fetch_fundamentals(ticker: str, *, mode: str = "mock", as_of: str = "", seed: int | None = None):
    """Legacy fundamentals mock — kept for backward compat."""
    as_of = as_of or datetime.now(timezone.utc).isoformat()
    rng = random.Random(seed)
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
        "fcf_history": [round(rng.uniform(-100, 500), 0) for _ in range(5)],
    }
    evidence = [
        make_evidence(metric=k, value=v, source_name="mock", quality=0.3, as_of=as_of)
        for k, v in financials.items() if isinstance(v, (int, float)) and k not in ("total_assets", "total_liabilities")
    ]
    return financials, evidence


if __name__ == "__main__":
    provider = FREDProvider()
    if provider.has_key:
        print("🔑 FRED API key detected — fetching live data...")
        snapshot = provider.get_macro_snapshot()
    else:
        print("⚠️  No FRED_API_KEY — using mock fallback")
        snapshot = provider._mock_snapshot(datetime.now(timezone.utc).isoformat())

    import json
    print(json.dumps(snapshot, indent=2, default=str))
