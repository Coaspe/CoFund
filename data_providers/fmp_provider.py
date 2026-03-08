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
                financials["company_name"] = p.get("companyName")
                financials["sector"] = p.get("sector", "Unknown")
                financials["industry"] = p.get("industry")
                financials["ceo"] = p.get("ceo")
                financials["beta"] = p.get("beta")
                financials["country"] = p.get("country")
                financials["exchange"] = p.get("exchange")
                financials["full_time_employees"] = p.get("fullTimeEmployees")
                financials["last_dividend"] = p.get("lastDividend")
                financials["avg_daily_volume_usd"] = p.get("volAvg", p.get("avgVolume"))
                financials["market_cap"] = p.get("mktCap", p.get("marketCap"))
                financials["current_price"] = p.get("price")

            # Key Metrics TTM
            metrics_data = self._get_first_available(
                ["key-metrics-ttm", "key-metrics"],
                ticker,
                limitations,
            )
            if metrics_data:
                m = metrics_data[0]
                financials["pe_ratio"] = m.get("peRatioTTM")
                financials["ps_ratio"] = m.get("priceToSalesRatioTTM")
                financials["fcf_yield"] = _pct(m.get("freeCashFlowYieldTTM"))
                financials["roe"] = _pct(m.get("roeTTM") or m.get("returnOnEquityTTM"))
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
                if financials.get("ps_ratio") is None:
                    financials["ps_ratio"] = r.get("priceToSalesRatioTTM")
                if financials.get("roe") is None:
                    financials["roe"] = _pct(r.get("returnOnEquityTTM"))
                if financials.get("pe_ratio") is None:
                    current_price = financials.get("current_price")
                    eps_ttm = r.get("netIncomePerShareTTM")
                    if current_price not in (None, 0) and eps_ttm not in (None, 0):
                        financials["pe_ratio"] = round(float(current_price) / float(eps_ttm), 2)

            # Income Statement (annual, last 2 for growth)
            income_data = self._get_first_available(
                ["income-statement"],
                ticker,
                limitations,
                limit=5,
            )
            if len(income_data) >= 2:
                rev_now = income_data[0].get("revenue", 0)
                rev_prev = income_data[1].get("revenue", 1)
                if rev_prev and rev_prev != 0:
                    financials["revenue_growth"] = round((rev_now / rev_prev - 1) * 100, 2)
                financials["revenue"] = rev_now
                financials["ebit"] = income_data[0].get("operatingIncome")
                financials["ebitda"] = income_data[0].get("ebitda")
                financials["interest_expense"] = income_data[0].get("interestExpense")
                ni_now = income_data[0].get("netIncome")
                ni_prev = income_data[1].get("netIncome")
                if ni_now is not None and ni_prev not in (None, 0):
                    financials["earnings_growth"] = round((ni_now / ni_prev - 1) * 100, 2)

            # Cash Flow (FCF)
            cf_data = self._get_first_available(
                ["cash-flow-statement"],
                ticker,
                limitations,
                limit=5,
            )
            if cf_data:
                financials["free_cash_flow"] = cf_data[0].get("freeCashFlow")
                financials["fcf_history"] = [
                    row.get("freeCashFlow") for row in cf_data if row.get("freeCashFlow") is not None
                ]
                latest_cf = cf_data[0]
                repurchased = abs(float(latest_cf.get("commonStockRepurchased") or 0))
                dividends = abs(float(latest_cf.get("commonDividendsPaid") or latest_cf.get("netDividendsPaid") or 0))
                acquisitions = abs(float(latest_cf.get("acquisitionsNet") or 0))
                debt_issuance = float(latest_cf.get("netDebtIssuance") or 0)
                financials["buybacks_last_fy"] = round(repurchased, 2) if repurchased else 0.0
                financials["dividends_last_fy"] = round(dividends, 2) if dividends else 0.0
                financials["acquisitions_last_fy"] = round(acquisitions, 2) if acquisitions else 0.0
                financials["net_debt_issuance_last_fy"] = round(debt_issuance, 2) if debt_issuance else 0.0
                financials["stock_based_comp_last_fy"] = latest_cf.get("stockBasedCompensation")
                mcap = financials.get("market_cap")
                if mcap not in (None, 0):
                    financials["buyback_yield_pct"] = round(repurchased / mcap * 100, 2)
                    financials["dividend_cash_yield_pct"] = round(dividends / mcap * 100, 2)
                    financials["capital_return_yield_pct"] = round((repurchased + dividends) / mcap * 100, 2)
                financials["debt_funded_buyback_flag"] = bool(repurchased > 0 and debt_issuance > 0)

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
                net_debt = (b.get("totalDebt") or 0) - (b.get("cashAndCashEquivalents") or 0)
                financials["net_debt"] = net_debt

            self._attach_earnings_context(financials, ticker, limitations, as_of)
            self._attach_estimate_context(financials, ticker, limitations, as_of)
            self._attach_price_target_context(financials, ticker, limitations)
            self._attach_rating_context(financials, ticker, limitations)
            self._attach_quote_context(financials, ticker, limitations)
            self._attach_grades_context(financials, ticker, limitations, as_of)

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
        current_price = round(rng.uniform(50, 300), 2)
        financials = {
            "total_assets": round(ta, 0), "current_assets": round(ta * rng.uniform(0.15, 0.40), 0),
            "current_liabilities": round(tl * rng.uniform(0.10, 0.30), 0),
            "retained_earnings": round(ta * rng.uniform(0.05, 0.35), 0),
            "ebit": round(ta * rng.uniform(0.03, 0.15), 0), "ebitda": round(ta * rng.uniform(0.05, 0.18), 0),
            "market_cap": round(ta * rng.uniform(1.5, 6.0), 0), "total_liabilities": round(tl, 0),
            "revenue": round(ta * rng.uniform(0.4, 1.2), 0), "interest_expense": round(ta * rng.uniform(0.005, 0.03), 0),
            "net_debt": round(tl * rng.uniform(0.3, 0.7), 0), "free_cash_flow": round(ta * rng.uniform(-0.02, 0.10), 0),
            "fcf_history": [round(ta * rng.uniform(-0.02, 0.10), 0) for _ in range(5)],
            "revenue_growth": round(rng.uniform(-5, 25), 2), "operating_margin": round(rng.uniform(5, 35), 2),
            "earnings_growth": round(rng.uniform(-10, 30), 2),
            "roe": round(rng.uniform(5, 40), 2), "debt_to_equity": round(tl / max(ta - tl, 1), 2),
            "pe_ratio": round(rng.uniform(10, 50), 2), "ps_ratio": round(rng.uniform(2, 12), 2),
            "fcf_yield": round(rng.uniform(1, 8), 2), "sector": "Technology",
            "industry": "Consumer Electronics",
            "ceo": "Jane Doe",
            "beta": round(rng.uniform(0.8, 1.6), 2),
            "last_dividend": round(rng.uniform(0.0, 2.0), 2),
            "current_price": current_price,
            "price_avg_50d": round(current_price * rng.uniform(0.92, 1.08), 2),
            "price_avg_200d": round(current_price * rng.uniform(0.85, 1.15), 2),
            "next_earnings_date": "2026-04-30",
            "earnings_in_days": 21,
            "next_eps_estimate": round(rng.uniform(1.0, 3.0), 2),
            "next_revenue_estimate": round(ta * rng.uniform(0.42, 1.15), 0),
            "analyst_eps_estimate_next_fy": round(rng.uniform(6, 14), 2),
            "analyst_revenue_estimate_next_fy": round(ta * rng.uniform(0.45, 1.2), 0),
            "price_target_consensus": round(current_price * rng.uniform(0.95, 1.20), 2),
            "price_target_high": round(current_price * rng.uniform(1.10, 1.35), 2),
            "price_target_low": round(current_price * rng.uniform(0.80, 0.95), 2),
            "rating": rng.choice(["A", "B", "C"]),
            "rating_overall_score": rng.randint(2, 4),
            "rating_dcf_score": rng.randint(2, 4),
            "analyst_consensus_label": rng.choice(["Buy", "Hold"]),
            "grades_consensus_buy": rng.randint(10, 40),
            "grades_consensus_hold": rng.randint(5, 20),
            "grades_consensus_sell": rng.randint(0, 5),
            "rating_revision_score_30d": round(rng.uniform(-2, 4), 2),
            "rating_revision_score_90d": round(rng.uniform(-4, 6), 2),
            "street_revision_proxy": rng.choice(["improving", "stable", "deteriorating"]),
            "buybacks_last_fy": round(ta * rng.uniform(0.00, 0.08), 0),
            "dividends_last_fy": round(ta * rng.uniform(0.00, 0.04), 0),
            "acquisitions_last_fy": round(ta * rng.uniform(0.00, 0.03), 0),
            "net_debt_issuance_last_fy": round(ta * rng.uniform(-0.02, 0.04), 0),
            "buyback_yield_pct": round(rng.uniform(0, 6), 2),
            "dividend_cash_yield_pct": round(rng.uniform(0, 3), 2),
            "capital_return_yield_pct": round(rng.uniform(0, 8), 2),
            "debt_funded_buyback_flag": rng.choice([True, False]),
        }
        if financials["current_price"] and financials["pe_ratio"]:
            eps_ttm = financials["current_price"] / financials["pe_ratio"]
            financials["eps_ttm_proxy"] = round(eps_ttm, 4)
            if financials.get("price_avg_50d"):
                financials["pe_ratio_50d_proxy"] = round(financials["price_avg_50d"] / eps_ttm, 2)
            if financials.get("price_avg_200d"):
                financials["pe_ratio_200d_proxy"] = round(financials["price_avg_200d"] / eps_ttm, 2)
        if financials["current_price"] and financials.get("market_cap"):
            financials["shares_outstanding_est"] = round(financials["market_cap"] / financials["current_price"], 0)
        if financials["current_price"] > 0:
            financials["price_target_upside_pct"] = round(
                (financials["price_target_consensus"] / financials["current_price"] - 1) * 100, 2
            )
        evidence = [
            make_evidence(metric=k, value=v, source_name="mock", quality=0.3, as_of=as_of)
            for k, v in financials.items() if isinstance(v, (int, float))
        ]
        return {"data": financials, "evidence": evidence, "data_ok": False,
                "limitations": ["FMP unavailable; using mock data"], "as_of": as_of}

    def _attach_earnings_context(self, financials: Dict[str, Any], ticker: str, limitations: list[str], as_of: str) -> None:
        data = self._get_first_available(["earnings"], ticker, limitations)
        if not data:
            return
        today = _safe_date(as_of)
        if today is None:
            return
        rows = [row for row in data if row.get("date")]
        future_rows = []
        past_rows = []
        for row in rows:
            d = _safe_date(row.get("date"))
            if d is None:
                continue
            if d >= today:
                future_rows.append((d, row))
            else:
                past_rows.append((d, row))
        if future_rows:
            next_date, next_row = min(future_rows, key=lambda x: x[0])
            financials["next_earnings_date"] = next_date.isoformat()
            financials["earnings_in_days"] = (next_date - today).days
            financials["next_eps_estimate"] = next_row.get("epsEstimated")
            financials["next_revenue_estimate"] = next_row.get("revenueEstimated")
            financials["earnings_last_updated"] = next_row.get("lastUpdated")
        if past_rows:
            last_date, last_row = max(past_rows, key=lambda x: x[0])
            financials["last_earnings_date"] = last_date.isoformat()
            financials["last_eps_actual"] = last_row.get("epsActual")
            financials["last_eps_estimate"] = last_row.get("epsEstimated")
            financials["last_revenue_actual"] = last_row.get("revenueActual")
            financials["last_revenue_estimate"] = last_row.get("revenueEstimated")
            eps_actual = last_row.get("epsActual")
            eps_est = last_row.get("epsEstimated")
            if eps_actual is not None and eps_est not in (None, 0):
                financials["last_eps_surprise_pct"] = round((eps_actual / eps_est - 1) * 100, 2)
            hist = []
            eps_beats = 0
            rev_beats = 0
            eps_surprises = []
            rev_surprises = []
            for d, row in sorted(past_rows, key=lambda x: x[0], reverse=True)[:4]:
                eps_actual = row.get("epsActual")
                eps_est = row.get("epsEstimated")
                rev_actual = row.get("revenueActual")
                rev_est = row.get("revenueEstimated")
                eps_surprise = None
                rev_surprise = None
                if eps_actual is not None and eps_est not in (None, 0):
                    eps_surprise = round((eps_actual / eps_est - 1) * 100, 2)
                    if eps_surprise > 0:
                        eps_beats += 1
                    eps_surprises.append(eps_surprise)
                if rev_actual is not None and rev_est not in (None, 0):
                    rev_surprise = round((rev_actual / rev_est - 1) * 100, 2)
                    if rev_surprise > 0:
                        rev_beats += 1
                    rev_surprises.append(rev_surprise)
                hist.append(
                    {
                        "date": d.isoformat(),
                        "eps_actual": eps_actual,
                        "eps_estimate": eps_est,
                        "eps_surprise_pct": eps_surprise,
                        "revenue_actual": rev_actual,
                        "revenue_estimate": rev_est,
                        "revenue_surprise_pct": rev_surprise,
                    }
                )
            financials["earnings_surprise_history"] = hist
            if hist:
                financials["earnings_beat_rate_4q"] = round(eps_beats / len(hist) * 100, 1)
                financials["revenue_beat_rate_4q"] = round(rev_beats / len(hist) * 100, 1)
            if eps_surprises:
                financials["eps_surprise_avg_4q"] = round(sum(eps_surprises) / len(eps_surprises), 2)
            if rev_surprises:
                financials["revenue_surprise_avg_4q"] = round(sum(rev_surprises) / len(rev_surprises), 2)

    def _attach_estimate_context(self, financials: Dict[str, Any], ticker: str, limitations: list[str], as_of: str) -> None:
        data = self._get_first_available(
            ["analyst-estimates"],
            ticker,
            limitations,
            period="annual",
            page=0,
            limit=5,
        )
        if not data:
            return
        dated = []
        for row in data:
            d = _safe_date(row.get("date"))
            if d is not None:
                dated.append((d, row))
        if not dated:
            return
        today = _safe_date(as_of)
        future = [(d, row) for d, row in dated if today is None or d >= today]
        if future:
            est_date, row = min(future, key=lambda x: x[0])
        else:
            est_date, row = max(dated, key=lambda x: x[0])
        financials["analyst_estimate_fy"] = est_date.isoformat()
        financials["analyst_revenue_estimate_next_fy"] = row.get("revenueAvg")
        financials["analyst_ebitda_estimate_next_fy"] = row.get("ebitdaAvg")
        financials["analyst_eps_estimate_next_fy"] = row.get("epsAvg")
        financials["analyst_num_revenue"] = row.get("numAnalystsRevenue")
        financials["analyst_num_eps"] = row.get("numAnalystsEps")

    def _attach_price_target_context(self, financials: Dict[str, Any], ticker: str, limitations: list[str]) -> None:
        data = self._get_first_available(["price-target-consensus"], ticker, limitations)
        if not data:
            return
        row = data[0]
        financials["price_target_consensus"] = row.get("targetConsensus")
        financials["price_target_median"] = row.get("targetMedian")
        financials["price_target_high"] = row.get("targetHigh")
        financials["price_target_low"] = row.get("targetLow")
        current_price = financials.get("current_price")
        target = financials.get("price_target_consensus")
        if current_price not in (None, 0) and target is not None:
            financials["price_target_upside_pct"] = round((target / current_price - 1) * 100, 2)

    def _attach_rating_context(self, financials: Dict[str, Any], ticker: str, limitations: list[str]) -> None:
        data = self._get_first_available(["ratings-snapshot"], ticker, limitations)
        if not data:
            return
        row = data[0]
        financials["rating"] = row.get("rating")
        financials["rating_overall_score"] = row.get("overallScore")
        financials["rating_dcf_score"] = row.get("discountedCashFlowScore")

    def _attach_quote_context(self, financials: Dict[str, Any], ticker: str, limitations: list[str]) -> None:
        data = self._get_first_available(["quote"], ticker, limitations)
        if not data:
            return
        row = data[0]
        current_price = row.get("price")
        if financials.get("current_price") is None:
            financials["current_price"] = current_price
        financials["price_avg_50d"] = row.get("priceAvg50")
        financials["price_avg_200d"] = row.get("priceAvg200")
        financials["year_high"] = row.get("yearHigh")
        financials["year_low"] = row.get("yearLow")
        financials["previous_close"] = row.get("previousClose")
        if financials.get("market_cap") is None:
            financials["market_cap"] = row.get("marketCap")
        pe = financials.get("pe_ratio")
        if current_price not in (None, 0) and pe not in (None, 0):
            eps_ttm = current_price / pe
            financials["eps_ttm_proxy"] = round(eps_ttm, 4)
            avg50 = financials.get("price_avg_50d")
            avg200 = financials.get("price_avg_200d")
            if avg50 not in (None, 0):
                financials["pe_ratio_50d_proxy"] = round(avg50 / eps_ttm, 2)
            if avg200 not in (None, 0):
                financials["pe_ratio_200d_proxy"] = round(avg200 / eps_ttm, 2)
        mcap = financials.get("market_cap")
        if current_price not in (None, 0) and mcap not in (None, 0):
            financials["shares_outstanding_est"] = round(mcap / current_price, 0)

    def _attach_grades_context(self, financials: Dict[str, Any], ticker: str, limitations: list[str], as_of: str) -> None:
        consensus_data = self._get_first_available(["grades-consensus"], ticker, limitations)
        if consensus_data:
            row = consensus_data[0]
            financials["analyst_consensus_label"] = row.get("consensus")
            financials["grades_consensus_strong_buy"] = row.get("strongBuy")
            financials["grades_consensus_buy"] = row.get("buy")
            financials["grades_consensus_hold"] = row.get("hold")
            financials["grades_consensus_sell"] = row.get("sell")
            financials["grades_consensus_strong_sell"] = row.get("strongSell")

        grades_data = self._get_first_available(["grades"], ticker, limitations, limit=200)
        if not grades_data:
            return
        today = _safe_date(as_of)
        recent_actions = []
        score_30d = 0.0
        score_90d = 0.0
        counts_30d = {"upgrade": 0, "downgrade": 0, "maintain": 0, "init": 0}
        counts_90d = {"upgrade": 0, "downgrade": 0, "maintain": 0, "init": 0}
        for row in grades_data:
            d = _safe_date(row.get("date"))
            if d is None:
                continue
            action = str(row.get("action", "")).strip().lower()
            if len(recent_actions) < 6:
                recent_actions.append(
                    {
                        "date": d.isoformat(),
                        "firm": row.get("gradingCompany"),
                        "action": action or "unknown",
                        "previous": row.get("previousGrade"),
                        "new": row.get("newGrade"),
                    }
                )
            if today is None:
                continue
            delta = (today - d).days
            score = _grade_action_score(action, row.get("previousGrade"), row.get("newGrade"))
            bucket = _grade_action_bucket(action, row.get("previousGrade"), row.get("newGrade"))
            if delta <= 30:
                score_30d += score
                counts_30d[bucket] = counts_30d.get(bucket, 0) + 1
            if delta <= 90:
                score_90d += score
                counts_90d[bucket] = counts_90d.get(bucket, 0) + 1
        financials["recent_rating_actions"] = recent_actions
        financials["rating_revision_score_30d"] = round(score_30d, 2)
        financials["rating_revision_score_90d"] = round(score_90d, 2)
        financials["rating_action_counts_30d"] = counts_30d
        financials["rating_action_counts_90d"] = counts_90d
        if score_30d >= 2:
            financials["street_revision_proxy"] = "improving"
        elif score_30d <= -2:
            financials["street_revision_proxy"] = "deteriorating"
        else:
            financials["street_revision_proxy"] = "stable"

    def get_peer_context(self, ticker: str, as_of: str = "", max_peers: int = 5) -> dict:
        as_of = as_of or datetime.now(timezone.utc).isoformat()
        if not self.has_key:
            return self._mock_peer_context(ticker, as_of, max_peers=max_peers)
        limitations: list[str] = []
        evidence: list = []
        peers_raw = self._get_first_available(["stock-peers"], ticker, limitations)
        if not peers_raw:
            return {"data": {"peers": [], "status": "insufficient_data"}, "evidence": evidence, "data_ok": False, "limitations": limitations, "as_of": as_of}
        symbols = []
        for row in peers_raw:
            sym = str(row.get("symbol", "")).strip().upper()
            if sym and sym != ticker.upper() and sym not in symbols:
                symbols.append(sym)
            if len(symbols) >= max_peers:
                break
        peers = []
        for sym in symbols:
            snap = self._get_peer_snapshot(sym, limitations)
            if snap:
                peers.append(snap)
                if snap.get("pe_ratio") is not None:
                    evidence.append(make_evidence(metric=f"peer_pe:{sym}", value=snap.get("pe_ratio"), source_name=f"FMP:peer:{sym}", source_type="api", quality=0.78, as_of=as_of))
        data_ok = len(peers) >= 2
        peer_pe = [p["pe_ratio"] for p in peers if p.get("pe_ratio") not in (None, 0)]
        peer_ps = [p["ps_ratio"] for p in peers if p.get("ps_ratio") not in (None, 0)]
        peer_roe = [p["roe"] for p in peers if p.get("roe") is not None]
        peer_rev = [p["revenue_growth"] for p in peers if p.get("revenue_growth") is not None]
        peer_margin = [p["operating_margin"] for p in peers if p.get("operating_margin") is not None]
        data = {
            "status": "ok" if data_ok else "insufficient_data",
            "peer_symbols": symbols,
            "peers": peers,
            "peer_median_pe": _median(peer_pe),
            "peer_median_ps": _median(peer_ps),
            "peer_median_roe": _median(peer_roe),
            "peer_median_revenue_growth": _median(peer_rev),
            "peer_median_operating_margin": _median(peer_margin),
        }
        return {"data": data, "evidence": evidence, "data_ok": data_ok, "limitations": limitations, "as_of": as_of}

    def _get_peer_snapshot(self, ticker: str, limitations: list[str]) -> dict:
        out: Dict[str, Any] = {"symbol": ticker}
        profile = self._get_first_available(["profile"], ticker, limitations)
        if profile:
            row = profile[0]
            out["company_name"] = row.get("companyName")
            out["sector"] = row.get("sector")
            out["industry"] = row.get("industry")
            out["market_cap"] = row.get("mktCap", row.get("marketCap"))
            out["current_price"] = row.get("price")
        metrics = self._get_first_available(["key-metrics-ttm", "key-metrics"], ticker, limitations)
        if metrics:
            row = metrics[0]
            out["pe_ratio"] = row.get("peRatioTTM")
            out["ps_ratio"] = row.get("priceToSalesRatioTTM")
            out["roe"] = _pct(row.get("roeTTM") or row.get("returnOnEquityTTM"))
        ratios = self._get_first_available(["ratios-ttm", "ratios"], ticker, limitations)
        if ratios:
            row = ratios[0]
            out["operating_margin"] = _pct(row.get("operatingProfitMarginTTM"))
            if out.get("ps_ratio") is None:
                out["ps_ratio"] = row.get("priceToSalesRatioTTM")
            if out.get("roe") is None:
                out["roe"] = _pct(row.get("returnOnEquityTTM"))
            if out.get("pe_ratio") is None:
                current_price = out.get("current_price")
                eps_ttm = row.get("netIncomePerShareTTM")
                if current_price not in (None, 0) and eps_ttm not in (None, 0):
                    out["pe_ratio"] = round(float(current_price) / float(eps_ttm), 2)
        income = self._get_first_available(["income-statement"], ticker, limitations, limit=2)
        if len(income) >= 2:
            rev_now = income[0].get("revenue")
            rev_prev = income[1].get("revenue")
            if rev_now is not None and rev_prev not in (None, 0):
                out["revenue_growth"] = round((rev_now / rev_prev - 1) * 100, 2)
        return out if len(out) > 1 else {}

    @staticmethod
    def _mock_peer_context(ticker: str, as_of: str, max_peers: int = 5) -> dict:
        base = ["MSFT", "GOOGL", "AMZN", "META", "NVDA", "ORCL"]
        symbols = [sym for sym in base if sym != ticker.upper()][:max_peers]
        peers = []
        for idx, sym in enumerate(symbols):
            mock = FMPProvider._mock_fundamentals(sym, as_of)["data"]
            peers.append(
                {
                    "symbol": sym,
                    "company_name": sym,
                    "sector": mock.get("sector"),
                    "industry": mock.get("industry"),
                    "market_cap": mock.get("market_cap"),
                    "current_price": mock.get("current_price"),
                    "pe_ratio": mock.get("pe_ratio"),
                    "ps_ratio": mock.get("ps_ratio"),
                    "roe": mock.get("roe"),
                    "operating_margin": mock.get("operating_margin"),
                    "revenue_growth": mock.get("revenue_growth"),
                }
            )
        peer_pe = [p["pe_ratio"] for p in peers if p.get("pe_ratio") not in (None, 0)]
        peer_ps = [p["ps_ratio"] for p in peers if p.get("ps_ratio") not in (None, 0)]
        peer_roe = [p["roe"] for p in peers if p.get("roe") is not None]
        peer_rev = [p["revenue_growth"] for p in peers if p.get("revenue_growth") is not None]
        peer_margin = [p["operating_margin"] for p in peers if p.get("operating_margin") is not None]
        return {
            "data": {
                "status": "ok",
                "peer_symbols": symbols,
                "peers": peers,
                "peer_median_pe": _median(peer_pe),
                "peer_median_ps": _median(peer_ps),
                "peer_median_roe": _median(peer_roe),
                "peer_median_revenue_growth": _median(peer_rev),
                "peer_median_operating_margin": _median(peer_margin),
            },
            "evidence": [
                make_evidence(metric=f"peer_pe:{p['symbol']}", value=p.get("pe_ratio"), source_name="mock_peer", quality=0.3, as_of=as_of)
                for p in peers if p.get("pe_ratio") is not None
            ],
            "data_ok": True,
            "limitations": ["Peer context unavailable live; using mock peers"],
            "as_of": as_of,
        }


def _pct(val) -> float | None:
    if val is None:
        return None
    return round(float(val) * 100, 2) if abs(float(val)) < 1 else round(float(val), 2)


def _median(values: list[float]) -> float | None:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    if n % 2:
        return round(vals[mid], 4)
    return round((vals[mid - 1] + vals[mid]) / 2, 4)


def _grade_action_bucket(action: Any, previous: Any, new: Any) -> str:
    act = str(action or "").strip().lower()
    prev = str(previous or "").strip().lower()
    nxt = str(new or "").strip().lower()
    if "down" in act:
        return "downgrade"
    if "up" in act:
        return "upgrade"
    if act in {"initiate", "initiated", "resume", "resumed", "reiterate", "reiterated"}:
        return "init"
    if prev and nxt and prev != nxt:
        bullish = {"buy", "outperform", "overweight", "strong buy"}
        bearish = {"sell", "underperform", "underweight", "strong sell"}
        if prev in bearish and nxt in bullish:
            return "upgrade"
        if prev in bullish and nxt in bearish:
            return "downgrade"
    return "maintain"


def _grade_action_score(action: Any, previous: Any, new: Any) -> float:
    bucket = _grade_action_bucket(action, previous, new)
    if bucket == "upgrade":
        return 1.0
    if bucket == "downgrade":
        return -1.0
    if bucket == "init":
        nxt = str(new or "").strip().lower()
        if nxt in {"buy", "outperform", "overweight", "strong buy"}:
            return 0.5
        if nxt in {"sell", "underperform", "underweight", "strong sell"}:
            return -0.5
        return 0.0
    return 0.0


def _safe_date(val: Any):
    if val is None:
        return None
    text = str(val).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return datetime.strptime(text[:10], "%Y-%m-%d").date()
        except ValueError:
            return None


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
