"""
data_providers/yahoo_structured_provider.py — Structured Yahoo Finance provider
==============================================================================
Uses yfinance structured quote summary tables for:
  - estimate / revision trends
  - ownership / holder concentration

No API key required. Dates from returned tables are preserved where available.
"""

from __future__ import annotations

import math
import random
from datetime import date, datetime, timezone
from typing import Any, Dict, List

from schemas.common import make_evidence

try:
    from data_providers.base import BaseProvider
except ImportError:
    BaseProvider = object  # type: ignore


class YahooStructuredProvider(BaseProvider if isinstance(BaseProvider, type) else object):
    PROVIDER_NAME = "yahoo_structured"

    def get_estimate_revision_snapshot(self, ticker: str, as_of: str = "") -> dict:
        as_of = as_of or datetime.now(timezone.utc).isoformat()
        try:
            data = self._fetch_live_estimate_revision(ticker)
            evidence = self._build_estimate_evidence(data, as_of)
            return {
                "data": data,
                "evidence": evidence,
                "data_ok": bool(data.get("status") == "ok"),
                "limitations": [],
                "as_of": as_of,
            }
        except Exception as e:
            data = self._mock_estimate_revision(ticker, as_of)
            evidence = self._build_estimate_evidence(data, as_of, source_name="mock", quality=0.3)
            return {
                "data": data,
                "evidence": evidence,
                "data_ok": False,
                "limitations": [f"yahoo structured estimate live error: {e}", "Using mock estimate revision snapshot"],
                "as_of": as_of,
            }

    def get_ownership_snapshot(self, ticker: str, as_of: str = "") -> dict:
        as_of = as_of or datetime.now(timezone.utc).isoformat()
        try:
            data = self._fetch_live_ownership(ticker)
            evidence = self._build_ownership_evidence(data, as_of)
            return {
                "data": data,
                "evidence": evidence,
                "data_ok": bool(data.get("status") == "ok"),
                "limitations": [],
                "as_of": as_of,
            }
        except Exception as e:
            data = self._mock_ownership_snapshot(ticker, as_of)
            evidence = self._build_ownership_evidence(data, as_of, source_name="mock", quality=0.3)
            return {
                "data": data,
                "evidence": evidence,
                "data_ok": False,
                "limitations": [f"yahoo structured ownership live error: {e}", "Using mock ownership snapshot"],
                "as_of": as_of,
            }

    def get_fundamental_history_snapshot(self, ticker: str, as_of: str = "") -> dict:
        as_of = as_of or datetime.now(timezone.utc).isoformat()
        try:
            data = self._fetch_live_fundamental_history(ticker)
            evidence = self._build_history_evidence(data, as_of)
            return {
                "data": data,
                "evidence": evidence,
                "data_ok": bool(data.get("valuation_history_real", {}).get("status") == "ok"),
                "limitations": [],
                "as_of": as_of,
            }
        except Exception as e:
            data = self._mock_fundamental_history_snapshot(ticker, as_of)
            evidence = self._build_history_evidence(data, as_of, source_name="mock", quality=0.3)
            return {
                "data": data,
                "evidence": evidence,
                "data_ok": False,
                "limitations": [f"yahoo structured history live error: {e}", "Using mock fundamental history snapshot"],
                "as_of": as_of,
            }

    def _fetch_live_estimate_revision(self, ticker: str) -> dict:
        import yfinance as yf  # type: ignore

        t = yf.Ticker(ticker)
        earnings_estimate = _table_records_by_index(getattr(t, "earnings_estimate", None))
        revenue_estimate = _table_records_by_index(getattr(t, "revenue_estimate", None))
        eps_trend = _table_records_by_index(getattr(t, "eps_trend", None))
        eps_revisions = _table_records_by_index(getattr(t, "eps_revisions", None))
        growth_estimates = _table_records_by_index(getattr(t, "growth_estimates", None))
        recommendations_summary = _table_records_by_period(getattr(t, "recommendations_summary", None))
        analyst_price_targets = getattr(t, "analyst_price_targets", None) or {}
        calendar = getattr(t, "calendar", None) or {}

        periods = {
            "0q": {
                "eps_estimate": _field(earnings_estimate, "0q", "avg"),
                "eps_low": _field(earnings_estimate, "0q", "low"),
                "eps_high": _field(earnings_estimate, "0q", "high"),
                "revenue_estimate": _field(revenue_estimate, "0q", "avg"),
                "revenue_low": _field(revenue_estimate, "0q", "low"),
                "revenue_high": _field(revenue_estimate, "0q", "high"),
                "eps_growth": _field(earnings_estimate, "0q", "growth"),
                "revenue_growth": _field(revenue_estimate, "0q", "growth"),
                "eps_revision_30d_pct": _revision_pct(eps_trend, "0q", "30daysAgo"),
                "eps_revision_90d_pct": _revision_pct(eps_trend, "0q", "90daysAgo"),
                "up_last_30d": _field(eps_revisions, "0q", "upLast30days"),
                "down_last_30d": _field(eps_revisions, "0q", "downLast30days"),
            },
            "0y": {
                "eps_estimate": _field(earnings_estimate, "0y", "avg"),
                "revenue_estimate": _field(revenue_estimate, "0y", "avg"),
                "eps_growth": _field(earnings_estimate, "0y", "growth"),
                "revenue_growth": _field(revenue_estimate, "0y", "growth"),
                "eps_revision_30d_pct": _revision_pct(eps_trend, "0y", "30daysAgo"),
                "eps_revision_90d_pct": _revision_pct(eps_trend, "0y", "90daysAgo"),
                "up_last_30d": _field(eps_revisions, "0y", "upLast30days"),
                "down_last_30d": _field(eps_revisions, "0y", "downLast30days"),
            },
            "+1y": {
                "eps_estimate": _field(earnings_estimate, "+1y", "avg"),
                "revenue_estimate": _field(revenue_estimate, "+1y", "avg"),
                "eps_growth": _field(earnings_estimate, "+1y", "growth"),
                "revenue_growth": _field(revenue_estimate, "+1y", "growth"),
                "eps_revision_30d_pct": _revision_pct(eps_trend, "+1y", "30daysAgo"),
                "eps_revision_90d_pct": _revision_pct(eps_trend, "+1y", "90daysAgo"),
                "up_last_30d": _field(eps_revisions, "+1y", "upLast30days"),
                "down_last_30d": _field(eps_revisions, "+1y", "downLast30days"),
            },
        }
        for period, row in periods.items():
            row["revision_state"] = _revision_state(
                row.get("eps_revision_30d_pct"),
                row.get("eps_revision_90d_pct"),
                row.get("up_last_30d"),
                row.get("down_last_30d"),
            )

        earnings_dates = calendar.get("Earnings Date")
        next_earnings_date = None
        if isinstance(earnings_dates, list) and earnings_dates:
            next_earnings_date = _clean_value(earnings_dates[0])

        data = {
            "status": "ok" if periods["0q"]["eps_estimate"] is not None or periods["0y"]["eps_estimate"] is not None else "insufficient_data",
            "estimate_source": "yfinance_quote_summary",
            "estimate_periods": periods,
            "earnings_estimate_table": earnings_estimate,
            "revenue_estimate_table": revenue_estimate,
            "eps_trend_table": eps_trend,
            "eps_revisions_table": eps_revisions,
            "growth_estimates_table": growth_estimates,
            "recommendations_summary": recommendations_summary,
            "analyst_price_targets_yahoo": {
                "current": _clean_value(analyst_price_targets.get("current")),
                "mean": _clean_value(analyst_price_targets.get("mean")),
                "median": _clean_value(analyst_price_targets.get("median")),
                "high": _clean_value(analyst_price_targets.get("high")),
                "low": _clean_value(analyst_price_targets.get("low")),
            },
            "next_earnings_date_yahoo": next_earnings_date,
        }
        return data

    def _fetch_live_ownership(self, ticker: str) -> dict:
        import yfinance as yf  # type: ignore

        t = yf.Ticker(ticker)
        institutional = _table_records(getattr(t, "institutional_holders", None))
        mutualfund = _table_records(getattr(t, "mutualfund_holders", None))
        major = _major_holders_dict(getattr(t, "major_holders", None))
        insider_transactions = _table_records(getattr(t, "insider_transactions", None))
        insider_purchases = _insider_purchases_dict(getattr(t, "insider_purchases", None))
        info = {}
        try:
            info = t.get_info()
        except Exception:
            info = {}

        top_inst = institutional[:10]
        top_mf = mutualfund[:10]
        institutional_top10_pct = round(sum(float(x.get("pctHeld") or 0.0) for x in top_inst), 4) if top_inst else None
        mutualfund_top10_pct = round(sum(float(x.get("pctHeld") or 0.0) for x in top_mf), 4) if top_mf else None
        top_holder_pct = round(float(top_inst[0].get("pctHeld") or 0.0), 4) if top_inst else None
        hhi_top10 = None
        if top_inst:
            weights = [float(x.get("pctHeld") or 0.0) for x in top_inst]
            hhi_top10 = round(sum(w * w for w in weights), 6)

        increases = sorted(
            [x for x in institutional if _safe_num(x.get("pctChange")) not in (None, 0)],
            key=lambda x: _safe_num(x.get("pctChange")) or 0.0,
            reverse=True,
        )
        decreases = sorted(
            [x for x in institutional if _safe_num(x.get("pctChange")) not in (None, 0)],
            key=lambda x: _safe_num(x.get("pctChange")) or 0.0,
        )

        buyers = [
            {
                "holder": row.get("Holder"),
                "pct_change": _safe_num(row.get("pctChange")),
                "pct_held": _safe_num(row.get("pctHeld")),
                "date_reported": _clean_value(row.get("Date Reported")),
            }
            for row in increases[:5]
            if (_safe_num(row.get("pctChange")) or 0.0) > 0
        ]
        sellers = [
            {
                "holder": row.get("Holder"),
                "pct_change": _safe_num(row.get("pctChange")),
                "pct_held": _safe_num(row.get("pctHeld")),
                "date_reported": _clean_value(row.get("Date Reported")),
            }
            for row in decreases[:5]
            if (_safe_num(row.get("pctChange")) or 0.0) < 0
        ]

        institutions_percent_held = _safe_num(major.get("institutionsPercentHeld"))
        insiders_percent_held = _safe_num(major.get("insidersPercentHeld"))
        institutions_float_percent_held = _safe_num(major.get("institutionsFloatPercentHeld"))
        institutions_count = _safe_num(major.get("institutionsCount"))
        net_shares = _safe_num(insider_purchases.get("Net Shares Purchased (Sold)"))
        insider_activity = "neutral"
        if net_shares is not None:
            if net_shares > 0:
                insider_activity = "buying"
            elif net_shares < 0:
                insider_activity = "selling"

        concentration_level = "unknown"
        if institutional_top10_pct is not None and top_holder_pct is not None:
            if institutional_top10_pct >= 0.35 or top_holder_pct >= 0.12:
                concentration_level = "high"
            elif institutional_top10_pct >= 0.22:
                concentration_level = "medium"
            else:
                concentration_level = "low"

        crowding_risk = "unknown"
        if institutions_percent_held is not None and institutional_top10_pct is not None:
            if institutions_percent_held >= 0.7 and institutional_top10_pct >= 0.3:
                crowding_risk = "elevated"
            elif institutions_percent_held >= 0.5:
                crowding_risk = "normal"
            else:
                crowding_risk = "low"

        latest_report = None
        if top_inst:
            latest_report = top_inst[0].get("Date Reported")
        elif top_mf:
            latest_report = top_mf[0].get("Date Reported")

        data = {
            "status": "ok" if top_inst or top_mf or major else "insufficient_data",
            "ownership_source": "yfinance_quote_summary",
            "major_holders": major,
            "institutional_holders": top_inst,
            "mutualfund_holders": top_mf,
            "recent_insider_transactions": insider_transactions[:10],
            "insider_purchases_summary": insider_purchases,
            "institutions_percent_held": institutions_percent_held,
            "insiders_percent_held": insiders_percent_held,
            "institutions_float_percent_held": institutions_float_percent_held,
            "institutions_count": institutions_count,
            "institutional_top10_pct": institutional_top10_pct,
            "mutualfund_top10_pct": mutualfund_top10_pct,
            "top_holder_pct": top_holder_pct,
            "institutional_hhi_top10": hhi_top10,
            "institutional_concentration_level": concentration_level,
            "crowding_risk": crowding_risk,
            "insider_net_activity": insider_activity,
            "insider_net_shares_6m": net_shares,
            "insider_purchase_transactions_6m": _safe_num(insider_purchases.get("Purchases Trans")),
            "insider_sale_transactions_6m": _safe_num(insider_purchases.get("Sales Trans")),
            "audit_risk_yahoo": _clean_value(info.get("auditRisk")),
            "board_risk_yahoo": _clean_value(info.get("boardRisk")),
            "compensation_risk_yahoo": _clean_value(info.get("compensationRisk")),
            "shareholder_rights_risk_yahoo": _clean_value(info.get("shareHolderRightsRisk")),
            "incremental_buyer_seller_map": {
                "buyers": buyers,
                "sellers": sellers,
            },
            "ownership_report_date": _clean_value(latest_report),
        }
        return data

    def _fetch_live_fundamental_history(self, ticker: str) -> dict:
        import yfinance as yf  # type: ignore

        t = yf.Ticker(ticker)
        qis = getattr(t, "quarterly_income_stmt", None)
        qcf = getattr(t, "quarterly_cashflow", None)
        qbs = getattr(t, "quarterly_balance_sheet", None)
        if qis is None or qcf is None or qbs is None or getattr(qis, "empty", True):
            raise ValueError("quarterly financial statements unavailable")

        price_df = yf.download(ticker, period="3y", interval="1d", progress=False, auto_adjust=False)
        if price_df is None or price_df.empty:
            raise ValueError("price history unavailable")
        close = price_df["Close"]
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        close = close.dropna()

        revenue_row = _row_series(qis, ["Total Revenue", "Operating Revenue", "Revenue"])
        op_income_row = _row_series(qis, ["Operating Income", "Total Operating Income As Reported", "EBIT"])
        eps_row = _row_series(qis, ["Diluted EPS", "Basic EPS"])
        fcf_row = _row_series(qcf, ["Free Cash Flow"])
        shares_row = _row_series(qbs, ["Ordinary Shares Number", "Share Issued"])
        if shares_row is None:
            shares_row = _row_series(qis, ["Diluted Average Shares", "Basic Average Shares"])

        quarter_dates = _sorted_common_dates(revenue_row, op_income_row, eps_row, shares_row)
        if len(quarter_dates) < 4:
            raise ValueError("insufficient quarterly history")

        quarterly_history: list[dict] = []
        valuation_points: list[dict] = []
        for idx, qdate in enumerate(quarter_dates):
            revenue = _series_value(revenue_row, qdate)
            op_income = _series_value(op_income_row, qdate)
            eps = _series_value(eps_row, qdate)
            shares = _series_value(shares_row, qdate)
            fcf = _series_value(fcf_row, qdate)
            price = _price_at_or_before(close, qdate)
            operating_margin = None
            if revenue not in (None, 0) and op_income is not None:
                operating_margin = round(op_income / revenue * 100, 2)
            quarterly_history.append(
                {
                    "date": qdate.isoformat(),
                    "revenue": revenue,
                    "operating_income": op_income,
                    "operating_margin_pct": operating_margin,
                    "diluted_eps": eps,
                    "free_cash_flow": fcf,
                    "shares_outstanding": shares,
                    "quarter_end_price": price,
                }
            )
            if idx >= 3:
                window = quarter_dates[idx - 3:idx + 1]
                ttm_revenue = _sum_series(revenue_row, window)
                ttm_op_income = _sum_series(op_income_row, window)
                ttm_eps = _sum_series(eps_row, window)
                ttm_fcf = _sum_series(fcf_row, window)
                if ttm_revenue not in (None, 0) and ttm_op_income is not None:
                    ttm_margin = round(ttm_op_income / ttm_revenue * 100, 2)
                else:
                    ttm_margin = None
                pe_ratio = None
                ps_ratio = None
                fcf_yield_pct = None
                if price not in (None, 0) and ttm_eps not in (None, 0):
                    pe_ratio = round(price / ttm_eps, 2)
                if price not in (None, 0) and ttm_revenue not in (None, 0) and shares not in (None, 0):
                    revenue_per_share = ttm_revenue / shares
                    if revenue_per_share not in (None, 0):
                        ps_ratio = round(price / revenue_per_share, 2)
                if price not in (None, 0) and shares not in (None, 0) and ttm_fcf not in (None, 0):
                    market_cap = price * shares
                    if market_cap not in (None, 0):
                        fcf_yield_pct = round(ttm_fcf / market_cap * 100, 2)
                valuation_points.append(
                    {
                        "date": qdate.isoformat(),
                        "quarter_end_price": price,
                        "ttm_revenue": ttm_revenue,
                        "ttm_operating_income": ttm_op_income,
                        "ttm_operating_margin_pct": ttm_margin,
                        "ttm_eps": round(ttm_eps, 4) if ttm_eps is not None else None,
                        "ttm_fcf": ttm_fcf,
                        "shares_outstanding": shares,
                        "pe_ratio": pe_ratio,
                        "ps_ratio": ps_ratio,
                        "fcf_yield_pct": fcf_yield_pct,
                    }
                )

        latest_point = valuation_points[-1] if valuation_points else {}
        data = {
            "fundamental_history_source": "yfinance_financials",
            "quarterly_history_real": quarterly_history[-8:],
            "valuation_history_real": {
                "status": "ok" if valuation_points else "insufficient_data",
                "source": "quarterly_statements_plus_quarter_end_prices",
                "valuation_points": valuation_points[-8:],
                "pe_ratios": [x["pe_ratio"] for x in valuation_points if x.get("pe_ratio") not in (None, 0)],
                "ps_ratios": [x["ps_ratio"] for x in valuation_points if x.get("ps_ratio") not in (None, 0)],
                "fcf_yield_pcts": [x["fcf_yield_pct"] for x in valuation_points if x.get("fcf_yield_pct") is not None],
            },
            "ttm_revenue_real": latest_point.get("ttm_revenue"),
            "ttm_operating_income_real": latest_point.get("ttm_operating_income"),
            "ttm_operating_margin_real": latest_point.get("ttm_operating_margin_pct"),
            "ttm_fcf_real": latest_point.get("ttm_fcf"),
            "ttm_fcf_margin_real": (
                round(latest_point["ttm_fcf"] / latest_point["ttm_revenue"] * 100, 2)
                if latest_point.get("ttm_fcf") not in (None, 0) and latest_point.get("ttm_revenue") not in (None, 0)
                else None
            ),
            "eps_ttm_real": latest_point.get("ttm_eps"),
            "shares_outstanding_real": latest_point.get("shares_outstanding"),
        }
        return data

    @staticmethod
    def _mock_estimate_revision(ticker: str, as_of: str) -> dict:
        rng = random.Random(hash((ticker, "estimate")) % (2**31))
        def _period(cur_eps: float, cur_rev: float) -> dict:
            rev30 = round(rng.uniform(-3, 6), 2)
            rev90 = round(rev30 + rng.uniform(-2, 2), 2)
            return {
                "eps_estimate": round(cur_eps, 4),
                "revenue_estimate": round(cur_rev, 2),
                "eps_revision_30d_pct": rev30,
                "eps_revision_90d_pct": rev90,
                "up_last_30d": rng.randint(0, 15),
                "down_last_30d": rng.randint(0, 10),
                "revision_state": _revision_state(rev30, rev90, 0, 0),
            }
        return {
            "status": "ok",
            "estimate_source": "mock",
            "estimate_periods": {
                "0q": _period(rng.uniform(1.0, 3.5), rng.uniform(5e10, 1.2e11)),
                "0y": _period(rng.uniform(6.0, 12.0), rng.uniform(2e11, 5e11)),
                "+1y": _period(rng.uniform(6.5, 13.0), rng.uniform(2.1e11, 5.3e11)),
            },
            "earnings_estimate_table": {},
            "revenue_estimate_table": {},
            "eps_trend_table": {},
            "eps_revisions_table": {},
            "growth_estimates_table": {},
            "recommendations_summary": {},
            "analyst_price_targets_yahoo": {
                "current": 100.0,
                "mean": 112.0,
                "median": 110.0,
                "high": 125.0,
                "low": 88.0,
            },
            "next_earnings_date_yahoo": "2026-05-01",
        }

    @staticmethod
    def _mock_ownership_snapshot(ticker: str, as_of: str) -> dict:
        rng = random.Random(hash((ticker, "ownership")) % (2**31))
        holders = []
        total = 0.0
        for idx, name in enumerate(["Vanguard", "BlackRock", "State Street", "FMR", "Geode"]):
            pct = round(max(0.01, rng.uniform(0.015, 0.09)), 4)
            total += pct
            holders.append(
                {
                    "Date Reported": "2025-12-31",
                    "Holder": name,
                    "pctHeld": pct,
                    "Shares": int(rng.uniform(1e7, 5e8)),
                    "Value": int(rng.uniform(1e9, 1e11)),
                    "pctChange": round(rng.uniform(-0.05, 0.05), 4),
                }
            )
        return {
            "status": "ok",
            "ownership_source": "mock",
            "major_holders": {
                "insidersPercentHeld": 0.012,
                "institutionsPercentHeld": 0.64,
                "institutionsFloatPercentHeld": 0.66,
                "institutionsCount": 3200.0,
            },
            "institutional_holders": holders,
            "mutualfund_holders": holders[:3],
            "recent_insider_transactions": [],
            "insider_purchases_summary": {
                "Purchases Shares": 100000.0,
                "Sales Shares": 80000.0,
                "Net Shares Purchased (Sold)": 20000.0,
                "Purchases Trans": 4.0,
                "Sales Trans": 2.0,
            },
            "institutions_percent_held": 0.64,
            "insiders_percent_held": 0.012,
            "institutions_float_percent_held": 0.66,
            "institutions_count": 3200.0,
            "institutional_top10_pct": round(total, 4),
            "mutualfund_top10_pct": round(total * 0.6, 4),
            "top_holder_pct": holders[0]["pctHeld"],
            "institutional_hhi_top10": round(sum(float(h["pctHeld"]) ** 2 for h in holders), 6),
            "institutional_concentration_level": "medium",
            "crowding_risk": "normal",
            "insider_net_activity": "buying",
            "insider_net_shares_6m": 20000.0,
            "insider_purchase_transactions_6m": 4.0,
            "insider_sale_transactions_6m": 2.0,
            "audit_risk_yahoo": 2,
            "board_risk_yahoo": 1,
            "compensation_risk_yahoo": 6,
            "shareholder_rights_risk_yahoo": 2,
            "incremental_buyer_seller_map": {
                "buyers": [
                    {"holder": holders[0]["Holder"], "pct_change": 0.03, "pct_held": holders[0]["pctHeld"], "date_reported": "2025-12-31"},
                ],
                "sellers": [
                    {"holder": holders[-1]["Holder"], "pct_change": -0.02, "pct_held": holders[-1]["pctHeld"], "date_reported": "2025-12-31"},
                ],
            },
            "ownership_report_date": "2025-12-31",
        }

    @staticmethod
    def _mock_fundamental_history_snapshot(ticker: str, as_of: str) -> dict:
        rng = random.Random(hash((ticker, "history")) % (2**31))
        points = []
        base_price = rng.uniform(80, 140)
        base_eps = rng.uniform(4.0, 8.0)
        base_rev = rng.uniform(2e11, 5e11)
        base_margin = rng.uniform(18, 32)
        base_fcf_yield = rng.uniform(2, 6)
        dates = ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31", "2025-03-31", "2025-06-30", "2025-09-30"]
        for idx, d in enumerate(dates):
            points.append(
                {
                    "date": d,
                    "quarter_end_price": round(base_price * (1 + 0.03 * idx), 2),
                    "ttm_revenue": round(base_rev * (1 + 0.02 * idx), 2),
                    "ttm_operating_income": round(base_rev * (1 + 0.02 * idx) * (base_margin / 100), 2),
                    "ttm_operating_margin_pct": round(base_margin + 0.2 * idx, 2),
                    "ttm_eps": round(base_eps * (1 + 0.03 * idx), 4),
                    "ttm_fcf": round(base_rev * (1 + 0.02 * idx) * 0.22, 2),
                    "shares_outstanding": 1.5e10,
                    "pe_ratio": round((base_price * (1 + 0.03 * idx)) / (base_eps * (1 + 0.03 * idx)), 2),
                    "ps_ratio": round(6.5 + 0.15 * idx, 2),
                    "fcf_yield_pct": round(base_fcf_yield - 0.1 * idx, 2),
                }
            )
        latest = points[-1]
        return {
            "fundamental_history_source": "mock",
            "quarterly_history_real": [],
            "valuation_history_real": {
                "status": "ok",
                "source": "mock",
                "valuation_points": points,
                "pe_ratios": [p["pe_ratio"] for p in points],
                "ps_ratios": [p["ps_ratio"] for p in points],
                "fcf_yield_pcts": [p["fcf_yield_pct"] for p in points],
            },
            "ttm_revenue_real": latest["ttm_revenue"],
            "ttm_operating_income_real": latest["ttm_operating_income"],
            "ttm_operating_margin_real": latest["ttm_operating_margin_pct"],
            "ttm_fcf_real": latest["ttm_fcf"],
            "ttm_fcf_margin_real": 22.0,
            "eps_ttm_real": latest["ttm_eps"],
            "shares_outstanding_real": latest["shares_outstanding"],
        }

    @staticmethod
    def _build_estimate_evidence(data: dict, as_of: str, source_name: str = "yfinance", quality: float = 0.78) -> List[dict]:
        periods = data.get("estimate_periods") or {}
        evidence: List[dict] = []
        p0q = periods.get("0q") or {}
        p0y = periods.get("0y") or {}
        if p0q.get("eps_revision_30d_pct") is not None:
            evidence.append(make_evidence(metric="eps_revision_30d_pct_0q", value=p0q.get("eps_revision_30d_pct"), source_name=source_name, source_type="api", quality=quality, as_of=as_of))
        if p0q.get("eps_revision_90d_pct") is not None:
            evidence.append(make_evidence(metric="eps_revision_90d_pct_0q", value=p0q.get("eps_revision_90d_pct"), source_name=source_name, source_type="api", quality=quality, as_of=as_of))
        if p0y.get("eps_revision_30d_pct") is not None:
            evidence.append(make_evidence(metric="eps_revision_30d_pct_0y", value=p0y.get("eps_revision_30d_pct"), source_name=source_name, source_type="api", quality=quality, as_of=as_of))
        targets = data.get("analyst_price_targets_yahoo") or {}
        if targets.get("mean") is not None:
            evidence.append(make_evidence(metric="price_target_consensus_yahoo", value=targets.get("mean"), source_name=source_name, source_type="api", quality=quality, as_of=as_of))
        return evidence

    @staticmethod
    def _build_ownership_evidence(data: dict, as_of: str, source_name: str = "yfinance", quality: float = 0.78) -> List[dict]:
        evidence: List[dict] = []
        for metric in (
            "institutions_percent_held",
            "insiders_percent_held",
            "institutional_top10_pct",
            "top_holder_pct",
            "insider_net_shares_6m",
        ):
            value = data.get(metric)
            if value is not None:
                evidence.append(make_evidence(metric=metric, value=value, source_name=source_name, source_type="api", quality=quality, as_of=str(data.get("ownership_report_date") or as_of)))
        return evidence

    @staticmethod
    def _build_history_evidence(data: dict, as_of: str, source_name: str = "yfinance", quality: float = 0.78) -> List[dict]:
        evidence: List[dict] = []
        for metric in (
            "ttm_revenue_real",
            "ttm_operating_margin_real",
            "ttm_fcf_real",
            "eps_ttm_real",
        ):
            value = data.get(metric)
            if value is not None:
                evidence.append(make_evidence(metric=metric, value=value, source_name=source_name, source_type="api", quality=quality, as_of=as_of))
        return evidence


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _safe_num(value: Any) -> float | None:
    value = _clean_value(value)
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _table_records(df: Any) -> list[dict]:
    if df is None or not hasattr(df, "to_dict"):
        return []
    rows = df.to_dict("records")
    return [{k: _clean_value(v) for k, v in row.items()} for row in rows]


def _table_records_by_index(df: Any) -> dict[str, dict]:
    if df is None or not hasattr(df, "iterrows"):
        return {}
    out: dict[str, dict] = {}
    for idx, row in df.iterrows():
        out[str(idx)] = {str(k): _clean_value(v) for k, v in row.items()}
    return out


def _table_records_by_period(df: Any) -> dict[str, dict]:
    rows = _table_records(df)
    out: dict[str, dict] = {}
    for row in rows:
        period = str(row.get("period", "")).strip()
        if period:
            out[period] = row
    return out


def _major_holders_dict(df: Any) -> dict[str, Any]:
    if df is None or not hasattr(df, "iterrows"):
        return {}
    out: dict[str, Any] = {}
    for idx, row in df.iterrows():
        try:
            out[str(idx)] = _clean_value(row["Value"])
        except Exception:
            continue
    return out


def _insider_purchases_dict(df: Any) -> dict[str, Any]:
    if df is None or not hasattr(df, "iterrows"):
        return {}
    out: dict[str, Any] = {}
    for _, row in df.iterrows():
        label = str(row.get("Insider Purchases Last 6m", "")).strip()
        if not label:
            continue
        out[f"{label} Shares"] = _clean_value(row.get("Shares"))
        out[f"{label} Trans"] = _clean_value(row.get("Trans"))
        if label == "Net Shares Purchased (Sold)":
            out[label] = _clean_value(row.get("Shares"))
    return out


def _field(table: dict[str, dict], period: str, field: str) -> Any:
    row = table.get(period) or {}
    return _clean_value(row.get(field))


def _revision_pct(table: dict[str, dict], period: str, old_key: str) -> float | None:
    row = table.get(period) or {}
    current = _safe_num(row.get("current"))
    old = _safe_num(row.get(old_key))
    if current in (None, 0) or old in (None, 0):
        return None
    return round((current / old - 1) * 100, 2)


def _revision_state(rev30: Any, rev90: Any, up_30d: Any, down_30d: Any) -> str:
    r30 = _safe_num(rev30) or 0.0
    r90 = _safe_num(rev90) or 0.0
    up = _safe_num(up_30d) or 0.0
    down = _safe_num(down_30d) or 0.0
    score = r30 + 0.5 * r90 + 0.2 * (up - down)
    if score >= 2.0:
        return "improving"
    if score <= -2.0:
        return "deteriorating"
    return "stable"


def _row_series(df: Any, candidates: list[str]):
    if df is None or not hasattr(df, "index"):
        return None
    for name in candidates:
        if name in df.index:
            return df.loc[name]
    return None


def _sorted_common_dates(*series_list) -> list[date]:
    dates = None
    for ser in series_list:
        if ser is None:
            continue
        idx = set()
        for raw in getattr(ser, "index", []):
            cleaned = _clean_value(raw)
            if isinstance(cleaned, str):
                idx.add(date.fromisoformat(cleaned[:10]))
        dates = idx if dates is None else dates & idx
    return sorted(dates or [])


def _series_value(series: Any, dt: date) -> float | None:
    if series is None:
        return None
    for raw in getattr(series, "index", []):
        cleaned = _clean_value(raw)
        if isinstance(cleaned, str) and cleaned[:10] == dt.isoformat():
            return _safe_num(series[raw])
    return None


def _sum_series(series: Any, dates: list[date]) -> float | None:
    vals = [_series_value(series, dt) for dt in dates]
    if any(v is None for v in vals):
        return None
    return round(sum(v for v in vals if v is not None), 4)


def _price_at_or_before(close_series: Any, dt: date) -> float | None:
    if close_series is None or not hasattr(close_series, "index"):
        return None
    eligible = close_series.loc[close_series.index.date <= dt]
    if getattr(eligible, "empty", True):
        return None
    try:
        return round(float(eligible.iloc[-1]), 4)
    except Exception:
        return None
