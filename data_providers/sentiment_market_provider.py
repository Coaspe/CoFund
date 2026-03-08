"""
data_providers/sentiment_market_provider.py
==========================================
Structured market sentiment / options / volatility snapshot.

Purpose:
  - Upgrade Sentiment Analyst from pure news mood to hedge-fund style
    positioning / options / vol structure overlay.
  - Use live, structured sources where available:
      * yfinance VIX family indices
      * yfinance option chain open interest / volume
      * yfinance quote summary short-interest fields
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from data_providers.base import BaseProvider
from schemas.common import make_evidence


class SentimentMarketProvider(BaseProvider):
    PROVIDER_NAME = "sentiment_market"

    _VOL_SYMBOLS = {
        "vix_level": "^VIX",
        "vvix_level": "^VVIX",
        "skew_index": "^SKEW",
        "vix_9d": "^VIX9D",
        "vix_3m": "^VIX3M",
    }

    def __init__(self, *, mode: str = "mock", **kwargs):
        self.mode = mode
        super().__init__(**kwargs)

    def get_snapshot(self, ticker: str, as_of: str = "") -> dict:
        as_of = as_of or datetime.now(timezone.utc).isoformat()
        if self.mode == "mock":
            data = self._mock_snapshot(ticker)
            return {
                "data": data,
                "evidence": self._build_evidence(data, as_of, source_name="mock", quality=0.3),
                "data_ok": False,
                "limitations": ["Mock mode — structured sentiment market snapshot"],
                "as_of": as_of,
            }

        try:
            data = self._fetch_live_snapshot(ticker, as_of)
            return {
                "data": data,
                "evidence": self._build_evidence(data, as_of, source_name="yfinance", quality=0.78),
                "data_ok": True,
                "limitations": [],
                "as_of": as_of,
            }
        except Exception as exc:
            data = self._mock_snapshot(ticker)
            return {
                "data": data,
                "evidence": self._build_evidence(data, as_of, source_name="mock", quality=0.3),
                "data_ok": False,
                "limitations": [f"sentiment_market live error: {exc}", "Using mock structured sentiment snapshot"],
                "as_of": as_of,
            }

    def _fetch_live_snapshot(self, ticker: str, as_of: str) -> dict:
        import yfinance as yf  # type: ignore

        data: dict[str, Any] = {}

        for metric, symbol in self._VOL_SYMBOLS.items():
            df = yf.download(symbol, period="1mo", interval="1d", progress=False, auto_adjust=False)
            if df is None or df.empty:
                continue
            close = df["Close"]
            if hasattr(close, "columns"):
                close = close.iloc[:, 0]
            close = close.dropna()
            if close.empty:
                continue
            data[metric] = round(float(close.iloc[-1]), 4)
            last_ts = close.index[-1]
            ts = getattr(last_ts, "to_pydatetime", lambda: last_ts)()
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                else:
                    ts = ts.astimezone(timezone.utc)
                data[f"{metric}_as_of"] = ts.isoformat()

        v9d = self._safe_float(data.get("vix_9d"))
        vix = self._safe_float(data.get("vix_level"))
        v3m = self._safe_float(data.get("vix_3m"))
        if v9d is not None and vix is not None and v3m is not None:
            data["vix_term_structure_slope_9d_3m"] = round(v3m - v9d, 4)
            if v9d > vix or vix > v3m:
                data["vix_term_structure"] = "backwardation"
            elif v3m > vix and vix >= v9d:
                data["vix_term_structure"] = "contango"
            else:
                data["vix_term_structure"] = "flat"
        elif vix is not None and v3m is not None:
            data["vix_term_structure_slope_1m_3m"] = round(v3m - vix, 4)
            data["vix_term_structure"] = "backwardation" if vix > v3m else "contango"

        t = yf.Ticker(ticker)
        info = getattr(t, "info", {}) or {}
        short_pct = self._safe_float(info.get("shortPercentOfFloat"))
        if short_pct is None:
            short_pct = self._safe_float(info.get("sharesPercentSharesOut"))
        if short_pct is not None:
            data["short_interest_pct"] = round(short_pct * 100, 3)
        shares_short = self._safe_float(info.get("sharesShort"))
        shares_short_prior = self._safe_float(info.get("sharesShortPriorMonth"))
        if shares_short is not None:
            data["shares_short"] = round(shares_short, 0)
        if shares_short_prior is not None:
            data["shares_short_prior_month"] = round(shares_short_prior, 0)
        if shares_short is not None and shares_short_prior not in (None, 0):
            data["short_interest_change_pct"] = round((shares_short - shares_short_prior) / shares_short_prior * 100, 2)
        held_inst = self._safe_float(info.get("heldPercentInstitutions"))
        if held_inst is not None:
            data["held_percent_institutions"] = round(held_inst * 100, 2)
        held_insiders = self._safe_float(info.get("heldPercentInsiders"))
        if held_insiders is not None:
            data["held_percent_insiders"] = round(held_insiders * 100, 2)

        expiries = list(getattr(t, "options", []) or [])
        if expiries:
            expiry = expiries[0]
            chain = t.option_chain(expiry)
            puts = getattr(chain, "puts", None)
            calls = getattr(chain, "calls", None)
            if puts is not None and calls is not None and not puts.empty and not calls.empty:
                put_oi = float(puts["openInterest"].fillna(0).sum()) if "openInterest" in puts else 0.0
                call_oi = float(calls["openInterest"].fillna(0).sum()) if "openInterest" in calls else 0.0
                put_vol = float(puts["volume"].fillna(0).sum()) if "volume" in puts else 0.0
                call_vol = float(calls["volume"].fillna(0).sum()) if "volume" in calls else 0.0
                if call_oi > 0:
                    data["put_call_oi_ratio"] = round(put_oi / call_oi, 4)
                    data["put_call_ratio"] = data["put_call_oi_ratio"]
                if call_vol > 0:
                    data["put_call_volume_ratio"] = round(put_vol / call_vol, 4)
                data["options_expiry_near"] = expiry
                data["options_put_oi"] = round(put_oi, 0)
                data["options_call_oi"] = round(call_oi, 0)

        if not data:
            raise ValueError("sentiment market snapshot unavailable")
        return data

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            if value in (None, ""):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _build_evidence(cls, data: dict, as_of: str, *, source_name: str, quality: float) -> list[dict]:
        evidence = []
        metrics = [
            "vix_level",
            "vvix_level",
            "skew_index",
            "vix_9d",
            "vix_3m",
            "put_call_oi_ratio",
            "put_call_volume_ratio",
            "short_interest_pct",
            "short_interest_change_pct",
            "held_percent_institutions",
            "held_percent_insiders",
        ]
        for metric in metrics:
            if data.get(metric) is None:
                continue
            metric_as_of = str(data.get(f"{metric}_as_of") or as_of)
            evidence.append(
                make_evidence(
                    metric=metric,
                    value=data[metric],
                    source_name=source_name,
                    source_type="api" if source_name != "mock" else "model",
                    quality=quality,
                    as_of=metric_as_of,
                )
            )
        return evidence

    @staticmethod
    def _mock_snapshot(ticker: str) -> dict:
        base = {
            "vix_level": 20.5,
            "vvix_level": 94.0,
            "skew_index": 136.0,
            "vix_9d": 19.2,
            "vix_3m": 22.1,
            "vix_term_structure": "contango",
            "vix_term_structure_slope_9d_3m": 2.9,
            "put_call_oi_ratio": 0.98,
            "put_call_volume_ratio": 1.04,
            "short_interest_pct": 4.2,
            "short_interest_change_pct": 3.5,
            "held_percent_institutions": 61.0,
            "held_percent_insiders": 2.5,
            "options_expiry_near": "2026-03-13",
            "options_put_oi": 1_250_000.0,
            "options_call_oi": 1_275_000.0,
        }
        if str(ticker).strip().upper() in {"QQQ", "XLK"}:
            base["put_call_oi_ratio"] = 0.85
            base["short_interest_pct"] = 2.6
        elif str(ticker).strip().upper() in {"GLD", "TLT"}:
            base["put_call_oi_ratio"] = 1.12
            base["short_interest_pct"] = 1.4
        return base
