"""
data_providers/data_hub.py — Central DataHub aggregator
=========================================================
Coordinates all providers with shared run_id, as_of, caching.
Single entry point for agents to fetch data.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np

from schemas.common import make_evidence
from data_providers.cache import DiskCache
from data_providers.rate_limiter import RateLimiter
from data_providers.fred_provider import FREDProvider
from data_providers.fmp_provider import FMPProvider
from data_providers.sec_edgar_provider import SECEdgarProvider
from data_providers.newsapi_provider import NewsAPIProvider
from data_providers.alphavantage_provider import AlphaVantageProvider
from data_providers.twelvedata_provider import TwelveDataProvider
from data_providers.fed_funds_futures_provider import FedFundsFuturesProvider
from data_providers.sofr_futures_provider import SofrFuturesProvider
from data_providers.macro_event_calendar_provider import MacroEventCalendarProvider
from data_providers.yahoo_structured_provider import YahooStructuredProvider
from data_providers.ir_press_release_provider import IRPressReleaseProvider
from data_providers.sentiment_market_provider import SentimentMarketProvider
from data_providers.base import is_rate_limit_error

# Legacy mock wrappers
from data_providers.fred_provider import (
    fetch_macro_indicators as _legacy_macro,
    fetch_sentiment_indicators as _legacy_sentiment,
    fetch_fundamentals as _legacy_fundamentals,
)
from data_providers.market_data_provider import fetch_prices as _legacy_prices
from data_providers.market_data_provider import fetch_macro_market_indicators as _legacy_macro_market


class DataHub:
    """
    Central data coordinator for all agents.

    In 'mock' mode, delegates to legacy mock functions for backward compat.
    In 'live' mode, uses real API providers with caching + rate limiting.
    """

    def __init__(self, *, run_id: str = "", as_of: str = "", mode: str = "mock"):
        self.run_id = run_id
        self.as_of = as_of or datetime.now(timezone.utc).isoformat()
        self.mode = mode

        # Shared infra
        self._cache = DiskCache()
        self._limiter = RateLimiter()

        # Initialize providers (lazy — only used in live mode)
        if mode == "live":
            self._fred = FREDProvider(cache=self._cache, rate_limiter=self._limiter)
            self._fmp = FMPProvider(cache=self._cache, rate_limiter=self._limiter)
            self._sec = SECEdgarProvider(cache=self._cache, rate_limiter=self._limiter)
            self._news = NewsAPIProvider(cache=self._cache, rate_limiter=self._limiter)
            self._av = AlphaVantageProvider(cache=self._cache, rate_limiter=self._limiter)
            self._td = TwelveDataProvider(cache=self._cache, rate_limiter=self._limiter)
            self._fff = FedFundsFuturesProvider(cache=self._cache, rate_limiter=self._limiter)
            self._sfr = SofrFuturesProvider(cache=self._cache, rate_limiter=self._limiter)
            self._mec = MacroEventCalendarProvider(cache=self._cache, rate_limiter=self._limiter)
            self._yfs = YahooStructuredProvider(cache=self._cache, rate_limiter=self._limiter)
            self._irpr = IRPressReleaseProvider(mode=mode, cache=self._cache, rate_limiter=self._limiter)
            self._smp = SentimentMarketProvider(mode=mode, cache=self._cache, rate_limiter=self._limiter)

    # ── Macro ─────────────────────────────────────────────────────────────

    def get_macro_indicators(self, ticker: str = "", horizon_days: int = 30,
                             seed: int | None = None) -> Tuple[dict, list, dict]:
        """Returns (macro_indicators, evidence, meta)."""
        if self.mode == "mock":
            data, evidence = _legacy_macro(mode="mock", as_of=self.as_of, seed=seed)
            return data, evidence, {"data_ok": True, "limitations": ["Mock data"]}

        snapshot = self._fred.get_macro_snapshot(as_of=self.as_of)
        market_data, market_evidence, market_meta = _legacy_macro_market(mode="live", as_of=self.as_of, seed=seed)
        futures_snapshot = self._fff.get_curve(as_of=self.as_of)
        sofr_snapshot = self._sfr.get_curve(as_of=self.as_of)
        merged = dict(snapshot["data"])
        for key, value in (market_data or {}).items():
            if key not in merged or merged.get(key) is None:
                merged[key] = value
            else:
                merged[f"{key}_market"] = value
        for key, value in (futures_snapshot.get("data", {}) or {}).items():
            if key not in merged or merged.get(key) is None:
                merged[key] = value
        for key, value in (sofr_snapshot.get("data", {}) or {}).items():
            if key not in merged or merged.get(key) is None:
                merged[key] = value
        ff_6m = merged.get("fed_funds_futures_6m_implied_rate")
        sofr_6m = merged.get("sofr_futures_6m_implied_rate")
        basis_as_of = self.as_of
        ff_curve = (futures_snapshot.get("data", {}) or {}).get("fed_funds_futures_curve", [])
        sofr_curve = (sofr_snapshot.get("data", {}) or {}).get("sofr_futures_curve", [])
        if isinstance(sofr_curve, list) and sofr_curve:
            basis_as_of = str(sofr_curve[min(2, len(sofr_curve) - 1)].get("as_of") or basis_as_of)
        elif isinstance(ff_curve, list) and ff_curve:
            basis_as_of = str(ff_curve[min(5, len(ff_curve) - 1)].get("as_of") or basis_as_of)
        if ff_6m is not None and sofr_6m is not None:
            merged["sofr_ff_6m_basis_bp"] = round((sofr_6m - ff_6m) * 100, 1)
        limitations = (
            list(snapshot["limitations"])
            + list((market_meta or {}).get("limitations", []))
            + list((futures_snapshot or {}).get("limitations", []))
            + list((sofr_snapshot or {}).get("limitations", []))
        )
        evidence = list(snapshot["evidence"]) + list(market_evidence) + list(futures_snapshot.get("evidence", [])) + list(sofr_snapshot.get("evidence", []))
        if ff_6m is not None and sofr_6m is not None:
            evidence.append(
                make_evidence(
                    metric="sofr_ff_6m_basis_bp",
                    value=merged["sofr_ff_6m_basis_bp"],
                    source_name="derived:SOFR-FF",
                    source_type="model",
                    quality=0.78,
                    as_of=basis_as_of,
                    note="SOFR 6m implied minus Fed funds 6m implied, in basis points",
                )
            )
        return merged, evidence, {
            "data_ok": bool(snapshot["data_ok"] or (market_meta or {}).get("data_ok") or futures_snapshot.get("data_ok") or sofr_snapshot.get("data_ok")),
            "limitations": limitations,
        }

    def get_macro_event_calendar(self) -> Tuple[list, list, dict]:
        """Returns structured macro event calendar from official sources where accessible."""
        if self.mode == "mock":
            result = MacroEventCalendarProvider._mock_calendar(self.as_of)
            items = result.get("items", [])
            return items, result.get("evidence", []), {
                "data_ok": result["data_ok"],
                "limitations": result["limitations"],
            }
        result = self._mec.get_calendar(as_of=self.as_of)
        items = result.get("items", [])
        return items, result.get("evidence", []), {
            "data_ok": result["data_ok"],
            "limitations": result["limitations"],
        }

    # ── Fundamentals ──────────────────────────────────────────────────────

    def get_fundamentals(self, ticker: str, seed: int | None = None) -> Tuple[dict, list, dict]:
        """Returns (financials, evidence, meta)."""
        if self.mode == "mock":
            data, evidence = _legacy_fundamentals(ticker, mode="mock", as_of=self.as_of, seed=seed)
            return data, evidence, {"data_ok": True, "limitations": ["Mock data"]}

        result = self._fmp.get_fundamentals(ticker, as_of=self.as_of)
        return result["data"], result["evidence"], {
            "data_ok": result["data_ok"], "limitations": result["limitations"],
        }

    def get_sec_flags(self, ticker: str) -> Tuple[dict | None, list, dict]:
        """Returns (sec_data, evidence, meta)."""
        if self.mode == "mock":
            return None, [], {"data_ok": False, "limitations": ["Mock mode — no SEC data"]}

        result = self._sec.get_sec_flags(ticker, as_of=self.as_of)
        return result["data"], result["evidence"], {
            "data_ok": result["data_ok"], "limitations": result["limitations"],
        }

    def get_peer_context(self, ticker: str, max_peers: int = 5) -> Tuple[dict, list, dict]:
        """Returns (peer_context, evidence, meta)."""
        if self.mode == "mock":
            result = FMPProvider._mock_peer_context(ticker, self.as_of, max_peers=max_peers)
            return result["data"], result["evidence"], {
                "data_ok": result["data_ok"], "limitations": result["limitations"],
            }
        result = self._fmp.get_peer_context(ticker, as_of=self.as_of, max_peers=max_peers)
        return result["data"], result["evidence"], {
            "data_ok": result["data_ok"], "limitations": result["limitations"],
        }

    def get_ownership_identity(self, ticker: str) -> Tuple[list, list, dict]:
        """Returns (items, evidence, meta)."""
        if self.mode == "mock":
            return [], [], {"data_ok": False, "limitations": ["Mock mode — no ownership data"]}
        result = self._sec.get_ownership_identity(ticker, as_of=self.as_of)
        items = result.get("items", [])
        return items, items, {
            "data_ok": result["data_ok"], "limitations": result["limitations"],
        }

    def get_8k_exhibits(self, ticker: str) -> Tuple[list, list, dict]:
        """Returns recent 8-K evidence items for catalyst/IR detection."""
        if self.mode == "mock":
            return [], [], {"data_ok": False, "limitations": ["Mock mode — no 8-K data"]}
        result = self._sec.get_8k_exhibits(ticker, as_of=self.as_of)
        items = result.get("items", [])
        return items, items, {
            "data_ok": result["data_ok"], "limitations": result["limitations"],
        }

    def get_ir_press_release_events(self, ticker: str) -> Tuple[list, list, dict]:
        """Returns structured investor day / product launch events from wire/IR releases."""
        if self.mode == "mock":
            result = IRPressReleaseProvider(mode="mock").get_catalyst_events(ticker, as_of=self.as_of)
            items = result.get("items", [])
            return items, items, {"data_ok": result["data_ok"], "limitations": result["limitations"]}
        result = self._irpr.get_catalyst_events(ticker, as_of=self.as_of)
        items = result.get("items", [])
        return items, items, {
            "data_ok": result["data_ok"], "limitations": result["limitations"],
        }

    def get_estimate_revision(self, ticker: str) -> Tuple[dict, list, dict]:
        """Returns structured estimate/revision snapshot."""
        if self.mode == "mock":
            result = YahooStructuredProvider._mock_estimate_revision(ticker, self.as_of)
            evidence = YahooStructuredProvider._build_estimate_evidence(result, self.as_of, source_name="mock", quality=0.3)
            return result, evidence, {"data_ok": False, "limitations": ["Mock mode — estimate revision snapshot"]}
        result = self._yfs.get_estimate_revision_snapshot(ticker, as_of=self.as_of)
        return result["data"], result["evidence"], {
            "data_ok": result["data_ok"], "limitations": result["limitations"],
        }

    def get_structured_ownership(self, ticker: str) -> Tuple[dict, list, dict]:
        """Returns structured holder concentration / insider summary."""
        if self.mode == "mock":
            result = YahooStructuredProvider._mock_ownership_snapshot(ticker, self.as_of)
            evidence = YahooStructuredProvider._build_ownership_evidence(result, self.as_of, source_name="mock", quality=0.3)
            return result, evidence, {"data_ok": False, "limitations": ["Mock mode — structured ownership snapshot"]}
        result = self._yfs.get_ownership_snapshot(ticker, as_of=self.as_of)
        return result["data"], result["evidence"], {
            "data_ok": result["data_ok"], "limitations": result["limitations"],
        }

    def get_fundamental_history(self, ticker: str) -> Tuple[dict, list, dict]:
        """Returns structured quarterly history / real valuation history."""
        if self.mode == "mock":
            result = YahooStructuredProvider._mock_fundamental_history_snapshot(ticker, self.as_of)
            evidence = YahooStructuredProvider._build_history_evidence(result, self.as_of, source_name="mock", quality=0.3)
            return result, evidence, {"data_ok": False, "limitations": ["Mock mode — fundamental history snapshot"]}
        result = self._yfs.get_fundamental_history_snapshot(ticker, as_of=self.as_of)
        return result["data"], result["evidence"], {
            "data_ok": result["data_ok"], "limitations": result["limitations"],
        }

    # ── Sentiment ─────────────────────────────────────────────────────────

    def get_news_sentiment(self, ticker: str, days: int = 7,
                           seed: int | None = None) -> Tuple[dict, list, dict]:
        """Returns (sentiment_indicators, evidence, meta)."""
        if self.mode == "mock":
            data, evidence = _legacy_sentiment(ticker, mode="mock", as_of=self.as_of, seed=seed)
            return data, evidence, {"data_ok": True, "limitations": ["Mock data"]}

        # Try Alpha Vantage first (higher quality), fallback to NewsAPI
        av_rate_limited = False
        if hasattr(self, '_av') and self._av.has_key:
            av_result = self._av.get_news_sentiment(ticker, days=days)
            av_rate_limited = any(
                is_rate_limit_error(msg) for msg in av_result.get("limitations", [])
            )
            if av_result["data_ok"] and av_result["data"]:
                indicators = {
                    "news_sentiment_score": av_result["data"]["sentiment_score"],
                    "article_count": av_result["data"]["article_count"],
                    "key_topics": av_result["data"]["key_topics"],
                    "upcoming_events": [],
                }
                return indicators, av_result["evidence"], {
                    "data_ok": True, "limitations": av_result["limitations"],
                }

        # NewsAPI
        result = self._news.search_ticker_news(ticker, days=days)
        news_rate_limited = any(is_rate_limit_error(msg) for msg in result.get("limitations", []))
        if av_rate_limited and news_rate_limited:
            print("   [API Router] ⚠️ 모든 Sentiment API가 rate limit 상태입니다.", flush=True)
        return result["data"], result["evidence"], {
            "data_ok": result["data_ok"], "limitations": result["limitations"],
        }

    def get_sentiment_market_snapshot(self, ticker: str) -> Tuple[dict, list, dict]:
        """Returns structured options/vol/short-interest sentiment snapshot."""
        if self.mode == "mock":
            result = SentimentMarketProvider(mode="mock").get_snapshot(ticker, as_of=self.as_of)
            return result["data"], result["evidence"], {
                "data_ok": result["data_ok"], "limitations": result["limitations"],
            }
        result = self._smp.get_snapshot(ticker, as_of=self.as_of)
        return result["data"], result["evidence"], {
            "data_ok": result["data_ok"], "limitations": result["limitations"],
        }

    # ── Prices (Quant/Risk) ───────────────────────────────────────────────

    def get_price_series(self, ticker: str, lookback_days: int = 504,
                         seed: int | None = None) -> Tuple[np.ndarray, list, dict]:
        """Returns (prices_array, evidence, meta)."""
        if self.mode == "mock":
            prices, evidence = _legacy_prices(ticker, lookback_days, mode="mock",
                                              as_of=self.as_of, seed=seed)
            return prices, evidence, {"data_ok": True, "limitations": ["Mock OU data"]}

        result = self._td.get_price_series(ticker, lookback_days=lookback_days, as_of=self.as_of)
        return result["data"], result["evidence"], {
            "data_ok": result["data_ok"], "limitations": result["limitations"],
        }

    def get_market_series(self, market: str = "SPY", lookback_days: int = 504,
                          seed: int | None = None) -> Tuple[np.ndarray, list, dict]:
        """Returns market benchmark prices."""
        return self.get_price_series(market, lookback_days=lookback_days, seed=seed)
