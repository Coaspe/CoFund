"""
data_providers/twelvedata_provider.py — Twelve Data price series provider
=========================================================================
Replaces yfinance for quant/risk agents. Fallback to yfinance if TwelveData fails.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np

from schemas.common import make_evidence
from config.settings import get_settings
from data_providers.base import BaseProvider, ProviderError, is_rate_limit_error

_TD_BASE = "https://api.twelvedata.com"


class TwelveDataProvider(BaseProvider):
    PROVIDER_NAME = "twelvedata"

    def __init__(self, **kwargs):
        self._api_key = get_settings().twelvedata_api_key
        super().__init__(**kwargs)

    @property
    def has_key(self) -> bool:
        return bool(self._api_key)

    def get_time_series(self, symbol: str, interval: str = "1day", outputsize: int = 800) -> List[dict]:
        if not self.has_key:
            raise ProviderError("TWELVEDATA_API_KEY not set")

        url = f"{_TD_BASE}/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": min(outputsize, 5000),
            "apikey": self._api_key,
            "format": "JSON",
        }

        data = self.get_json(url, params)

        if data.get("status") == "error":
            raise ProviderError(f"TwelveData error: {data.get('message', 'unknown')}")

        values = data.get("values", [])
        if not values:
            raise ProviderError(f"No data returned for {symbol}")

        return values

    def to_numpy_close(self, series: List[dict]) -> np.ndarray:
        closes = []
        for bar in reversed(series):  # API returns newest first
            try:
                closes.append(float(bar["close"]))
            except (KeyError, ValueError):
                continue
        return np.array(closes, dtype=np.float64)

    def get_prices(self, symbols: List[str], interval: str = "1day",
                   lookback_days: int = 504) -> Dict[str, np.ndarray]:
        outputsize = min(lookback_days, 5000)
        result: Dict[str, np.ndarray] = {}
        for sym in symbols:
            try:
                series = self.get_time_series(sym, interval=interval, outputsize=outputsize)
                result[sym] = self.to_numpy_close(series)
            except ProviderError:
                result[sym] = self._yfinance_fallback(sym, lookback_days)
        return result

    def get_price_series(self, symbol: str, lookback_days: int = 504,
                         as_of: str = "") -> dict:
        as_of = as_of or datetime.now(timezone.utc).isoformat()
        evidence: list = []
        limitations: list = []

        if not self.has_key:
            prices = self._mock_prices(symbol, lookback_days)
            evidence.append(make_evidence(
                metric="close_prices", value=f"{len(prices)}pt mock",
                source_name="mock", quality=0.3, as_of=as_of,
            ))
            return {"data": prices, "evidence": evidence, "data_ok": False,
                    "limitations": ["TWELVEDATA_API_KEY not set; using mock"], "as_of": as_of}

        try:
            series = self.get_time_series(symbol, outputsize=lookback_days)
            prices = self.to_numpy_close(series)
            evidence.append(make_evidence(
                metric="close_prices", value=f"{len(prices)}pt",
                source_name=f"TwelveData:{symbol}", source_type="api",
                quality=0.85, as_of=as_of,
            ))
            return {"data": prices, "evidence": evidence, "data_ok": True,
                    "limitations": limitations, "as_of": as_of}
        except ProviderError as e:
            if is_rate_limit_error(e):
                print("   [API Router] twelvedata: rate limit으로 yfinance fallback 전환", flush=True)
            limitations.append(f"TwelveData error: {e}")
            # Fallback to yfinance
            try:
                prices = self._yfinance_fallback(symbol, lookback_days)
                evidence.append(make_evidence(
                    metric="close_prices", value=f"{len(prices)}pt yfinance fallback",
                    source_name="yfinance", source_type="api", quality=0.75, as_of=as_of,
                ))
                return {"data": prices, "evidence": evidence, "data_ok": True,
                        "limitations": limitations + ["Using yfinance fallback"], "as_of": as_of}
            except Exception:
                prices = self._mock_prices(symbol, lookback_days)
                evidence.append(make_evidence(
                    metric="close_prices", value=f"{len(prices)}pt mock",
                    source_name="mock", quality=0.3, as_of=as_of,
                ))
                return {"data": prices, "evidence": evidence, "data_ok": False,
                        "limitations": limitations + ["All price sources failed; using mock"], "as_of": as_of}

    @staticmethod
    def _yfinance_fallback(symbol: str, days: int) -> np.ndarray:
        import yfinance as yf  # type: ignore
        from datetime import timedelta
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=int(days * 1.5))
        df = yf.download(symbol, start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
        if df.empty or len(df) < 50:
            raise ValueError(f"yfinance: insufficient data for {symbol}")
        return df["Close"].dropna().values.flatten().astype(np.float64)[-days:]

    @staticmethod
    def _mock_prices(symbol: str, days: int) -> np.ndarray:
        rng = np.random.default_rng(hash(symbol) % (2**31))
        mu, theta, sigma = 150.0, 0.05, 1.5
        px = [mu]
        for _ in range(days - 1):
            px.append(max(px[-1] + theta * (mu - px[-1]) + rng.normal(0, sigma), 1.0))
        return np.array(px)

    def get_quote(self, symbol: str) -> dict | None:
        if not self.has_key:
            return None
        try:
            url = f"{_TD_BASE}/quote"
            params = {"symbol": symbol, "apikey": self._api_key}
            return self.get_json(url, params)
        except ProviderError:
            return None


if __name__ == "__main__":
    import json
    provider = TwelveDataProvider()
    symbol = "AAPL"
    if provider.has_key:
        print(f"📈 TwelveData — fetching {symbol} prices...")
    else:
        print(f"⚠️  No TWELVEDATA_API_KEY — mock for {symbol}")
    result = provider.get_price_series(symbol)
    prices = result["data"]
    print(f"Prices: {len(prices)} data points, last={prices[-1]:.2f}")
    print(f"data_ok: {result['data_ok']}")
    print(f"Limitations: {result['limitations']}")
