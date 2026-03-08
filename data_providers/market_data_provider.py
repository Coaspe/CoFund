"""
data_providers/market_data_provider.py — 시장 가격 데이터 프로바이더
===================================================================
CHANGELOG:
  v1.0 (2026-02-22) — 신규 생성. yfinance optional, mock fallback.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import numpy as np

from schemas.common import make_evidence

_MACRO_MARKET_SYMBOLS = {
    "wti_front_month": "CL=F",
    "brent_front_month": "BZ=F",
    "vix_index": "^VIX",
}


def fetch_prices(
    ticker: str,
    lookback_days: int = 504,
    *,
    mode: str = "mock",
    as_of: str = "",
    seed: int | None = None,
) -> Tuple[np.ndarray, List[dict]]:
    """
    가격 배열 + evidence 반환.

    mode="live": yfinance 사용 (미설치 시 에러)
    mode="mock": OU 프로세스 합성
    """
    as_of = as_of or datetime.now(timezone.utc).isoformat()

    if mode == "live":
        return _fetch_live(ticker, lookback_days, as_of)
    else:
        return _fetch_mock(ticker, lookback_days, as_of, seed)


def _fetch_live(ticker: str, days: int, as_of: str) -> Tuple[np.ndarray, List[dict]]:
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        raise ImportError("yfinance 미설치. pip install yfinance 또는 mode=mock 사용.")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(days * 1.5))
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                     progress=False, auto_adjust=True)
    if df.empty or len(df) < 50:
        raise ValueError(f"[{ticker}] 데이터 부족 ({len(df)}행)")

    prices = df["Close"].dropna().values.flatten().astype(np.float64)[-days:]
    evidence = [make_evidence(
        metric="adj_close", value=f"{len(prices)}pt",
        source_name="yfinance", source_type="api", quality=0.8, as_of=as_of,
        note=f"{ticker} live prices ({len(prices)} trading days)",
    )]
    return prices, evidence


def _fetch_mock(ticker: str, days: int, as_of: str, seed: int | None) -> Tuple[np.ndarray, List[dict]]:
    rng = np.random.default_rng(seed)
    mu, theta, sigma = 150.0, 0.05, 1.5
    px = [mu]
    for _ in range(days - 1):
        px.append(max(px[-1] + theta * (mu - px[-1]) + rng.normal(0, sigma), 1.0))
    prices = np.array(px)
    evidence = [make_evidence(
        metric="adj_close", value=f"{days}pt OU synth",
        source_name="mock", quality=0.3, as_of=as_of,
        note=f"{ticker} synthetic via OU process",
    )]
    return prices, evidence


def fetch_macro_market_indicators(
    *,
    mode: str = "mock",
    as_of: str = "",
    seed: int | None = None,
) -> Tuple[dict, List[dict], dict]:
    as_of = as_of or datetime.now(timezone.utc).isoformat()
    if mode == "live":
        try:
            data, evidence = _fetch_live_macro_market(as_of)
            return data, evidence, {"data_ok": bool(data), "limitations": [], "as_of": as_of}
        except Exception as e:
            data, evidence = _fetch_mock_macro_market(as_of, seed)
            return data, evidence, {
                "data_ok": False,
                "limitations": [f"market_data_provider live error: {e}", "Using mock macro market data"],
                "as_of": as_of,
            }
    data, evidence = _fetch_mock_macro_market(as_of, seed)
    return data, evidence, {"data_ok": False, "limitations": ["Mock macro market data"], "as_of": as_of}


def _fetch_live_macro_market(as_of: str) -> Tuple[dict, List[dict]]:
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        raise ImportError("yfinance 미설치. pip install yfinance 필요")

    data: dict = {}
    evidence: List[dict] = []

    for metric, symbol in _MACRO_MARKET_SYMBOLS.items():
        df = yf.download(symbol, period="1mo", interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty:
            continue
        close = df["Close"]
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        close = close.dropna()
        if close.empty:
            continue
        last_ts = close.index[-1]
        last_val = float(close.iloc[-1])
        ts = getattr(last_ts, "to_pydatetime", lambda: last_ts)()
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts.astimezone(timezone.utc)
            point_as_of = ts.isoformat()
        else:
            point_as_of = as_of
        data[metric] = round(last_val, 4)
        evidence.append(
            make_evidence(
                metric=metric,
                value=data[metric],
                source_name=f"yfinance:{symbol}",
                source_type="api",
                quality=0.75,
                as_of=point_as_of,
                note=f"last trading date for {symbol}",
            )
        )
    return data, evidence


def _fetch_mock_macro_market(as_of: str, seed: int | None) -> Tuple[dict, List[dict]]:
    rng = np.random.default_rng(seed if seed is not None else 42)
    data = {
        "wti_front_month": round(float(rng.uniform(65, 95)), 2),
        "brent_front_month": round(float(rng.uniform(68, 100)), 2),
        "vix_index": round(float(rng.uniform(12, 32)), 2),
    }
    evidence = [
        make_evidence(
            metric=metric,
            value=value,
            source_name="mock",
            quality=0.3,
            as_of=as_of,
            note="synthetic macro market fallback",
        )
        for metric, value in data.items()
    ]
    return data, evidence
