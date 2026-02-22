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
