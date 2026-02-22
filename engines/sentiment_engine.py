"""
engines/sentiment_engine.py — Sentiment 순수 연산 엔진
=====================================================
CHANGELOG:
  v1.0 (2026-02-22) — 신규 생성.

PCR, VIX regime, skew bucketing, positioning, tilt_factor.
Iron Rule R5: tilt_factor는 [0.7, 1.3]로 하드캡.
데이터 수집 금지.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def compute_sentiment_features(indicators: dict) -> dict:
    """
    감성/포지셔닝/변동성 지표에서 state bucket + tilt factor를 파생.

    Args:
        indicators: {
            "put_call_ratio": float,
            "pcr_percentile_90d": float (0-100),
            "vix_level": float,
            "vix_term_structure": "contango"|"flat"|"backwardation",
            "skew_index": float (CBOE SKEW),
            "short_interest_pct": float,
            "insider_net_activity": "buying"|"selling"|"neutral",
            "news_sentiment_score": float (-1 ~ 1),
            "article_count_7d": int,
            "upcoming_events": list[str],
            "crypto_funding_rate": float (optional),
            "crypto_oi_pct_change_24h": float (optional),
        }
    """
    pcr = indicators.get("put_call_ratio")
    pcr_pct = indicators.get("pcr_percentile_90d")
    vix = indicators.get("vix_level")
    vix_term = indicators.get("vix_term_structure", "contango")
    skew = indicators.get("skew_index")
    si_pct = indicators.get("short_interest_pct")
    insider = indicators.get("insider_net_activity", "neutral")
    news_score = indicators.get("news_sentiment_score")
    events = indicators.get("upcoming_events", [])
    funding = indicators.get("crypto_funding_rate")
    oi_change = indicators.get("crypto_oi_pct_change_24h")

    # ── PCR State ─────────────────────────────────────────────────────
    if pcr_pct is not None:
        if pcr_pct > 90:
            pcr_state = "extreme_fear"
        elif pcr_pct > 70:
            pcr_state = "fear"
        elif pcr_pct < 10:
            pcr_state = "extreme_greed"
        elif pcr_pct < 30:
            pcr_state = "greed"
        else:
            pcr_state = "neutral"
    elif pcr is not None:
        if pcr > 1.2:
            pcr_state = "extreme_fear"
        elif pcr > 0.9:
            pcr_state = "fear"
        elif pcr < 0.5:
            pcr_state = "extreme_greed"
        elif pcr < 0.7:
            pcr_state = "greed"
        else:
            pcr_state = "neutral"
    else:
        pcr_state = "unknown"

    # ── VIX / Volatility Regime ───────────────────────────────────────
    if vix is not None:
        if vix > 35:
            vol_regime = "crisis"
        elif vix > 25:
            vol_regime = "high"
        elif vix > 15:
            vol_regime = "normal"
        else:
            vol_regime = "low"
    else:
        vol_regime = "unknown"

    # ── Skew ──────────────────────────────────────────────────────────
    if skew is not None:
        if skew > 150:
            skew_state = "extreme"
        elif skew > 135:
            skew_state = "elevated"
        elif skew > 120:
            skew_state = "normal"
        else:
            skew_state = "compressed"
    else:
        skew_state = "unknown"

    # ── Positioning / Crowding ────────────────────────────────────────
    if si_pct is not None:
        if si_pct > 20:
            positioning = "short_crowded"
        elif si_pct > 10:
            positioning = "balanced"
        else:
            positioning = "long_crowded"
    else:
        positioning = "balanced"

    # ── Sentiment Regime (종합) ───────────────────────────────────────
    fear_signals = sum([
        pcr_state in ("extreme_fear", "fear"),
        vol_regime in ("crisis", "high"),
        news_score is not None and news_score < -0.3,
    ])
    greed_signals = sum([
        pcr_state in ("extreme_greed", "greed"),
        vol_regime == "low",
        news_score is not None and news_score > 0.3,
    ])

    if fear_signals >= 2:
        sentiment_regime = "panic" if vol_regime == "crisis" else "fear"
    elif greed_signals >= 2:
        sentiment_regime = "euphoria" if pcr_state == "extreme_greed" else "greed"
    else:
        sentiment_regime = "neutral"

    # ── Event Risk ────────────────────────────────────────────────────
    event_risk_flag = len(events) > 0 and any(
        kw in str(events).lower() for kw in ["earnings", "fomc", "cpi", "nonfarm"]
    )

    # ── Crypto Signals (optional) ─────────────────────────────────────
    if funding is not None:
        if funding > 0.05:
            funding_state = "long_crowded_leveraged"
        elif funding < -0.02:
            funding_state = "short_crowded_leveraged"
        else:
            funding_state = "balanced"
    else:
        funding_state = "unknown"

    oi_shock = oi_change is not None and abs(oi_change) > 15

    # ── Base Tilt Factor [0.7, 1.3] HARD CAP (Iron Rule R5) ──────────
    tilt = 1.0

    # Fear → contrarian long tilt (buy fear)
    if sentiment_regime == "panic":
        tilt += 0.15  # contrarian: extreme fear = buy opportunity
    elif sentiment_regime == "fear":
        tilt += 0.10

    # Greed → reduce tilt
    if sentiment_regime == "euphoria":
        tilt -= 0.20
    elif sentiment_regime == "greed":
        tilt -= 0.10

    # Short crowding → potential squeeze → long tilt
    if positioning == "short_crowded":
        tilt += 0.10

    # Event risk → dampen
    if event_risk_flag:
        tilt -= 0.05

    # Vol crisis → cap long tilt to 0.9
    if vol_regime in ("crisis", "high") and tilt > 0.9:
        tilt = 0.9

    # HARD CAP
    tilt = round(max(0.7, min(1.3, tilt)), 2)

    return {
        "pcr_state": pcr_state,
        "volatility_regime": vol_regime,
        "skew_state": skew_state,
        "positioning_crowding": positioning,
        "sentiment_regime": sentiment_regime,
        "event_risk_flag": event_risk_flag,
        "funding_state": funding_state,
        "oi_shock": oi_shock,
        "base_tilt_factor": tilt,
        "insider_activity": insider,
    }
