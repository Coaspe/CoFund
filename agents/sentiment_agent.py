"""
agents/sentiment_agent.py — ④ Sentiment Analyst Agent
=======================================================
CHANGELOG:
  v1.0 (2026-02-22) — 신규 생성. engine + mock LLM decision.

Iron Rule R5: Sentiment는 tactical overlay만. tilt_factor [0.7, 1.3] 하드캡.
              방향성을 단독으로 뒤집지 않음.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from engines.sentiment_engine import compute_sentiment_features
from schemas.common import make_evidence, make_risk_flag


def sentiment_analyst_run(
    ticker: str,
    sentiment_indicators: dict,
    *,
    run_id: str = "",
    as_of: str = "",
    horizon_days: int = 7,
    source_name: str = "mock",
) -> dict:
    """
    Sentiment Analyst 파이프라인: engine compute → mock decision → output dict.

    Sentiment는 approve/reject를 내리지 않음. allow_with_limits가 최대.
    """
    as_of = as_of or datetime.now(timezone.utc).isoformat()

    # ── Python 연산 (엔진) ────────────────────────────────────────────
    features = compute_sentiment_features(sentiment_indicators)

    sent_regime = features["sentiment_regime"]
    vol_regime = features["volatility_regime"]
    positioning = features["positioning_crowding"]
    tilt = features["base_tilt_factor"]
    event_risk = features["event_risk_flag"]

    # ── Evidence 생성 ─────────────────────────────────────────────────
    evidence = []
    q = 0.3 if source_name == "mock" else 0.7

    for key in ["put_call_ratio", "pcr_percentile_90d", "vix_level", "short_interest_pct", "news_sentiment_score"]:
        val = sentiment_indicators.get(key)
        if val is not None:
            evidence.append(make_evidence(metric=key, value=val, source_name=source_name, quality=q, as_of=as_of))

    for bucket_key in ["pcr_state", "volatility_regime", "sentiment_regime", "positioning_crowding"]:
        evidence.append(make_evidence(
            metric=bucket_key, value=features[bucket_key],
            source_name="sentiment_engine", source_type="model", quality=0.9, as_of=as_of,
        ))

    evidence.append(make_evidence(
        metric="tilt_factor", value=tilt,
        source_name="sentiment_engine", source_type="model", quality=0.9, as_of=as_of,
        note=f"hard_cap [0.7, 1.3]",
    ))

    # ── Risk Flags ────────────────────────────────────────────────────
    risk_flags = []
    if vol_regime in ("crisis", "high"):
        risk_flags.append(make_risk_flag("high_volatility", "high", f"VIX regime: {vol_regime}"))
    if event_risk:
        risk_flags.append(make_risk_flag("event_risk", "medium", f"Upcoming catalyst events"))

    # ── Mock LLM 결정 ─────────────────────────────────────────────────
    # Entry timing signal
    if sent_regime in ("panic", "fear") and positioning == "short_crowded":
        timing = "favorable_for_gradual_entry"
    elif sent_regime == "euphoria" or (vol_regime == "crisis" and sent_regime != "panic"):
        timing = "unfavorable_for_new_longs"
    elif vol_regime == "crisis":
        timing = "avoid_trading"
    else:
        timing = "neutral"

    # Catalysts
    catalysts = []
    if positioning == "short_crowded":
        catalysts.append("Short squeeze potential — crowded short interest")
    if event_risk:
        events = sentiment_indicators.get("upcoming_events", [])
        catalysts.append(f"Upcoming events: {', '.join(events[:3])}")
    if sent_regime in ("panic", "fear"):
        catalysts.append("Contrarian signal — extreme fear may present entry opportunity")
    if sent_regime in ("euphoria", "greed"):
        catalysts.append("Euphoria warning — crowded long positioning risks")

    # Sentiment NEVER outputs reject. Max is allow_with_limits.
    primary_decision = (
        "bearish" if timing == "unfavorable_for_new_longs"
        else ("bullish" if timing == "favorable_for_gradual_entry"
              else "neutral")
    )

    data_ok = len([e for e in evidence if e.get("source_name") != "sentiment_engine"]) > 0

    limitations = []
    if source_name == "mock":
        limitations.append("Mock 감성 데이터 — 실제 시장 포지셔닝과 차이 가능")
    limitations.append("Sentiment는 tactical overlay만 — 단독 방향성 결정 불가 (R5)")

    return {
        "agent_type": "sentiment",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of": as_of,
        "ticker": ticker,
        "horizon_days": horizon_days,
        "primary_decision": primary_decision,
        "recommendation": "allow_with_limits",  # NEVER reject from sentiment alone
        "confidence": 0.45 if data_ok else 0.3,
        "risk_flags": risk_flags,
        "evidence": evidence,
        "limitations": limitations,
        "data_ok": data_ok,
        "summary": f"Sentiment: {sent_regime}, Vol: {vol_regime}, Timing: {timing}, Tilt: {tilt}",
        "status": "ok",
        # Sentiment-specific
        "sentiment_regime": sent_regime,
        "positioning_crowding": positioning,
        "volatility_regime": vol_regime,
        "entry_timing_signal": timing,
        "catalysts": catalysts,
        "tilt_factor": tilt,
        "tactical_notes": f"Tilt {tilt:.2f} applied. Vol regime {vol_regime}.",
        # Backward compat
        "overall_sentiment": primary_decision,
        "sentiment_score": sentiment_indicators.get("news_sentiment_score", 0),
    }
