"""
agents/sentiment_agent.py — ④ Sentiment Analyst Agent (v2)
===========================================================
v2: news dedupe + news_volume_z + infer_vol_regime + catalyst_risk
    + key_drivers / what_to_watch / scenario_notes.
Iron Rule R5: tilt_factor [0.7, 1.3] 하드캡 유지. 방향성 단독 결정 금지.
LLM 없음. Python-only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from engines.sentiment_engine import (
    compute_sentiment_features,
    dedupe_and_weight_news,
    compute_sentiment_velocity,
    detect_catalyst_risk,
    infer_vol_regime,
)
from schemas.common import make_evidence, make_risk_flag


# ── Template helpers ──────────────────────────────────────────────────────────

def _build_key_drivers(features: dict, news_info: dict, catalyst: dict, vol_info: dict) -> list[str]:
    drivers = []
    vol_z   = news_info.get("news_volume_z")
    if vol_z is not None and abs(vol_z) >= 1.5:
        arrow = "↑" if vol_z > 0 else "↓"
        drivers.append(f"News volume spike {arrow} (z={vol_z:+.1f})")

    sr = features.get("sentiment_regime", "neutral")
    if sr in ("panic", "euphoria"):
        drivers.append(f"Sentiment extreme: {sr.upper()}")
    elif sr in ("fear", "greed"):
        drivers.append(f"Sentiment: {sr}")

    if catalyst.get("catalyst_present"):
        ctypes = ", ".join(catalyst.get("catalyst_type", []))
        drivers.append(f"Catalyst: {ctypes} (risk={catalyst.get('catalyst_risk_level', '?')})")

    vr = vol_info.get("vol_regime", "normal")
    if vr in ("crisis", "high"):
        src = vol_info.get("source", "")
        drivers.append(f"Vol regime {vr} (via {src})")

    pos = features.get("positioning_crowding", "balanced")
    if pos == "short_crowded":
        drivers.append("Short squeeze risk: crowded short interest")
    elif pos == "long_crowded":
        drivers.append("Long crowding: potential unwind risk")

    vel = features.get("_velocity", {}) or {}
    if vel.get("sharp_reversal"):
        drivers.append("Sentiment sharp reversal detected")

    return drivers[:6]


def _build_what_to_watch(features: dict, catalyst: dict) -> list[str]:
    items = []
    if catalyst.get("catalyst_present"):
        win = catalyst.get("event_window_days", 7)
        items.append(f"실적 발표 전후 {win}일 변동성 확대 — 분할 진입 권고")
    if features.get("vol_regime_from_inference") in ("crisis", "high"):
        items.append("VIX/vol 정상화(≤20) 확인 후 비중 확대 재고")
    if features.get("sentiment_regime") in ("euphoria", "greed"):
        items.append("뉴스 볼륨 정상화 시까지 tilt 상향 금지")
    if features.get("positioning_crowding") == "short_crowded":
        items.append("Short squeeze 발생 시 급등 후 역방향 반전 주의")
    items.append("PCR / VIX / 뉴스 볼륨 주 1회 이상 모니터링")
    return items[:5]


def _build_scenario_notes(features: dict, tilt: float, catalyst: dict) -> dict:
    sr  = features.get("sentiment_regime", "neutral")
    vol = features.get("vol_regime_from_inference", "normal")
    cat_high = catalyst.get("catalyst_risk_level") == "high"

    if sr == "panic":
        return {
            "bull": "극단적 공포 = 역발상 매수 기회. 분할 진입, 손절선 strict 설정.",
            "base": f"Tilt {tilt:.2f}: 공포 기반 tilt 상승. 기존 포지션 소폭 확대.",
            "bear": "공황 심화 시 추가 하락 위험. 헤지 없이 전량 진입 금지.",
        }
    if sr == "euphoria":
        return {
            "bull": "모멘텀 지속 가능하나 포지션 청산 시점 주의.",
            "base": f"Tilt {tilt:.2f}: 과열로 tilt 하향. 신규 롱 축소.",
            "bear": "포지션 청산 가속화 시 급락. 헤지 비중 확대 검토.",
        }
    if cat_high:
        return {
            "bull": "이벤트 서프라이즈 시 폭등 가능. 옵션 활용 비대칭 포지션.",
            "base": f"Tilt {tilt:.2f}: 이벤트 전 보수적. 비중 분할 진입.",
            "bear": "이벤트 실망 시 급락. 포지션 최소화 후 결과 확인.",
        }
    return {
        "bull": "감성 개선 + 볼륨 정상 = 중립~약간 우호적 타이밍.",
        "base": f"Tilt {tilt:.2f}: 일반 중립 구간. 기존 전략 유지.",
        "bear": "돌발 이벤트/뉴스 급증 시 즉시 tilt 재계산.",
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def sentiment_analyst_run(
    ticker: str,
    sentiment_indicators: dict,
    *,
    run_id: str = "",
    as_of: str = "",
    horizon_days: int = 7,
    source_name: str = "mock",
    state: Optional[dict] = None,          # for infer_vol_regime
    score_series: Optional[list] = None,   # for velocity
) -> dict:
    """
    Sentiment Analyst v2: news dedupe, news_volume_z, infer_vol_regime (3-tier),
    catalyst detection, tilt hardening, key_drivers/what_to_watch/scenario_notes.
    R5: recommendation 항상 allow_with_limits. tilt [0.7, 1.3] 하드캡.
    """
    as_of  = as_of or datetime.now(timezone.utc).isoformat()
    state  = state or {}

    # ── Engine calls ─────────────────────────────────────────────────
    features = compute_sentiment_features(sentiment_indicators)

    # vol_regime from infer_vol_regime (3-tier priority)
    vol_info = infer_vol_regime(state, sentiment_indicators)
    vol_regime = vol_info["vol_regime"]
    features["vol_regime_from_inference"] = vol_regime   # override for template helpers

    # News dedupe + volume_z
    articles = sentiment_indicators.get("news_articles", [])
    news_info = dedupe_and_weight_news(articles, as_of=as_of) if articles else {
        "effective_article_count": 0, "top_headlines": [], "count_by_day": {},
        "today_count": 0, "baseline_days_used": 0,
        "baseline_mean": None, "baseline_std": None, "news_volume_z": None,
        "data_quality": {"warnings": ["no_articles_provided"]},
    }
    news_volume_z = news_info.get("news_volume_z")

    # Velocity
    velocity = compute_sentiment_velocity(score_series or [])

    # Catalyst risk
    events   = sentiment_indicators.get("upcoming_events", [])
    catalyst = detect_catalyst_risk(events, news_volume_z)
    catalyst_level = catalyst.get("catalyst_risk_level", "low")

    # ── Tilt factor (amended) ─────────────────────────────────────────
    tilt = features["base_tilt_factor"]

    # Amendment: catalyst_high → tilt → 1.0 (no directional bet during catalyst)
    if catalyst_level == "high":
        tilt = 1.0
    # vol crisis → tilt ≤ 0.9
    if vol_regime == "crisis" and tilt > 0.9:
        tilt = 0.9
    # Final hardcap R5
    tilt = round(max(0.7, min(1.3, tilt)), 2)

    # ── Evidence ──────────────────────────────────────────────────────
    q = 0.3 if source_name == "mock" else 0.7
    evidence = []
    for key in ["put_call_ratio", "pcr_percentile_90d", "vix_level",
                "short_interest_pct", "news_sentiment_score"]:
        val = sentiment_indicators.get(key)
        if val is not None:
            evidence.append(make_evidence(metric=key, value=val, source_name=source_name, quality=q, as_of=as_of))
    for bk in ["pcr_state", "volatility_regime", "sentiment_regime", "positioning_crowding"]:
        evidence.append(make_evidence(metric=bk, value=features[bk],
                                      source_name="sentiment_engine", source_type="model", quality=0.9, as_of=as_of))
    evidence.append(make_evidence(metric="tilt_factor", value=tilt,
                                  source_name="sentiment_engine", source_type="model", quality=0.9, as_of=as_of,
                                  note="hard_cap [0.7,1.3] + catalyst/vol amendments"))

    # ── Risk flags ────────────────────────────────────────────────────
    risk_flags = []
    if vol_regime in ("crisis", "high"):
        risk_flags.append(make_risk_flag("high_volatility", "high", f"Vol: {vol_regime} (src={vol_info['source']})"))
    if catalyst.get("catalyst_present"):
        risk_flags.append(make_risk_flag("event_risk", "medium" if catalyst_level == "medium" else "high",
                                         f"Catalyst: {', '.join(catalyst.get('catalyst_type', []))}"))

    # ── Primary decision (sentinel: no reject) ────────────────────────
    sr  = features["sentiment_regime"]
    pos = features["positioning_crowding"]

    if sr in ("panic", "fear") and pos == "short_crowded":
        timing = "favorable_for_gradual_entry"
    elif sr == "euphoria" or (vol_regime == "crisis" and sr != "panic"):
        timing = "unfavorable_for_new_longs"
    elif vol_regime == "crisis":
        timing = "avoid_trading"
    else:
        timing = "neutral"

    primary_decision = (
        "bearish"  if timing == "unfavorable_for_new_longs"
        else ("bullish" if timing == "favorable_for_gradual_entry" else "neutral")
    )

    # ── Data quality ──────────────────────────────────────────────────
    data_ok  = bool([e for e in evidence if e.get("source_name") != "sentiment_engine"])
    warnings = list(news_info.get("data_quality", {}).get("warnings", []))
    if vol_info.get("warnings"):
        warnings.extend(vol_info["warnings"])

    limitations = []
    if source_name == "mock":
        limitations.append("Mock 감성 데이터 — 실제 시장 포지셔닝과 차이 가능")
    limitations.append("Sentiment는 tactical overlay만 — 단독 방향성 결정 불가 (R5)")
    if not articles:
        limitations.append("뉴스 기사 미제공 — news_volume_z 계산 불가")

    # ── Narrative ─────────────────────────────────────────────────────
    features["_velocity"] = velocity
    key_drivers    = _build_key_drivers(features, news_info, catalyst, vol_info)
    what_to_watch  = _build_what_to_watch(features, catalyst)
    scenario_notes = _build_scenario_notes(features, tilt, catalyst)

    data_quality = {
        "missing_fields": [] if data_ok else ["news_articles", "vix_level"],
        "anomaly_flags":  warnings,
        "is_mock":        source_name == "mock",
        "source_timestamps": {},
    }

    return {
        "agent_type":      "sentiment",
        "run_id":          run_id,
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "as_of":           as_of,
        "ticker":          ticker,
        "horizon_days":    horizon_days,
        "primary_decision": primary_decision,
        "recommendation":  "allow_with_limits",  # R5: never reject
        "confidence":      0.45 if data_ok else 0.30,
        "signal_strength": abs(tilt - 1.0) / 0.3,  # distance from neutral
        "risk_flags":      risk_flags,
        "evidence":        evidence,
        "data_quality":    data_quality,
        "limitations":     limitations,
        "data_ok":         data_ok,
        "summary":         (f"Sentiment: {sr}, Vol: {vol_regime}({vol_info['source']}), "
                            f"Tilt: {tilt}, Catalyst: {catalyst_level}"),
        "status":          "ok",
        # Structured
        "sentiment_regime":      sr,
        "positioning_crowding":  pos,
        "volatility_regime":     vol_regime,
        "vol_source":            vol_info,
        "entry_timing_signal":   timing,
        "tilt_factor":           tilt,
        "news_analysis":         news_info,
        "velocity":              velocity,
        "catalyst_risk":         catalyst,
        "key_drivers":           key_drivers,
        "what_to_watch":         what_to_watch,
        "scenario_notes":        scenario_notes,
        # Backward compat
        "overall_sentiment":     primary_decision,
        "sentiment_score":       sentiment_indicators.get("news_sentiment_score", 0),
        "catalysts": (
            [f"Upcoming: {', '.join(events[:3])}"] if events else
            ["Contrarian signal" if sr in ("panic", "fear") else ""]
        ),
        "tactical_notes": f"Tilt {tilt:.2f}. Vol {vol_regime}. Catalyst: {catalyst_level}.",
    }
