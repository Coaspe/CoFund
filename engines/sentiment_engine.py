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


# ── v2 additions ──────────────────────────────────────────────────────────────

def dedupe_and_weight_news(
    articles: list,
    as_of: Optional[str] = None,
    source_weights: Optional[dict] = None,
    half_life_days: float = 2.5,
) -> dict:
    """
    1) difflib 유사도 기반 중복 제거 (ratio > 0.80 = 중복)
    2) 시간 감쇠 + 소스 가중치
    3) news_volume_z: 오늘 기사 수 vs 30일 baseline (평균/표준편차)
    """
    import math
    from difflib import SequenceMatcher
    from datetime import datetime, timezone, timedelta

    sw = source_weights or {}
    now = datetime.now(timezone.utc)
    if as_of:
        try:
            now = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass

    # ── Dedupe ───────────────────────────────────────────────────────
    seen: list[str] = []
    unique: list[dict] = []
    for art in articles:
        title = (art.get("title") or "").strip().lower()[:120]
        is_dup = any(SequenceMatcher(None, title, s).ratio() > 0.80 for s in seen)
        if not is_dup:
            seen.append(title)
            unique.append(art)

    # ── Time decay + source weight ────────────────────────────────────
    ln2 = math.log(2)
    count_by_day: dict[str, int] = {}
    weighted: list[dict] = []
    for art in unique:
        pub_at = art.get("published_at") or art.get("publishedAt") or ""
        try:
            pub_dt   = datetime.fromisoformat(pub_at.replace("Z", "+00:00"))
            days_old = max(0.0, (now - pub_dt).total_seconds() / 86400)
        except (ValueError, TypeError):
            pub_dt   = now - timedelta(days=7)
            days_old = 7.0
        decay = math.exp(-ln2 * days_old / half_life_days)
        src   = art.get("source") or art.get("source_name") or ""
        art   = dict(art)
        art["_weight"] = round(decay * sw.get(src, 1.0), 4)
        day_str = pub_dt.strftime("%Y-%m-%d")
        count_by_day[day_str] = count_by_day.get(day_str, 0) + 1
        weighted.append(art)

    weighted.sort(key=lambda a: a.get("_weight", 0), reverse=True)
    top_headlines = [
        {"title": a.get("title", ""), "source": a.get("source") or a.get("source_name", ""),
         "weight": a.get("_weight", 0)}
        for a in weighted[:8]
    ]

    # ── news_volume_z ─────────────────────────────────────────────────
    today_str   = now.strftime("%Y-%m-%d")
    today_count = count_by_day.get(today_str, 0)

    baseline_counts: list[int] = []
    baseline_days_with_data = 0
    for i in range(1, 31):
        d = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        if d in count_by_day:
            baseline_days_with_data += 1
        baseline_counts.append(count_by_day.get(d, 0))

    warnings: list[str] = []
    effective_count = len(unique)

    if effective_count < 10 or baseline_days_with_data < 10:
        warnings.append("insufficient_news_baseline")
        b_mean = b_std = news_volume_z = None
    else:
        b_mean = sum(baseline_counts) / len(baseline_counts)
        variance = sum((x - b_mean) ** 2 for x in baseline_counts) / len(baseline_counts)
        b_std  = variance ** 0.5
        news_volume_z = 0.0 if b_std < 1e-9 else round((today_count - b_mean) / b_std, 3)

    return {
        "effective_article_count": effective_count,
        "top_headlines":            top_headlines,
        "count_by_day":             count_by_day,
        "today_count":              today_count,
        "baseline_days_used":       baseline_days_with_data,
        "baseline_mean":            round(b_mean, 3) if b_mean is not None else None,
        "baseline_std":             round(b_std,  3) if b_std  is not None else None,
        "news_volume_z":            news_volume_z,
        "data_quality":             {"warnings": warnings},
    }


def compute_sentiment_velocity(score_series: list) -> dict:
    """
    3d/7d 변화율 + 급격한 반전 감지.
    입력: list of float or list of (date, float).
    """
    scores: list[float] = []
    for item in score_series:
        if isinstance(item, (int, float)):
            scores.append(float(item))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                scores.append(float(item[1]))
            except (TypeError, ValueError):
                pass

    if len(scores) < 3:
        return {"velocity_3d": None, "velocity_7d": None, "sharp_reversal": False}

    v3 = round(scores[-1] - scores[-3], 4)
    v7 = round(scores[-1] - scores[-7], 4) if len(scores) >= 7 else None
    sharp = (v7 is not None and abs(v3) > 0.15 and (v3 > 0) != (v7 > 0))

    return {"velocity_3d": v3, "velocity_7d": v7, "sharp_reversal": sharp}


_CATALYST_TYPES: dict[str, list[str]] = {
    "earnings":     ["earnings", "실적", "eps", "어닝"],
    "regulatory":   ["regulatory", "sec", "doj", "규제", "조사", "investigation"],
    "m&a_rumor":    ["merger", "acquisition", "buyout", "인수", "합병"],
    "product":      ["launch", "release", "제품", "발표"],
    "macro_event":  ["fomc", "cpi", "nonfarm", "gdp", "금리", "연준", "fed"],
}


def detect_catalyst_risk(
    upcoming_events: list,
    news_volume_z: Optional[float],
    event_window_days: int = 7,
) -> dict:
    """이벤트 타입 + 위험도: high = macro_event/regulatory OR z>2."""
    detected: list[str] = []
    for ev in upcoming_events:
        ev_str = str(ev).lower()
        for cat, kws in _CATALYST_TYPES.items():
            if any(kw in ev_str for kw in kws) and cat not in detected:
                detected.append(cat)

    present = bool(detected)
    if not present:
        level = "low"
    elif any(c in detected for c in ("macro_event", "regulatory")):
        level = "high"
    elif (news_volume_z is not None and news_volume_z > 2.0) or len(detected) >= 2:
        level = "high"
    elif news_volume_z is not None and news_volume_z > 1.0:
        level = "medium"
    else:
        level = "low"

    return {
        "catalyst_present":    present,
        "catalyst_type":       detected,
        "event_window_days":   event_window_days,
        "catalyst_risk_level": level,
    }


def infer_vol_regime(state: dict, sentiment_inputs: dict) -> dict:
    """
    vol_regime 단일 소스 함수 — 우선순위:
      1) quant regime_2_high_vol
      2) macro tail_risk_warning + credit_stress_level / risk_on_off
      3) sentiment_inputs.vix_level
      4) default normal + warning
    """
    warnings: list[str] = []

    # Priority 1: quant HMM high-vol regime probability
    quant = state.get("technical_analysis", {})
    r2 = quant.get("regime_2_high_vol")
    if r2 is not None:
        try:
            r2f = float(r2)
            vol  = "crisis" if r2f >= 0.50 else ("high" if r2f >= 0.35 else "normal")
            return {"vol_regime": vol, "source": "quant_regime_2", "value": r2f}
        except (TypeError, ValueError):
            pass

    # Priority 2: macro signals
    macro = state.get("macro_analysis", {})
    tail  = macro.get("tail_risk_warning", False)
    indicators_block = macro.get("indicators", {})
    credit_stress = indicators_block.get("credit_stress_level", "normal")
    ron_block = macro.get("risk_on_off", {})
    ron_val   = ron_block.get("risk_on_off", "") if isinstance(ron_block, dict) else ""

    if tail and credit_stress in ("crisis", "stressed"):
        return {"vol_regime": "crisis", "source": "macro_tail+credit", "value": None}
    if ron_val == "risk_off":
        return {"vol_regime": "high",   "source": "macro_risk_off",   "value": None}

    # Priority 3: VIX
    vix = sentiment_inputs.get("vix_level")
    if vix is not None:
        try:
            vf   = float(vix)
            vol  = "crisis" if vf >= 30 else ("high" if vf >= 22 else "normal")
            return {"vol_regime": vol, "source": "vix_level", "value": vf}
        except (TypeError, ValueError):
            pass

    # Priority 4: fallback
    warnings.append("vol_regime_unknown")
    return {"vol_regime": "normal", "source": "default_fallback", "value": None, "warnings": warnings}
