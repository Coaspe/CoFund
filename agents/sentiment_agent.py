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

from agents.autonomy_overlay import apply_llm_overlay_sentiment
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
    vol_z = news_info.get("news_volume_z")
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
    sr = features.get("sentiment_regime", "neutral")
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


_SENTI_EVIDENCE_KINDS = {
    "catalyst_event_detail",
    "macro_headline_context",
    "press_release_or_ir",
    "ownership_identity",
}
_PATCHABLE_FIELDS = {
    "key_drivers",
    "what_to_watch",
    "scenario_notes",
    "open_questions",
    "decision_sensitivity",
    "followups",
    "react_trace",
}


def _request_key(req: dict) -> tuple:
    return (
        req.get("desk", ""),
        req.get("kind", ""),
        req.get("ticker", ""),
        req.get("series_id", ""),
        req.get("query", ""),
    )


def _merge_requests(base: list[dict], extra: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen = set()
    for req in (base or []) + (extra or []):
        if not isinstance(req, dict):
            continue
        k = _request_key(req)
        if k in seen:
            continue
        seen.add(k)
        out.append(req)
    return out


def _build_evidence_digest(state: Optional[dict], ticker: str, max_items: int = 5) -> list[dict]:
    store = (state or {}).get("evidence_store", {})
    if not isinstance(store, dict):
        return []
    out: list[dict] = []
    for item in store.values():
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", ""))
        desk = str(item.get("desk", ""))
        if kind and kind not in _SENTI_EVIDENCE_KINDS and desk not in ("sentiment", ""):
            continue
        item_ticker = str(item.get("ticker", "")).upper()
        if item_ticker and item_ticker != ticker.upper():
            continue
        out.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "published_at": item.get("published_at", ""),
                "trust_tier": item.get("trust_tier", 0.4),
                "kind": kind,
                "resolver_path": item.get("resolver_path", ""),
            }
        )
    out.sort(key=lambda x: (x.get("published_at") or ""), reverse=True)
    return out[:max_items]


def _default_open_questions(ticker: str, focus_areas: list[str], catalyst_level: str) -> list[dict]:
    out: list[dict] = []
    for focus in focus_areas[:2]:
        out.append(
            {
                "q": f"{focus} 이슈가 {ticker}의 단기 감성 레짐을 반전시킬 가능성은?",
                "why": "sentiment 결론을 바꾸는 촉매 확인",
                "kind": "catalyst_event_detail",
                "priority": 2,
                "recency_days": 14,
            }
        )
    if catalyst_level == "high":
        out.append(
            {
                "q": f"{ticker} 이벤트 리스크의 정확한 일정/서프라이즈 범위는?",
                "why": "이벤트 전후 tilt 조정의 핵심 입력",
                "kind": "press_release_or_ir",
                "priority": 1,
                "recency_days": 14,
            }
        )
    if not out:
        out.append(
            {
                "q": f"{ticker} 뉴스 볼륨 급증의 원인이 일시적 노이즈인지 구조적 촉매인지?",
                "why": "전술적 진입 타이밍 판별",
                "kind": "macro_headline_context",
                "priority": 3,
                "recency_days": 7,
            }
        )
    return out[:5]


def _default_decision_sensitivity(vol_regime: str) -> list[dict]:
    out = [
        {
            "if": "이벤트 리스크가 소멸하고 뉴스 볼륨 z-score가 정상화",
            "then_change": "중립 유지에서 제한적 리스크온으로 완화",
            "impact": "medium",
        }
    ]
    if vol_regime in ("high", "crisis"):
        out.append(
            {
                "if": "변동성 레짐이 normal로 하향",
                "then_change": "보수적 timing 신호를 중립으로 상향",
                "impact": "high",
            }
        )
    return out[:5]


def _default_followups() -> list[dict]:
    return [
        {
            "type": "run_research",
            "detail": "이벤트/뉴스 촉매 세부 확인을 위한 리서치 실행",
            "params": {"desk": "sentiment"},
        },
        {
            "type": "rerun_desk",
            "detail": "신규 증거 반영 후 감성 레짐/tilt 재평가",
            "params": {"desk": "sentiment"},
        },
    ]


def _default_react_trace(has_evidence: bool) -> list[dict]:
    return [
        {"phase": "THOUGHT", "summary": "감성 결론을 바꾸는 이벤트 공백을 식별"},
        {"phase": "ACTION", "summary": "질문과 후속 액션을 구조화"},
        {
            "phase": "OBSERVATION",
            "summary": "신규 근거를 key drivers/watchlist에 반영" if has_evidence else "증거 대기 상태로 기본 가정 유지",
        },
        {"phase": "REFLECTION", "summary": "조건부 민감도를 점검해 결론 강도를 명시"},
    ]


def _apply_overlay_patch(output: dict, patch: dict) -> None:
    if not isinstance(patch, dict):
        return
    for key in _PATCHABLE_FIELDS:
        if key in patch and patch[key]:
            output[key] = patch[key]
    if patch.get("evidence_requests"):
        output["evidence_requests"] = _merge_requests(
            output.get("evidence_requests", []),
            patch.get("evidence_requests", []),
        )
        output["needs_more_data"] = bool(output.get("evidence_requests"))


def _generate_evidence_requests(
    ticker: str,
    features: dict,
    catalyst: dict,
    news_info: dict,
    vol_info: dict,
    focus_areas: Optional[list[str]] = None,
) -> list:
    """센티먼트 evidence request 생성."""
    reqs = []
    cat_level = catalyst.get("catalyst_risk_level", "low")
    cat_types = catalyst.get("catalyst_type", [])
    vol_z = news_info.get("news_volume_z")

    if cat_level == "high":
        cats = ", ".join(cat_types[:3]) if cat_types else "unknown"
        reqs.append(
            {
                "desk": "sentiment",
                "kind": "catalyst_event_detail",
                "ticker": ticker,
                "query": f"{ticker} {cats} event upcoming catalyst detail",
                "priority": 1,
                "recency_days": 14,
                "max_items": 5,
                "rationale": f"catalyst_risk_level=high ({cats})",
            }
        )
    if vol_z is not None and vol_z > 2.0:
        reqs.append(
            {
                "desk": "sentiment",
                "kind": "macro_headline_context",
                "ticker": ticker,
                "query": f"{ticker} news volume spike reason",
                "priority": 2,
                "recency_days": 3,
                "max_items": 3,
                "rationale": f"news_volume_z={vol_z:.1f} spike — cause unknown",
            }
        )
    if vol_info.get("vol_regime") == "crisis" and vol_info.get("source") == "default_fallback":
        reqs.append(
            {
                "desk": "sentiment",
                "kind": "macro_headline_context",
                "ticker": ticker,
                "query": "VIX spike market volatility crisis reason",
                "priority": 2,
                "recency_days": 3,
                "max_items": 3,
                "rationale": "vol_regime=crisis but source=default_fallback",
            }
        )

    for focus in (focus_areas or [])[:2]:
        reqs.append(
            {
                "desk": "sentiment",
                "kind": "catalyst_event_detail",
                "ticker": ticker,
                "query": f"{ticker} {focus} sentiment catalyst",
                "priority": 3,
                "recency_days": 14,
                "max_items": 3,
                "rationale": f"focus area follow-up: {focus}",
            }
        )

    return reqs


# ── Main pipeline ─────────────────────────────────────────────────────────────

def sentiment_analyst_run(
    ticker: str,
    sentiment_indicators: dict,
    *,
    run_id: str = "",
    as_of: str = "",
    horizon_days: int = 7,
    source_name: str = "mock",
    state: Optional[dict] = None,
    score_series: Optional[list] = None,
    focus_areas: Optional[list[str]] = None,
) -> dict:
    """
    Sentiment Analyst v2: news dedupe, news_volume_z, infer_vol_regime (3-tier),
    catalyst detection, tilt hardening, key_drivers/what_to_watch/scenario_notes.
    R5: recommendation always allow_with_limits. tilt [0.7, 1.3] hardcap.
    """
    as_of = as_of or datetime.now(timezone.utc).isoformat()
    state = state or {}
    focus_areas = [str(x).strip() for x in (focus_areas or []) if str(x).strip()]

    features = compute_sentiment_features(sentiment_indicators)

    vol_info = infer_vol_regime(state, sentiment_indicators)
    vol_regime = vol_info["vol_regime"]
    features["vol_regime_from_inference"] = vol_regime

    articles = sentiment_indicators.get("news_articles", [])
    news_info = (
        dedupe_and_weight_news(articles, as_of=as_of)
        if articles
        else {
            "effective_article_count": 0,
            "top_headlines": [],
            "count_by_day": {},
            "today_count": 0,
            "baseline_days_used": 0,
            "baseline_mean": None,
            "baseline_std": None,
            "news_volume_z": None,
            "data_quality": {"warnings": ["no_articles_provided"]},
        }
    )
    news_volume_z = news_info.get("news_volume_z")

    velocity = compute_sentiment_velocity(score_series or [])

    events = sentiment_indicators.get("upcoming_events", [])
    catalyst = detect_catalyst_risk(events, news_volume_z)
    catalyst_level = catalyst.get("catalyst_risk_level", "low")

    tilt = features["base_tilt_factor"]
    if catalyst_level == "high":
        tilt = 1.0
    if vol_regime == "crisis" and tilt > 0.9:
        tilt = 0.9
    tilt = round(max(0.7, min(1.3, tilt)), 2)

    q = 0.3 if source_name == "mock" else 0.7
    evidence = []
    for key in ["put_call_ratio", "pcr_percentile_90d", "vix_level", "short_interest_pct", "news_sentiment_score"]:
        val = sentiment_indicators.get(key)
        if val is not None:
            evidence.append(make_evidence(metric=key, value=val, source_name=source_name, quality=q, as_of=as_of))
    for bk in ["pcr_state", "volatility_regime", "sentiment_regime", "positioning_crowding"]:
        evidence.append(
            make_evidence(
                metric=bk,
                value=features[bk],
                source_name="sentiment_engine",
                source_type="model",
                quality=0.9,
                as_of=as_of,
            )
        )
    evidence.append(
        make_evidence(
            metric="tilt_factor",
            value=tilt,
            source_name="sentiment_engine",
            source_type="model",
            quality=0.9,
            as_of=as_of,
            note="hard_cap [0.7,1.3] + catalyst/vol amendments",
        )
    )

    risk_flags = []
    if vol_regime in ("crisis", "high"):
        risk_flags.append(make_risk_flag("high_volatility", "high", f"Vol: {vol_regime} (src={vol_info['source']})"))
    if catalyst.get("catalyst_present"):
        risk_flags.append(
            make_risk_flag(
                "event_risk",
                "medium" if catalyst_level == "medium" else "high",
                f"Catalyst: {', '.join(catalyst.get('catalyst_type', []))}",
            )
        )

    sr = features["sentiment_regime"]
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
        "bearish"
        if timing == "unfavorable_for_new_longs"
        else ("bullish" if timing == "favorable_for_gradual_entry" else "neutral")
    )

    data_ok = bool([e for e in evidence if e.get("source_name") != "sentiment_engine"])
    warnings = list(news_info.get("data_quality", {}).get("warnings", []))
    if vol_info.get("warnings"):
        warnings.extend(vol_info["warnings"])

    limitations = []
    if source_name == "mock":
        limitations.append("Mock 감성 데이터 — 실제 시장 포지셔닝과 차이 가능")
    limitations.append("Sentiment는 tactical overlay만 — 단독 방향성 결정 불가 (R5)")
    if not articles:
        limitations.append("뉴스 기사 미제공 — news_volume_z 계산 불가")

    ev_reqs = _generate_evidence_requests(ticker, features, catalyst, news_info, vol_info, focus_areas)

    features["_velocity"] = velocity
    key_drivers = _build_key_drivers(features, news_info, catalyst, vol_info)
    what_to_watch = _build_what_to_watch(features, catalyst)
    scenario_notes = _build_scenario_notes(features, tilt, catalyst)
    evidence_digest = _build_evidence_digest(state, ticker)
    if evidence_digest:
        title = str(evidence_digest[0].get("title", "external evidence")).strip()[:90]
        kinds = sorted({str(item.get("kind", "")).strip() for item in evidence_digest if item.get("kind")})
        kind_label = ", ".join(kinds[:2]) if kinds else "sentiment_evidence"
        resolver = str(evidence_digest[0].get("resolver_path", "")).strip() or "unknown"
        key_drivers = (
            key_drivers
            + [
                f"Evidence update: {title}",
                f"Evidence coverage: {len(evidence_digest)} items ({kind_label})",
            ]
        )[:6]
        what_to_watch = (
            what_to_watch
            + [
                "신규 근거와 변동성/촉매 시그널 충돌 여부 재확인",
                f"최신 근거 경로({resolver})의 이벤트 후속 확인",
            ]
        )[:5]

    open_questions = _default_open_questions(ticker, focus_areas, catalyst_level)
    decision_sensitivity = _default_decision_sensitivity(vol_regime)
    followups = _default_followups()
    react_trace = _default_react_trace(bool(evidence_digest))

    missing_fields = [] if data_ok else ["news_articles", "vix_level"]
    missing_pct = round(len(missing_fields) / 2.0, 3)
    data_quality = {
        "missing_pct": missing_pct,
        "freshness_days": 0.0,
        "warnings": list(warnings),
        "missing_fields": missing_fields,
        "anomaly_flags": warnings,
        "is_mock": source_name == "mock",
        "source_timestamps": {},
    }

    output = {
        "agent_type": "sentiment",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of": as_of,
        "ticker": ticker,
        "horizon_days": horizon_days,
        "focus_areas": focus_areas,
        "primary_decision": primary_decision,
        "recommendation": "allow_with_limits",
        "confidence": 0.45 if data_ok else 0.30,
        "signal_strength": abs(tilt - 1.0) / 0.3,
        "risk_flags": risk_flags,
        "evidence": evidence,
        "data_quality": data_quality,
        "limitations": limitations,
        "data_ok": data_ok,
        "summary": f"Sentiment: {sr}, Vol: {vol_regime}({vol_info['source']}), Tilt: {tilt}, Catalyst: {catalyst_level}",
        "status": "ok",
        # Structured
        "sentiment_regime": sr,
        "positioning_crowding": pos,
        "volatility_regime": vol_regime,
        "vol_source": vol_info,
        "entry_timing_signal": timing,
        "tilt_factor": tilt,
        "news_analysis": news_info,
        "velocity": velocity,
        "catalyst_risk": catalyst,
        "key_drivers": key_drivers,
        "what_to_watch": what_to_watch,
        "scenario_notes": scenario_notes,
        "evidence_digest": evidence_digest,
        "open_questions": open_questions,
        "decision_sensitivity": decision_sensitivity,
        "followups": followups,
        "react_trace": react_trace,
        "needs_more_data": bool(ev_reqs),
        "evidence_requests": ev_reqs,
        # Backward compat
        "overall_sentiment": primary_decision,
        "sentiment_score": sentiment_indicators.get("news_sentiment_score", 0),
        "catalysts": [f"Upcoming: {', '.join(events[:3])}"] if events else ["Contrarian signal" if sr in ("panic", "fear") else ""],
        "tactical_notes": f"Tilt {tilt:.2f}. Vol {vol_regime}. Catalyst: {catalyst_level}.",
    }

    patch = apply_llm_overlay_sentiment(output, state, focus_areas, evidence_digest)
    _apply_overlay_patch(output, patch)

    if not output.get("open_questions"):
        output["open_questions"] = open_questions
    if not output.get("decision_sensitivity"):
        output["decision_sensitivity"] = decision_sensitivity
    if not output.get("followups"):
        output["followups"] = followups

    # R5 final hardcap guard (overlay must not bypass)
    output["tilt_factor"] = round(max(0.7, min(1.3, float(output.get("tilt_factor", 1.0)))), 2)

    return output
