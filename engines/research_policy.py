"""
engines/research_policy.py — Web research trigger and scoring policy
====================================================================

Policy-fixed rules:
  evidence_score = coverage(0~40) + freshness(0~25) + source_trust(0~25) - contradiction_penalty(0~15)

Coverage:
  - 4 high-impact buckets: earnings, macro, ownership, valuation
  - each bucket with at least one evidence item gives +10 (max 40)

Freshness:
  - Per evidence age days score:
      <=1d:25, <=3d:20, <=7d:15, <=14d:10, <=30d:5, >30d:0
  - freshness score = average of per-item scores (rounded), max 25

Source trust:
  - tier average mapped to 25-point scale
  - tier values:
      official/regulatory/IR: 1.0
      wire: 0.8
      general news: 0.6
      other: 0.4
  - source_trust = round(25 * avg_tier), max 25

Contradiction penalty:
  - group evidence by (ticker, kind)
  - if both positive and negative contradiction keywords co-exist in a group:
      penalty +5 per group, capped at 15
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

HIGH_IMPACT_BUCKETS = ("earnings", "macro", "ownership", "valuation")

POSITIVE_KWS = {
    "beat", "beats", "upgrade", "upgrades", "buy", "accumulate", "bullish", "reaffirm",
    "strong", "growth", "expansion", "raise", "raised",
}
NEGATIVE_KWS = {
    "miss", "misses", "downgrade", "downgrades", "sell", "bearish", "weak", "fraud",
    "investigation", "delay", "cut", "cuts", "warn", "warning", "decline",
}

OFFICIAL_DOMAINS = (
    "sec.gov", "federalreserve.gov", "treasury.gov", "bea.gov", "bls.gov",
    "fred.stlouisfed.org", "nyfed.org", "eia.gov",
)
WIRE_DOMAINS = ("prnewswire.com", "businesswire.com", "globenewswire.com")
NEWS_DOMAINS = ("reuters.com", "bloomberg.com", "wsj.com", "cnbc.com", "newsapi.org")


def _iso_to_dt(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _domain(url: str | None) -> str:
    if not url:
        return ""
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def _domain_in(host: str, allowed: tuple[str, ...]) -> bool:
    return any(host == d or host.endswith("." + d) for d in allowed)


def _kind_to_bucket(kind: str) -> str | None:
    k = (kind or "").lower()
    if k in ("press_release_or_ir", "sec_filing", "sec_8k", "catalyst_event_detail"):
        return "earnings"
    if k in ("macro_headline_context", "macro_release"):
        return "macro"
    if k in ("ownership_identity", "insider_institutional"):
        return "ownership"
    if k in ("valuation_context", "peers_context"):
        return "valuation"
    if k in HIGH_IMPACT_BUCKETS:
        return k
    return None


def _trust_tier(item: dict[str, Any]) -> float:
    tier = item.get("trust_tier")
    if isinstance(tier, (int, float)):
        return float(max(0.0, min(1.0, tier)))

    source = str(item.get("source", "")).lower()
    host = _domain(item.get("url"))
    if _domain_in(host, OFFICIAL_DOMAINS) or "investor relations" in source or "ir" in source:
        return 1.0
    if _domain_in(host, WIRE_DOMAINS):
        return 0.8
    if _domain_in(host, NEWS_DOMAINS):
        return 0.6
    return 0.4


def _freshness_points(age_days: float) -> int:
    if age_days <= 1:
        return 25
    if age_days <= 3:
        return 20
    if age_days <= 7:
        return 15
    if age_days <= 14:
        return 10
    if age_days <= 30:
        return 5
    return 0


def compute_contradiction_penalty(items: list[dict[str, Any]]) -> int:
    """
    Same (ticker, kind) group containing both positive and negative cues => +5.
    Penalty capped at 15.
    """
    groups: dict[tuple[str, str], str] = {}
    for item in items or []:
        key = (str(item.get("ticker", "")), str(item.get("kind", "")))
        text = " ".join([
            str(item.get("title", "")),
            str(item.get("snippet", "")),
            str(item.get("source", "")),
        ]).lower()
        groups[key] = groups.get(key, "") + " " + text

    penalty = 0
    for text in groups.values():
        has_pos = any(k in text for k in POSITIVE_KWS)
        has_neg = any(k in text for k in NEGATIVE_KWS)
        if has_pos and has_neg:
            penalty += 5
    return min(15, penalty)


def compute_evidence_score(
    evidence_store: dict[str, dict[str, Any]],
    as_of: str,
    required_buckets: list[str] | None = None,
) -> dict[str, int]:
    """
    Fixed score formula:
      score = coverage + freshness + source_trust - contradiction_penalty
    """
    items = list((evidence_store or {}).values())
    buckets = required_buckets or list(HIGH_IMPACT_BUCKETS)
    bucket_hit = {b: False for b in buckets}

    for item in items:
        b = _kind_to_bucket(str(item.get("kind", "")))
        if b in bucket_hit:
            bucket_hit[b] = True
    coverage = sum(10 for hit in bucket_hit.values() if hit)

    as_of_dt = _iso_to_dt(as_of) or datetime.now(timezone.utc)
    freshness_scores: list[int] = []
    trust_scores: list[float] = []
    for item in items:
        pub_dt = _iso_to_dt(item.get("published_at")) or _iso_to_dt(item.get("retrieved_at"))
        if pub_dt is not None:
            age_days = max(0.0, (as_of_dt - pub_dt).total_seconds() / 86400.0)
            freshness_scores.append(_freshness_points(age_days))
        trust_scores.append(_trust_tier(item))

    freshness = round(sum(freshness_scores) / len(freshness_scores)) if freshness_scores else 0
    freshness = int(max(0, min(25, freshness)))

    trust_avg = (sum(trust_scores) / len(trust_scores)) if trust_scores else 0.0
    source_trust = int(max(0, min(25, round(25 * trust_avg))))

    contradiction_penalty = compute_contradiction_penalty(items)
    score = int(max(0, min(100, coverage + freshness + source_trust - contradiction_penalty)))
    return {
        "score": score,
        "coverage": int(coverage),
        "freshness": int(freshness),
        "source_trust": int(source_trust),
        "contradiction_penalty": int(contradiction_penalty),
    }


def compute_impact_score(
    desk_outputs: dict[str, dict[str, Any]] | None = None,
    evidence_requests: list[dict[str, Any]] | None = None,
) -> int:
    """
    Impact score (0~4):
      +2: earnings/catalyst request exists
      +1: macro shift/tail warning context request exists
      +1: ownership/valuation context request exists
    """
    desk_outputs = desk_outputs or {}
    reqs = evidence_requests or []
    kinds = {str(r.get("kind", "")) for r in reqs}
    score = 0

    if {"press_release_or_ir", "catalyst_event_detail", "sec_filing"} & kinds:
        score += 2
    if "macro_headline_context" in kinds:
        score += 1
    if {"ownership_identity", "valuation_context"} & kinds:
        score += 1

    macro = desk_outputs.get("macro", {})
    if macro.get("tail_risk_warning"):
        score = max(score, 3)
    return min(4, score)


def compute_uncertainty_score(
    desk_outputs: dict[str, dict[str, Any]] | None = None,
    evidence_requests: list[dict[str, Any]] | None = None,
) -> int:
    """
    Uncertainty score (0~4):
      +1 each desk with needs_more_data
      +1 if low confidence desk exists (<0.45)
      +1 if high missing_pct desk exists (>0.30)
    """
    desk_outputs = desk_outputs or {}
    reqs = evidence_requests or []

    score = 0
    for desk in ("macro", "fundamental", "sentiment"):
        out = desk_outputs.get(desk, {})
        if out.get("needs_more_data"):
            score += 1
    if any((desk_outputs.get(d, {}).get("confidence", 0.5) < 0.45) for d in ("macro", "fundamental", "sentiment")):
        score += 1
    if any((desk_outputs.get(d, {}).get("data_quality", {}).get("missing_pct", 0.0) > 0.30) for d in ("macro", "fundamental", "sentiment")):
        score += 1
    if reqs:
        score += 1
    return min(4, score)


def _is_user_direct_research_question(user_request: str) -> bool:
    text = (user_request or "").lower()
    keys = ("촉매", "이벤트", "누가 샀", "누가 샀는지", "who bought", "catalyst", "event", "insider")
    return any(k in text for k in keys)


def _has_high_impact_missing_fields(evidence_requests: list[dict[str, Any]]) -> bool:
    for req in evidence_requests or []:
        kind = str(req.get("kind", ""))
        if _kind_to_bucket(kind) is None:
            continue
        if not req.get("ticker") and not req.get("series_id"):
            return True
        if not req.get("query"):
            return True
    return False


def cap_requests_by_budget(
    requests: list[dict[str, Any]],
    *,
    queries_used_total: int,
    queries_used_by_ticker: dict[str, int] | None = None,
    max_web_queries_per_run: int = 6,
    max_web_queries_per_ticker: int = 3,
) -> list[dict[str, Any]]:
    """
    Priority-order request cap:
      - run budget max_web_queries_per_run
      - ticker budget max_web_queries_per_ticker
    """
    used_by_ticker = dict(queries_used_by_ticker or {})
    out: list[dict[str, Any]] = []
    remaining = max(0, max_web_queries_per_run - max(0, queries_used_total))
    if remaining <= 0:
        return out

    ordered = sorted(requests or [], key=lambda r: int(r.get("priority", 5)))
    for req in ordered:
        if len(out) >= remaining:
            break
        ticker = str(req.get("ticker", "")).upper() or "__GLOBAL__"
        if used_by_ticker.get(ticker, 0) >= max_web_queries_per_ticker:
            continue
        out.append(req)
        used_by_ticker[ticker] = used_by_ticker.get(ticker, 0) + 1
    return out


def should_run_web_research(
    *,
    state: dict[str, Any],
    desk_outputs: dict[str, dict[str, Any]] | None = None,
    disagreement_score: float = 0.0,
    evidence_requests: list[dict[str, Any]] | None = None,
    user_request: str = "",
    max_web_queries_per_run: int = 6,
    max_web_queries_per_ticker: int = 3,
) -> dict[str, Any]:
    """
    Trigger OR conditions:
      - ResearchNeedScore >= 4
      - disagreement_score > 0.5
      - high-impact event flag exists and required fields are missing
      - user directly asked catalyst/who bought/event

    Break conditions:
      - research_round >= max_research_rounds
      - evidence_score >= 75
      - last_research_delta < 2 (after at least one research round)
      - run budget exhausted
    """
    desk_outputs = desk_outputs or {}
    reqs = evidence_requests or []
    audit = (state.get("audit") or {}).get("research", {})
    used_total = int(audit.get("web_queries_total", 0))
    used_by_ticker = dict(audit.get("web_queries_by_ticker", {}))

    impact = compute_impact_score(desk_outputs=desk_outputs, evidence_requests=reqs)
    uncertainty = compute_uncertainty_score(desk_outputs=desk_outputs, evidence_requests=reqs)
    need_score = impact + uncertainty

    evidence_score = int(state.get("evidence_score", 0))
    research_round = int(state.get("research_round", 0))
    max_rounds = int(state.get("max_research_rounds", 2))
    last_delta = int(state.get("last_research_delta", 0))

    if research_round >= max_rounds:
        return {"run": False, "reason": "max_research_rounds", "impact_score": impact, "uncertainty_score": uncertainty, "research_need_score": need_score}
    if evidence_score >= 75:
        return {"run": False, "reason": "evidence_score_enough", "impact_score": impact, "uncertainty_score": uncertainty, "research_need_score": need_score}
    if research_round > 0 and last_delta < 2:
        return {"run": False, "reason": "low_added_evidence_delta", "impact_score": impact, "uncertainty_score": uncertainty, "research_need_score": need_score}
    if used_total >= max_web_queries_per_run:
        return {"run": False, "reason": "run_budget_exhausted", "impact_score": impact, "uncertainty_score": uncertainty, "research_need_score": need_score}

    trigger_need = need_score >= 4
    trigger_disagreement = disagreement_score > 0.5
    trigger_missing = _has_high_impact_missing_fields(reqs)
    trigger_user = _is_user_direct_research_question(user_request)

    allowed_requests = cap_requests_by_budget(
        reqs,
        queries_used_total=used_total,
        queries_used_by_ticker=used_by_ticker,
        max_web_queries_per_run=max_web_queries_per_run,
        max_web_queries_per_ticker=max_web_queries_per_ticker,
    )
    run = bool(trigger_need or trigger_disagreement or trigger_missing or trigger_user) and bool(allowed_requests)

    reasons = []
    if trigger_need:
        reasons.append("research_need_score")
    if trigger_disagreement:
        reasons.append("disagreement")
    if trigger_missing:
        reasons.append("high_impact_missing_fields")
    if trigger_user:
        reasons.append("user_direct_research_question")
    if not allowed_requests and reqs:
        reasons.append("budget_cap")

    return {
        "run": run,
        "reason": ",".join(reasons) if reasons else "no_trigger",
        "impact_score": impact,
        "uncertainty_score": uncertainty,
        "research_need_score": need_score,
        "allowed_requests": allowed_requests,
        "max_web_queries_per_run": max_web_queries_per_run,
        "max_web_queries_per_ticker": max_web_queries_per_ticker,
    }

