"""
agents/autonomy_planner.py
==========================
Autonomous runtime recovery planner.
- Detects runtime issues from desk limitations / risk metadata.
- Produces recovery actions + extra evidence requests.
- LLM-first when available, deterministic fallback otherwise.
"""

from __future__ import annotations

import json
from typing import Any

from llm.router import get_llm

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    HAS_LC = True
except ImportError:
    HAS_LC = False


_ALLOWED_ACTION_TYPES = {
    "run_research",
    "rerun_desk",
    "adjust_risk",
    "add_hedge",
    "provider_fallback",
    "degrade_mode",
}
_DEFAULT_KIND_BY_DESK = {
    "macro": "macro_headline_context",
    "fundamental": "valuation_context",
    "sentiment": "catalyst_event_detail",
    "quant": "web_search",
    "risk": "macro_headline_context",
}


def _request_key(req: dict[str, Any]) -> tuple:
    return (
        req.get("desk", ""),
        req.get("kind", ""),
        req.get("ticker", ""),
        req.get("series_id", ""),
        req.get("query", ""),
    )


def _merge_requests(*request_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen = set()
    for reqs in request_lists:
        for req in reqs or []:
            if not isinstance(req, dict):
                continue
            k = _request_key(req)
            if k in seen:
                continue
            seen.add(k)
            out.append(req)
    return out


def _trim(s: Any, n: int = 220) -> str:
    text = str(s or "").strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _extract_runtime_issues(state: dict[str, Any], desk_outputs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    for desk in ("macro", "fundamental", "sentiment", "quant"):
        out = desk_outputs.get(desk, {}) or {}
        confidence = float(out.get("confidence", 0.5) or 0.5)
        dqm = out.get("data_quality", {}) or {}
        missing_pct = float(dqm.get("missing_pct", 0.0) or 0.0)
        needs_more_data = bool(out.get("needs_more_data", False))
        open_questions = out.get("open_questions", [])

        for lim in out.get("limitations", []) or []:
            text = _trim(lim, 300)
            lower = text.lower()
            code = "runtime_limitation"
            severity = "medium"

            if "fmp" in lower and ("403" in lower or "endpoint unavailable" in lower):
                code = "fmp_endpoint_restricted"
            elif "newsapi" in lower and "426" in lower:
                code = "newsapi_upgrade_required"
            elif any(k in lower for k in ("timeout", "timed out", "dns", "connection", "429", "503", "504")):
                code = "provider_runtime_error"
            elif "insufficient" in lower or "missing" in lower:
                code = "data_gap"
                severity = "low"

            issues.append(
                {
                    "code": code,
                    "severity": severity,
                    "desk": desk,
                    "detail": text,
                }
            )

        if needs_more_data:
            issues.append(
                {
                    "code": "insight_gap",
                    "severity": "low",
                    "desk": desk,
                    "detail": "Desk marked needs_more_data=True",
                }
            )
        if missing_pct > 0.30:
            issues.append(
                {
                    "code": "missing_data_high",
                    "severity": "medium",
                    "desk": desk,
                    "detail": f"data_quality.missing_pct={missing_pct:.2f}",
                }
            )
        if confidence < 0.45:
            issues.append(
                {
                    "code": "low_confidence",
                    "severity": "low",
                    "desk": desk,
                    "detail": f"confidence={confidence:.2f}",
                }
            )
        if isinstance(open_questions, list) and open_questions:
            issues.append(
                {
                    "code": "unresolved_questions",
                    "severity": "low",
                    "desk": desk,
                    "detail": f"open_questions={len(open_questions)}",
                }
            )

    risk_decision = (state.get("risk_assessment") or {}).get("risk_decision", {})
    enrich_status = str(risk_decision.get("_llm_enrichment_status", "")).strip()
    if enrich_status.startswith("failed"):
        issues.append(
            {
                "code": "risk_llm_enrichment_failed",
                "severity": "low",
                "desk": "risk",
                "detail": _trim(risk_decision.get("_llm_enrichment_error", enrich_status), 280),
            }
        )
    evidence_score = int(state.get("evidence_score", 0) or 0)
    if evidence_score < 55:
        issues.append(
            {
                "code": "low_evidence_score",
                "severity": "low",
                "desk": "orchestrator",
                "detail": f"evidence_score={evidence_score}",
            }
        )

    # dedupe by (code, desk, detail)
    out: list[dict[str, Any]] = []
    seen = set()
    for i in issues:
        k = (i.get("code"), i.get("desk"), i.get("detail"))
        if k in seen:
            continue
        seen.add(k)
        out.append(i)
    return out


def _fallback_plan(state: dict[str, Any], issues: list[dict[str, Any]]) -> dict[str, Any]:
    ticker = state.get("target_ticker", "AAPL")

    actions: list[dict[str, Any]] = []
    reqs: list[dict[str, Any]] = []
    notes: list[str] = []

    codes = {i.get("code") for i in issues}

    if "fmp_endpoint_restricted" in codes:
        actions.append(
            {
                "type": "provider_fallback",
                "detail": "FMP 제한 엔드포인트를 우회하고 SEC/IR/valuation evidence로 보강",
                "params": {"provider": "fmp", "mode": "partial_success"},
            }
        )
        reqs.extend(
            [
                {
                    "desk": "fundamental",
                    "kind": "sec_filing",
                    "ticker": ticker,
                    "query": f"{ticker} latest 10-Q 10-K key metrics",
                    "priority": 2,
                    "recency_days": 365,
                    "max_items": 3,
                    "rationale": "fmp_endpoint_restricted_backfill",
                },
                {
                    "desk": "fundamental",
                    "kind": "valuation_context",
                    "ticker": ticker,
                    "query": f"{ticker} valuation peers historical context",
                    "priority": 2,
                    "recency_days": 90,
                    "max_items": 3,
                    "rationale": "fmp_endpoint_restricted_backfill",
                },
            ]
        )
        notes.append("FMP 제한 감지: structured/SEC 근거로 valuation 갭 보강")

    if "newsapi_upgrade_required" in codes:
        actions.append(
            {
                "type": "provider_fallback",
                "detail": "NewsAPI 제한 감지: official/filing 경로 우선으로 전환",
                "params": {"provider": "newsapi", "mode": "official_first"},
            }
        )
        reqs.extend(
            [
                {
                    "desk": "sentiment",
                    "kind": "press_release_or_ir",
                    "ticker": ticker,
                    "query": f"{ticker} investor relations announcement",
                    "priority": 2,
                    "recency_days": 30,
                    "max_items": 3,
                    "rationale": "newsapi_upgrade_required_backfill",
                }
            ]
        )
        notes.append("NewsAPI 제한 감지: official release/SEC 경로로 대체")

    if "risk_llm_enrichment_failed" in codes:
        actions.append(
            {
                "type": "adjust_risk",
                "detail": "Risk narrative enrichment 실패 시 Python 결정을 유지하고 compact enrichment 재시도",
                "params": {"mode": "compact_enrichment"},
            }
        )
        notes.append("Risk LLM 오류는 의사결정에 영향 없이 서사만 compact 재시도")

    if "provider_runtime_error" in codes:
        actions.append(
            {
                "type": "provider_fallback",
                "detail": "외부 API 런타임 오류 감지: 공식 소스/캐시 기반 경로 우선",
                "params": {"mode": "official_or_cached_first"},
            }
        )
        notes.append("런타임 연결 오류 감지: fallback 경로로 자동 전환")

    for issue in issues:
        code = str(issue.get("code", ""))
        desk = str(issue.get("desk", "orchestrator"))
        if code not in {"insight_gap", "missing_data_high", "unresolved_questions", "low_confidence", "low_evidence_score"}:
            continue
        kind = _DEFAULT_KIND_BY_DESK.get(desk, "web_search")
        detail = _trim(issue.get("detail"), 120)
        query = f"{ticker} {desk} {kind} {detail}".strip()
        reqs.append(
            {
                "desk": desk if desk in ("macro", "fundamental", "sentiment") else "orchestrator",
                "kind": kind,
                "ticker": ticker,
                "query": query,
                "priority": 3 if code != "missing_data_high" else 2,
                "recency_days": 30 if code != "unresolved_questions" else 14,
                "max_items": 3,
                "rationale": f"autonomy_{code}",
            }
        )
        if desk in ("macro", "fundamental", "sentiment", "quant"):
            actions.append(
                {
                    "type": "rerun_desk",
                    "detail": f"{desk} desk rerun for {code}",
                    "params": {"desk": desk},
                }
            )

    if "insight_gap" in codes or "unresolved_questions" in codes:
        actions.append(
            {
                "type": "run_research",
                "detail": "open_questions 기반 추가 근거 수집 실행",
                "params": {"source": "autonomy_planner"},
            }
        )

    if not actions:
        notes.append("No critical runtime issue detected")

    return {
        "issues": issues,
        "actions": actions[:5],
        "evidence_requests": _merge_requests(reqs)[:5],
        "notes": notes[:3],
    }


def _safe_json_obj(text: str) -> dict[str, Any] | None:
    s = (text or "").strip()
    if not s:
        return None
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _sanitize_action(action: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(action, dict):
        return None
    action_type = str(action.get("type", "")).strip()
    detail = _trim(action.get("detail"), 180)
    params = action.get("params") if isinstance(action.get("params"), dict) else {}
    if action_type not in _ALLOWED_ACTION_TYPES or not detail:
        return None
    return {"type": action_type, "detail": detail, "params": params}


def _sanitize_request(req: dict[str, Any], default_ticker: str) -> dict[str, Any] | None:
    if not isinstance(req, dict):
        return None
    kind = _trim(req.get("kind"), 64)
    if not kind:
        return None
    query = _trim(req.get("query"), 180)
    ticker = _trim(req.get("ticker") or default_ticker, 16).upper()
    if not (query or ticker or req.get("series_id")):
        return None
    try:
        priority = int(req.get("priority", 3))
    except (TypeError, ValueError):
        priority = 3
    try:
        recency = int(req.get("recency_days", 30))
    except (TypeError, ValueError):
        recency = 30
    try:
        max_items = int(req.get("max_items", 3))
    except (TypeError, ValueError):
        max_items = 3
    return {
        "desk": str(req.get("desk") or "orchestrator"),
        "kind": kind,
        "ticker": ticker,
        "series_id": _trim(req.get("series_id"), 64) or None,
        "query": query or None,
        "priority": max(1, min(5, priority)),
        "recency_days": max(1, min(365, recency)),
        "max_items": max(1, min(20, max_items)),
        "rationale": _trim(req.get("rationale"), 180),
    }


def plan_runtime_recovery(
    state: dict[str, Any],
    desk_outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    issues = _extract_runtime_issues(state, desk_outputs)
    base = _fallback_plan(state, issues)

    if not HAS_LC:
        return base

    llm = get_llm("orchestrator")
    if llm is None:
        return base

    try:
        prompt = {
            "task": "runtime_recovery_planner",
            "rules": [
                "Return JSON object only.",
                "Do not include chain-of-thought.",
                "actions <= 5, evidence_requests <= 3",
                "Prefer official/filing paths when data provider limits are detected.",
            ],
            "user_request": state.get("user_request", ""),
            "target_ticker": state.get("target_ticker", ""),
            "issues": issues,
            "existing_backlog": state.get("task_backlog", [])[:10],
        }
        msgs = [
            SystemMessage(content="You are an autonomous recovery planner. Output JSON only."),
            HumanMessage(content=json.dumps(prompt, ensure_ascii=False)),
        ]
        resp = llm.invoke(msgs)
        obj = _safe_json_obj(getattr(resp, "content", ""))
        if not obj:
            return base

        actions = []
        for a in obj.get("actions", [])[:5]:
            sa = _sanitize_action(a)
            if sa:
                actions.append(sa)

        reqs = []
        for r in obj.get("evidence_requests", [])[:3]:
            sr = _sanitize_request(r, default_ticker=str(state.get("target_ticker", "")))
            if sr:
                reqs.append(sr)

        merged_actions = base["actions"] + [a for a in actions if a not in base["actions"]]
        merged_reqs = _merge_requests(base["evidence_requests"], reqs)
        notes = base.get("notes", [])
        if obj.get("notes") and isinstance(obj["notes"], list):
            notes = (notes + [_trim(x, 140) for x in obj["notes"] if str(x).strip()])[:3]

        return {
            "issues": issues,
            "actions": merged_actions[:5],
            "evidence_requests": merged_reqs[:5],
            "notes": notes[:3],
        }
    except Exception:
        return base
