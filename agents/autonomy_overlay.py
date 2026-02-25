"""
agents/autonomy_overlay.py
==========================
LLM overlay layer for desk autonomy (macro/fundamental/sentiment).
- Deterministic outputs stay authoritative.
- LLM may only return JSON patch for narrative/research-planning fields.
- Any invalid/unsafe patch is ignored (no-op fallback).
"""

from __future__ import annotations

import json
import re
from typing import Any

from llm.router import get_llm

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    HAS_LC = True
except ImportError:
    HAS_LC = False


MAX_ITEMS = 5
MAX_STR_LEN = 220
MAX_REACT_LEN = 120

_ALLOWED_PATCH_FIELDS = {
    "key_drivers",
    "what_to_watch",
    "scenario_notes",
    "open_questions",
    "decision_sensitivity",
    "followups",
    "evidence_requests",
    "react_trace",
}
_ALLOWED_FOLLOWUP_TYPES = {"run_research", "rerun_desk", "adjust_risk", "add_hedge"}
_ALLOWED_IMPACTS = {"high", "medium", "low"}
_ALLOWED_PHASES = {"THOUGHT", "ACTION", "OBSERVATION", "REFLECTION"}
_DEFAULT_KIND_BY_DESK = {
    "macro": "macro_headline_context",
    "fundamental": "valuation_context",
    "sentiment": "catalyst_event_detail",
}

_SYSTEM_PROMPT = """
You are a hedge-fund desk analyst overlay.
You must internally run ReAct phases (THOUGHT/ACTION/OBSERVATION/REFLECTION),
but you MUST NEVER output chain-of-thought.

Output rules:
- Return JSON object patch ONLY. No prose, no markdown.
- Allowed keys only:
  key_drivers, what_to_watch, scenario_notes,
  open_questions, decision_sensitivity, followups, evidence_requests, react_trace.
- Keep lists concise (max 5 items), strings <= 220 chars.
- evidence_requests max 3 items.
- Do not modify engine outputs, risk_flags, evidence numeric values, or tilt_factor.
- open_questions, decision_sensitivity, followups should be non-empty whenever possible.

ReAct semantics to encode in structured fields:
1) THOUGHT (internal): identify top uncertainty gaps that can change decision.
2) ACTION: produce open_questions (+kind), optional evidence_requests, and followups.
3) OBSERVATION: if evidence_digest exists, reflect updates in narrative fields.
4) REFLECTION: update decision_sensitivity with conditional robustness statements.
""".strip()


def _truncate_text(value: Any, max_len: int = MAX_STR_LEN) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def clamp_list_len(items: list[Any] | None, max_len: int = MAX_ITEMS) -> list[Any]:
    if not isinstance(items, list):
        return []
    return items[:max_len]


def safe_json_loads(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    text = raw if isinstance(raw, str) else getattr(raw, "content", "")
    if not isinstance(text, str):
        return None

    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"```$", "", s).strip()

    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None


def _sanitize_string_list(values: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(values, list):
        return out
    for v in values[:MAX_ITEMS]:
        s = _truncate_text(v)
        if s:
            out.append(s)
    return out


def validate_evidence_requests(
    values: Any,
    *,
    desk: str,
    ticker: str,
) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []

    out: list[dict[str, Any]] = []
    for item in values[:3]:
        if not isinstance(item, dict):
            continue
        kind = _truncate_text(item.get("kind"), 64)
        if not kind:
            continue
        query = _truncate_text(item.get("query"), 200) if item.get("query") else ""
        req_ticker = _truncate_text(item.get("ticker") or ticker, 16).upper()
        series_id = _truncate_text(item.get("series_id"), 64) if item.get("series_id") else ""
        if not (query or req_ticker or series_id):
            continue

        try:
            priority = int(item.get("priority", 3))
        except (TypeError, ValueError):
            priority = 3
        try:
            recency_days = int(item.get("recency_days", 30))
        except (TypeError, ValueError):
            recency_days = 30
        try:
            max_items = int(item.get("max_items", 5))
        except (TypeError, ValueError):
            max_items = 5

        req = {
            "desk": desk,
            "kind": kind,
            "ticker": req_ticker or None,
            "series_id": series_id or None,
            "query": query or None,
            "priority": max(1, min(5, priority)),
            "recency_days": max(1, min(365, recency_days)),
            "max_items": max(1, min(20, max_items)),
            "rationale": _truncate_text(item.get("rationale"), 180),
        }
        out.append(req)
    return out


def ensure_schema(
    patch: dict[str, Any] | None,
    *,
    desk: str,
    ticker: str,
) -> dict[str, Any]:
    if not isinstance(patch, dict):
        return {}

    sanitized: dict[str, Any] = {}
    for key in patch.keys():
        if key not in _ALLOWED_PATCH_FIELDS:
            continue

        if key in ("key_drivers", "what_to_watch"):
            sanitized[key] = _sanitize_string_list(patch.get(key))

        elif key == "scenario_notes":
            v = patch.get("scenario_notes")
            if isinstance(v, dict):
                sn: dict[str, str] = {}
                for k in ("bull", "base", "bear"):
                    if k in v:
                        sn[k] = _truncate_text(v.get(k))
                if sn:
                    sanitized[key] = sn

        elif key == "open_questions":
            oq_out: list[dict[str, Any]] = []
            for it in clamp_list_len(patch.get("open_questions")):
                if not isinstance(it, dict):
                    continue
                q = _truncate_text(it.get("q"), 180)
                why = _truncate_text(it.get("why"), 180)
                kind = _truncate_text(it.get("kind"), 64) or _DEFAULT_KIND_BY_DESK.get(desk, "web_search")
                if not q:
                    continue
                try:
                    pr = int(it.get("priority", 3))
                except (TypeError, ValueError):
                    pr = 3
                try:
                    rd = int(it.get("recency_days", 30))
                except (TypeError, ValueError):
                    rd = 30
                oq_out.append({
                    "q": q,
                    "why": why or "Decision-relevant uncertainty",
                    "kind": kind,
                    "priority": max(1, min(5, pr)),
                    "recency_days": max(1, min(365, rd)),
                })
            if oq_out:
                sanitized[key] = oq_out

        elif key == "decision_sensitivity":
            ds_out: list[dict[str, Any]] = []
            for it in clamp_list_len(patch.get("decision_sensitivity")):
                if not isinstance(it, dict):
                    continue
                cond = _truncate_text(it.get("if"), 160)
                change = _truncate_text(it.get("then_change"), 180)
                impact = str(it.get("impact", "medium")).lower()
                if cond and change:
                    ds_out.append({
                        "if": cond,
                        "then_change": change,
                        "impact": impact if impact in _ALLOWED_IMPACTS else "medium",
                    })
            if ds_out:
                sanitized[key] = ds_out

        elif key == "followups":
            fu_out: list[dict[str, Any]] = []
            for it in clamp_list_len(patch.get("followups")):
                if not isinstance(it, dict):
                    continue
                f_type = str(it.get("type", "")).strip()
                detail = _truncate_text(it.get("detail"), 180)
                params = it.get("params") if isinstance(it.get("params"), dict) else {}
                if f_type in _ALLOWED_FOLLOWUP_TYPES and detail:
                    fu_out.append({
                        "type": f_type,
                        "detail": detail,
                        "params": params,
                    })
            if fu_out:
                sanitized[key] = fu_out

        elif key == "evidence_requests":
            reqs = validate_evidence_requests(patch.get("evidence_requests"), desk=desk, ticker=ticker)
            if reqs:
                sanitized[key] = reqs

        elif key == "react_trace":
            rt_out: list[dict[str, Any]] = []
            for it in clamp_list_len(patch.get("react_trace")):
                if not isinstance(it, dict):
                    continue
                phase = str(it.get("phase", "")).upper().strip()
                summary = _truncate_text(it.get("summary"), MAX_REACT_LEN)
                if phase in _ALLOWED_PHASES and summary:
                    rt_out.append({"phase": phase, "summary": summary})
            if rt_out:
                sanitized[key] = rt_out

    return sanitized


def _build_human_payload(
    *,
    desk: str,
    output: dict[str, Any],
    state: dict[str, Any] | None,
    focus_areas: list[str] | None,
    evidence_digest: list[dict[str, Any]] | None,
) -> str:
    compact_output = {
        "ticker": output.get("ticker"),
        "horizon_days": output.get("horizon_days"),
        "primary_decision": output.get("primary_decision"),
        "confidence": output.get("confidence"),
        "needs_more_data": output.get("needs_more_data"),
        "data_quality": output.get("data_quality"),
        "key_drivers": output.get("key_drivers", [])[:5],
        "what_to_watch": output.get("what_to_watch", [])[:5],
        "scenario_notes": output.get("scenario_notes", {}),
        "open_questions": output.get("open_questions", [])[:5],
        "decision_sensitivity": output.get("decision_sensitivity", [])[:5],
        "followups": output.get("followups", [])[:5],
        "evidence_requests": output.get("evidence_requests", [])[:5],
    }
    payload = {
        "desk": desk,
        "user_request": (state or {}).get("user_request", ""),
        "focus_areas": focus_areas or [],
        "evidence_digest": (evidence_digest or [])[:7],
        "deterministic_output": compact_output,
        "instruction": "Return JSON patch object only.",
    }
    return json.dumps(payload, ensure_ascii=False)


def _apply_overlay(
    desk: str,
    output: dict[str, Any],
    state: dict[str, Any] | None,
    focus_areas: list[str] | None,
    evidence_digest: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if not HAS_LC:
        return {}

    llm = get_llm(desk)
    if llm is None:
        return {}

    try:
        msgs = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=_build_human_payload(
                desk=desk,
                output=output,
                state=state,
                focus_areas=focus_areas,
                evidence_digest=evidence_digest,
            )),
        ]
        resp = llm.invoke(msgs)
        patch = safe_json_loads(getattr(resp, "content", resp))
        return ensure_schema(patch, desk=desk, ticker=str(output.get("ticker", "")))
    except Exception:
        return {}


def apply_llm_overlay_macro(
    output: dict[str, Any],
    state: dict[str, Any] | None,
    focus_areas: list[str] | None,
    evidence_digest: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    return _apply_overlay("macro", output, state, focus_areas, evidence_digest)


def apply_llm_overlay_fundamental(
    output: dict[str, Any],
    state: dict[str, Any] | None,
    focus_areas: list[str] | None,
    evidence_digest: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    return _apply_overlay("fundamental", output, state, focus_areas, evidence_digest)


def apply_llm_overlay_sentiment(
    output: dict[str, Any],
    state: dict[str, Any] | None,
    focus_areas: list[str] | None,
    evidence_digest: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    return _apply_overlay("sentiment", output, state, focus_areas, evidence_digest)
