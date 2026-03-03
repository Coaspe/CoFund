"""
investment_team.py — 7-Agent AI Investment Team (V3 — Final Integration)
========================================================================
CHANGELOG:
  v3.0 (2026-02-22) — Final Integration.
    - New directory structure: engines/, agents/, data_providers/, schemas/
    - 4 desk agents (Macro, Fundamental, Sentiment, Quant) with engine isolation
    - barrier_node for fan-in (risk_manager 1회 실행 보장)
    - Telemetry hooks on all nodes
    - Conditional routing: orchestrator → requested desks only
    - Risk Manager feedback loop with MAX_ITERATIONS cap

실행:
    python investment_team.py
    python investment_team.py --mode mock --seed 42

Iron Rules Enforced:
  R0: No Trading (no broker code)
  R1: Python-LLM Separation (engines compute, LLM interprets)
  R2: No Evidence, No Trade (all outputs have evidence[])
  R3: Quant Engine Isolation (no data fetching in engines/)
  R4: Risk 5-Gate Order Fixed (Gate1→2→3→4→5)
  R5: Sentiment Tilt Cap [0.7, 1.3]
  R6: Disagreement = Model Risk
  R7: Auditability (run_id, events.jsonl, final_state.json)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import numpy as np
from langgraph.graph import END, START, StateGraph

from schemas.common import (
    InvestmentState,
    create_initial_state,
    make_evidence,
    compute_disagreement_score,
)
import telemetry

# ── Data Providers ────────────────────────────────────────────────────────
from data_providers.data_hub import DataHub
from data_providers.web_research_provider import WebResearchProvider
from data_providers.sec_edgar_provider import SECEdgarProvider
from data_providers.perplexity_search_provider import PerplexitySearchProvider

# ── Agents ────────────────────────────────────────────────────────────────
from agents.macro_agent import macro_analyst_run
from agents.fundamental_agent import fundamental_analyst_run
from agents.sentiment_agent import sentiment_analyst_run
from agents.autonomy_planner import plan_runtime_recovery

# ── Engines (for quant) ──────────────────────────────────────────────────
from engines.quant_engine import generate_quant_payload, mock_quant_decision
from engines.research_policy import compute_evidence_score, should_run_web_research

# ── Existing agent imports (graceful) ─────────────────────────────────────
try:
    from agents.orchestrator_agent import (
        orchestrator_node as _orch_node_impl,
        plan_additional_research as _plan_research_impl,
    )
    HAS_ORCHESTRATOR = True
except ImportError:
    HAS_ORCHESTRATOR = False
    _plan_research_impl = None  # type: ignore

try:
    from agents.risk_agent import risk_manager_node as _risk_node_impl
except ImportError:
    _risk_node_impl = None  # type: ignore

try:
    from agents.report_agent import report_writer_node as _report_node_impl
except ImportError:
    _report_node_impl = None  # type: ignore


MAX_ITERATIONS = 3
MAX_WEB_QUERIES_PER_RUN = 6
MAX_WEB_QUERIES_PER_TICKER = 3
ALL_DESKS = ["macro", "fundamental", "sentiment", "quant"]
_BLOCKING_ISSUE_CODES = {
    "fmp_endpoint_restricted",
    "newsapi_upgrade_required",
    "provider_runtime_error",
}
_USER_ACTION_HINTS = {
    "fmp_endpoint_restricted": "FMP API plan/권한 확인 또는 대체 엔드포인트 키 활성화",
    "newsapi_upgrade_required": "NewsAPI 플랜 업그레이드 또는 everything 미사용 정책 고정",
    "provider_runtime_error": "DNS/방화벽/프록시와 API 상태를 점검",
    "risk_llm_enrichment_failed": "Risk LLM 모델 한도(TPM/컨텍스트) 상향 또는 모델 교체",
}

# ── Task key normalization map ────────────────────────────────────────────
_TASK_TO_DESK = {
    "macro_analysis": "macro", "macro": "macro",
    "fundamental_analysis": "fundamental", "fundamental": "fundamental",
    "sentiment_analysis": "sentiment", "sentiment": "sentiment",
    "technical_analysis": "quant", "quant": "quant",
}

_DESK_TO_STATE_KEY = {
    "macro": "macro_analysis",
    "fundamental": "fundamental_analysis",
    "sentiment": "sentiment_analysis",
    "quant": "technical_analysis",
}

_SWARM_REQUIRED_BUCKETS = ("earnings", "macro", "ownership", "valuation")
_SWARM_KIND_TO_BUCKET = {
    "press_release_or_ir": "earnings",
    "macro_headline_context": "macro",
    "ownership_identity": "ownership",
    "valuation_context": "valuation",
    "sec_filing": "other",
    "catalyst_event_detail": "other",
    "web_search": "other",
}
_SWARM_KIND_TO_IMPACTED = {
    "press_release_or_ir": ("fundamental", "sentiment"),
    "macro_headline_context": ("macro", "sentiment"),
    "ownership_identity": ("fundamental",),
    "valuation_context": ("fundamental",),
    "sec_filing": ("fundamental",),
    "catalyst_event_detail": ("sentiment", "fundamental"),
}
_MAX_SWARM_CANDIDATES = 20
_MAX_RERUN_DESKS = 2
_ETF_LIKE_TICKERS = {
    "SPY", "QQQ", "IWM", "DIA", "TLT", "SHY", "GLD", "SLV", "USO", "XLE", "XLF", "XLK",
    "XLV", "XLY", "XLI", "XLB", "XLP", "XLU", "XLC", "VNQ", "HYG", "LQD", "EEM", "EWJ",
}
_INDEX_LIKE_TICKERS = {"SPX", "NDX", "DJI", "RUT", "VIX"}
_URL_DROP_QUERY_KEYS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "ref", "ref_src", "source",
}
_LANDING_PATH_PATTERNS = (
    "/news",
    "/press-releases",
    "/pressreleases",
    "/newsevents/pressreleases",
    "/newsevents/pressreleases.htm",
)
_LANDING_TITLE_TOKENS = ("press releases", "newsroom", "news", "latest news")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Telemetry wrapper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _graph_trace_enabled() -> bool:
    return _env_flag("GRAPH_TRACE", _env_flag("ORCH_TRACE", False) or _env_flag("LLM_TRACE", False))


def _short_text(value: Any, max_len: int = 140) -> str:
    s = " ".join(str(value or "").split())
    if len(s) <= max_len:
        return s
    return s[: max_len - 3].rstrip() + "..."


def _normalize_query_text(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _detect_output_language(user_request: str) -> str:
    text = str(user_request or "")
    if re.search(r"[가-힣]", text):
        return "ko"
    return "en"


def _infer_intent_from_request(user_request: str, universe: list[str]) -> str:
    text = _normalize_query_text(user_request)
    if any(k in text for k in ("시장", "macro", "market outlook", "전망", "지정학", "event risk", "이벤트")):
        return "market_outlook"
    if any(k in text for k in ("헤지", "hedge")):
        return "hedge_design"
    if len(universe or []) >= 2:
        return "relative_value"
    return "single_name"


def _infer_asset_type(ticker: str) -> str:
    t = str(ticker or "").strip().upper()
    if not t:
        return "EQUITY"
    if t in _ETF_LIKE_TICKERS:
        return "ETF"
    if t in _INDEX_LIKE_TICKERS or t.startswith("^"):
        return "INDEX"
    if t in {"WTI", "BRENT", "CL", "GC", "SI"}:
        return "COMMODITY"
    if t in {"TLT", "IEF", "SHY", "BND"}:
        return "BOND"
    return "EQUITY"


def _build_asset_type_map(universe: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in universe or []:
        t = str(raw or "").strip().upper()
        if not t:
            continue
        out[t] = _infer_asset_type(t)
    return out


def _extract_universe_from_directives(directives: dict) -> list[str]:
    if not isinstance(directives, dict):
        return []
    brief = directives.get("investment_brief", {})
    if not isinstance(brief, dict):
        return []
    raw = brief.get("target_universe", [])
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen = set()
    for item in raw:
        t = str(item or "").strip().upper()
        if not t:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _canonicalize_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
    except ValueError:
        return raw
    scheme = (parsed.scheme or "https").lower()
    host = (parsed.hostname or "").lower()
    if not host:
        return raw
    path = parsed.path or "/"
    path = re.sub(r"/+", "/", path)
    if path != "/":
        path = path.rstrip("/")
    pairs = []
    for k, v in parse_qsl(parsed.query, keep_blank_values=False):
        key_l = str(k or "").strip().lower()
        if not key_l or key_l in _URL_DROP_QUERY_KEYS:
            continue
        pairs.append((k, v))
    query = urlencode(sorted(pairs), doseq=True)
    return urlunparse((scheme, host, path, "", query, ""))


def _evidence_store_key_from_url(url: str) -> str:
    canonical = _canonicalize_url(url)
    if not canonical:
        return ""
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _query_tokens(query: str) -> set[str]:
    stop = {
        "the", "and", "for", "with", "from", "latest", "market", "context", "news",
        "impact", "details", "detail", "price", "prices", "on", "of", "in", "to",
        "최근", "관련", "영향", "상세", "시장", "뉴스", "컨텍스트",
    }
    toks = set(re.findall(r"[a-zA-Z0-9가-힣]{3,}", _normalize_query_text(query)))
    return {t for t in toks if t not in stop}


def _is_landing_page_candidate(url: str, title: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    path = (parsed.path or "").lower().rstrip("/")
    if not path:
        return True
    if path in ("/", "/news", "/press-releases", "/newsevents", "/newsevents/pressreleases"):
        return True
    if any(path.endswith(pat) for pat in _LANDING_PATH_PATTERNS):
        return True
    title_l = str(title or "").strip().lower()
    return any(tok in title_l for tok in _LANDING_TITLE_TOKENS)


def _item_matches_request(item: dict, req: dict) -> bool:
    query = str(req.get("query", "")).strip()
    if not query:
        return True
    q_tokens = _query_tokens(query)
    if not q_tokens:
        return True
    hay = " ".join(
        [
            str(item.get("title", "")),
            str(item.get("snippet", "")),
            str(item.get("url", "")),
        ]
    ).lower()
    return any(tok in hay for tok in q_tokens)


def _sanitize_fundamental_requests_for_asset_type(
    requests: list[dict],
    *,
    ticker: str,
    asset_type: str,
) -> list[dict]:
    at = str(asset_type or "").upper()
    if at not in {"ETF", "INDEX"}:
        return list(requests or [])

    cleaned: list[dict] = []
    for req in requests or []:
        if not isinstance(req, dict):
            continue
        kind = str(req.get("kind", "")).strip().lower()
        query = _normalize_query_text(req.get("query", ""))
        if kind in {"ownership_identity", "sec_filing"}:
            continue
        if "insider" in query or "form 4" in query or "institutional ownership" in query:
            continue
        cleaned.append(req)

    if cleaned:
        return _dedupe_requests(cleaned)

    base = [
        {
            "desk": "fundamental",
            "kind": "valuation_context",
            "ticker": ticker,
            "query": f"{ticker} ETF holdings top 10 sector weights index valuation forward PE",
            "priority": 2,
            "recency_days": 30,
            "max_items": 5,
            "rationale": "etf_index_mode_holdings_sector_valuation",
        },
        {
            "desk": "fundamental",
            "kind": "web_search",
            "ticker": ticker,
            "query": f"{ticker} ETF flow creation redemption tracking error liquidity expense ratio",
            "priority": 2,
            "recency_days": 30,
            "max_items": 5,
            "rationale": "etf_index_mode_flow_liquidity",
        },
    ]
    return _dedupe_requests(base)


def _decision_snapshot_for_desk(desk: str, payload: dict | None) -> dict:
    out = payload if isinstance(payload, dict) else {}
    d = str(desk or "").strip().lower()
    if d == "macro":
        return {
            "primary_decision": out.get("primary_decision"),
            "macro_regime": out.get("macro_regime"),
        }
    if d == "fundamental":
        return {
            "primary_decision": out.get("primary_decision"),
            "recommendation": out.get("recommendation"),
            "analysis_mode": out.get("analysis_mode"),
        }
    if d == "sentiment":
        tilt = out.get("tilt_factor")
        if tilt is not None:
            try:
                tilt = round(float(tilt), 6)
            except (TypeError, ValueError):
                pass
        return {
            "primary_decision": out.get("primary_decision"),
            "sentiment_regime": out.get("sentiment_regime"),
            "tilt_factor": tilt,
        }
    if d == "quant":
        alloc = out.get("final_allocation_pct")
        if alloc is not None:
            try:
                alloc = round(float(alloc), 6)
            except (TypeError, ValueError):
                pass
        return {
            "decision": out.get("decision"),
            "final_allocation_pct": alloc,
            "analysis_mode": out.get("analysis_mode"),
        }
    return {}


def _decision_changed(prev: dict, curr: dict) -> bool:
    if not prev:
        return False
    keys = set(prev.keys()) | set(curr.keys())
    for k in keys:
        if prev.get(k) != curr.get(k):
            return True
    return False


def _collect_evidence_refs_for_desk(output: dict, state: dict, ticker: str, max_items: int = 3) -> list[dict]:
    refs: list[dict] = []
    seen = set()
    digest = output.get("evidence_digest", []) if isinstance(output, dict) else []
    if isinstance(digest, list):
        for item in digest:
            if not isinstance(item, dict):
                continue
            h = str(item.get("hash", "")).strip()
            url = str(item.get("url", "") or item.get("canonical_url", "")).strip()
            key = (h, url)
            if key in seen:
                continue
            seen.add(key)
            refs.append({"hash": h, "url": url})
            if len(refs) >= max_items:
                return refs

    store = state.get("evidence_store", {}) if isinstance(state, dict) else {}
    if isinstance(store, dict):
        for item in store.values():
            if not isinstance(item, dict):
                continue
            item_ticker = str(item.get("ticker", "")).strip().upper()
            if item_ticker and item_ticker != str(ticker or "").strip().upper():
                continue
            h = str(item.get("hash", "")).strip()
            url = str(item.get("url", "") or item.get("canonical_url", "")).strip()
            key = (h, url)
            if key in seen:
                continue
            seen.add(key)
            refs.append({"hash": h, "url": url})
            if len(refs) >= max_items:
                break
    return refs


def _attach_decision_change_log(
    *,
    desk: str,
    output: dict,
    state: dict,
    prev_output: dict | None,
    rerun_reason: str,
    ticker: str,
) -> dict:
    prev_snap = _decision_snapshot_for_desk(desk, prev_output)
    curr_snap = _decision_snapshot_for_desk(desk, output)
    was_rerun = "rerun selected" in str(rerun_reason or "").lower()
    changed = _decision_changed(prev_snap, curr_snap)
    evidence_refs = _collect_evidence_refs_for_desk(output, state, ticker)

    if not prev_snap:
        status = "initial_run_no_previous"
        reason = "이전 실행 결과가 없어 기준 비교가 불가능합니다."
    elif changed:
        status = "changed_with_evidence"
        reason = "신규 evidence 반영 후 핵심 판단 필드가 변경되었습니다."
    else:
        status = "unchanged_after_evidence_review" if was_rerun else "unchanged_without_rerun"
        reason = (
            "신규 evidence를 반영했지만 결론 민감도 기준에서 기존 판단을 유지했습니다."
            if was_rerun
            else "재실행 없이 기존 판단 체계를 유지했습니다."
        )

    output["decision_change_log"] = {
        "desk": desk,
        "was_rerun": was_rerun,
        "status": status,
        "changed": changed,
        "reason": reason,
        "previous_snapshot": prev_snap,
        "current_snapshot": curr_snap,
        "evidence_refs": evidence_refs,
    }
    return output


def _graph_trace(message: str) -> None:
    if _graph_trace_enabled():
        print(f"   [GRAPH TRACE] {message}", flush=True)


def _ops_log_console_enabled() -> bool:
    return _env_flag("OPS_LOG_CONSOLE", True)


def _ops_log_file_enabled() -> bool:
    return _env_flag("OPS_LOG_FILE", True)


def _ops_log(state: dict, node: str, title: str, lines: list[str]) -> None:
    clean_lines = [str(line).strip() for line in (lines or []) if str(line).strip()]
    if not clean_lines:
        return

    if _ops_log_console_enabled():
        print(f"   [OPS][{node}] {title}")
        for line in clean_lines:
            print(f"      - {line}")

    if not _ops_log_file_enabled():
        return

    run_id = str(state.get("run_id", "")).strip()
    if not run_id:
        return
    try:
        run_dir = Path("runs") / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "operator_timeline.log"
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with path.open("a", encoding="utf-8") as f:
            f.write(f"{ts} [{node}] {title}\n")
            for line in clean_lines:
                f.write(f"  - {line}\n")
    except OSError:
        pass


def _list_preview(values: Any, *, max_items: int = 3, max_len: int = 120) -> str:
    if not isinstance(values, list):
        return ""
    parts = [str(v).strip() for v in values[:max_items] if str(v).strip()]
    if not parts:
        return ""
    return _short_text("; ".join(parts), max_len=max_len)


def _rerun_reason_for_desk(state: dict, desk: str) -> str:
    research_round = int(state.get("research_round", 0))
    if research_round <= 0:
        return "initial delegation"
    rerun_plan = state.get("_rerun_plan", {}) if isinstance(state.get("_rerun_plan"), dict) else {}
    selected = set(rerun_plan.get("selected_desks", []) or [])
    if desk not in selected:
        return f"research phase (round={research_round})"
    reason_map = rerun_plan.get("reasons", {}) if isinstance(rerun_plan.get("reasons"), dict) else {}
    reasons = reason_map.get(desk, []) if isinstance(reason_map.get(desk, []), list) else []
    executed_kinds = rerun_plan.get("executed_kinds", []) if isinstance(rerun_plan.get("executed_kinds", []), list) else []
    reason_text = ",".join(str(x) for x in reasons[:3]) if reasons else "top-k rerun"
    kind_text = ",".join(str(k) for k in executed_kinds[:4]) if executed_kinds else "unknown"
    return f"rerun selected (kinds={kind_text}; {reason_text})"


def _react_trace_preview(output: dict) -> str:
    trace = output.get("react_trace", []) if isinstance(output, dict) else []
    if not isinstance(trace, list):
        return ""
    thought = ""
    action = ""
    for item in trace:
        if not isinstance(item, dict):
            continue
        phase = str(item.get("phase", "")).strip().upper()
        summary = str(item.get("summary", "")).strip()
        if phase == "THOUGHT" and summary and not thought:
            thought = summary
        elif phase == "ACTION" and summary and not action:
            action = summary
    parts = []
    if thought:
        parts.append(f"thought={thought}")
    if action:
        parts.append(f"action={action}")
    return _short_text(" | ".join(parts), max_len=180)


def _write_operator_summary(run_id: str) -> Path | None:
    run = str(run_id or "").strip()
    if not run:
        return None
    run_dir = Path("runs") / run
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return None

    interesting = {
        "orchestrator",
        "macro",
        "fundamental",
        "sentiment",
        "quant",
        "research_router",
        "research_executor",
        "rerun_selector",
        "research_round",
        "risk_manager",
        "report_writer",
    }
    lines = [
        "# Operator Summary",
        "",
        f"- run_id: `{run}`",
        f"- generated_at_utc: `{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`",
        "",
        "## Timeline",
    ]
    try:
        for raw in events_path.read_text(encoding="utf-8").splitlines():
            event = json.loads(raw)
            if event.get("phase") != "exit":
                continue
            node = str(event.get("node_name", ""))
            if node not in interesting:
                continue
            iteration = event.get("iteration", 0)
            summary = event.get("outputs_summary", {})
            brief = _short_text(json.dumps(summary, ensure_ascii=False), max_len=220)
            lines.append(f"- iter={iteration} `{node}`: {brief}")
    except (OSError, json.JSONDecodeError):
        return None

    out = run_dir / "operator_summary.md"
    try:
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        return None
    return out


def _log(state: dict, node: str, phase: str, data: dict | None = None):
    rid = state.get("run_id")
    if not rid:
        return
    telemetry.log_event(
        rid, node_name=node, iteration=state.get("iteration_count", 0),
        phase=phase,
        inputs_summary={"ticker": state.get("target_ticker")} if phase == "enter" else None,
        outputs_summary=data,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ① Orchestrator Node
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def orchestrator_node(state: InvestmentState) -> dict:
    """① Orchestrator — CIO/PM."""
    _log(state, "orchestrator", "enter")
    user_request = str(state.get("user_request", ""))

    if HAS_ORCHESTRATOR:
        result = _orch_node_impl(state)
    else:
        iteration = state.get("iteration_count", 0)
        ticker = "AAPL"
        result = {
            "target_ticker": ticker,
            "analysis_tasks": ALL_DESKS[:],
            "iteration_count": iteration + 1,
            "orchestrator_directives": {
                "action_type": "initial_delegation" if iteration == 0 else "fallback_abort",
            },
        }

    directives = result.get("orchestrator_directives", {})
    if not isinstance(directives, dict):
        directives = {}
    universe = _extract_universe_from_directives(directives)
    if not universe and state.get("universe"):
        universe = [str(t).strip().upper() for t in state.get("universe", []) if str(t).strip()]
    target_ticker = str(result.get("target_ticker", "")).strip().upper()
    if not universe and target_ticker:
        universe = [target_ticker]
    if universe:
        target_ticker = universe[0]

    output_language = str(state.get("output_language", "")).strip().lower() or _detect_output_language(user_request)
    intent = str(directives.get("intent", "")).strip() or _infer_intent_from_request(user_request, universe)
    execution_mode = "B_main_plus_hedge_lite" if len(universe) >= 2 else "single_main"
    asset_type_by_ticker = _build_asset_type_map(universe)

    result["target_ticker"] = target_ticker
    result["universe"] = universe
    result["asset_type_by_ticker"] = asset_type_by_ticker
    result["intent"] = intent
    result["output_language"] = output_language
    result["analysis_execution_mode"] = execution_mode

    # Normalize analysis_tasks
    raw = result.get("analysis_tasks", [])
    normalized = list(dict.fromkeys(
        _TASK_TO_DESK.get(t, t) for t in raw if _TASK_TO_DESK.get(t, t) in ALL_DESKS
    ))
    result["analysis_tasks"] = normalized if normalized else ALL_DESKS[:]

    # Init completed_tasks for this iteration
    ct = {d: (d not in result["analysis_tasks"]) for d in ALL_DESKS}
    result["completed_tasks"] = ct

    trace_entry = {
        "node": "orchestrator",
        "iteration": result.get("iteration_count", 0),
        "action_type": directives.get("action_type"),
        "analysis_tasks": result["analysis_tasks"],
        "intent": intent,
        "execution_mode": execution_mode,
        "universe_size": len(universe),
    }
    result["trace"] = [trace_entry]

    brief = directives.get("investment_brief", {}) if isinstance(directives.get("investment_brief"), dict) else {}
    _ops_log(
        state,
        "orchestrator",
        "dispatch plan",
        [
            f"action={trace_entry.get('action_type')} targets={_list_preview(brief.get('target_universe', []), max_items=7, max_len=160) or 'n/a'}",
            f"intent={intent} mode={execution_mode} language={output_language}",
            f"tasks={','.join(result.get('analysis_tasks', []))}",
            f"rationale={_short_text(brief.get('rationale', ''), max_len=180) or 'n/a'}",
        ],
    )

    _log(state, "orchestrator", "exit", {"action": trace_entry.get("action_type")})
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ②~⑤ Desk Analyst Nodes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _make_seed(state: dict) -> int:
    """Stable seed: CLI seed if present, else sha256 of run_id (cross-process reproducible)."""
    explicit_seed = state.get("seed")
    if explicit_seed is not None:
        return int(explicit_seed) % (2**31)
    import hashlib
    run_id = state.get("run_id", "42")
    h = hashlib.sha256(run_id.encode()).digest()
    return int.from_bytes(h[:4], "big") % (2**31)


def _request_key(req: dict) -> tuple:
    return (
        str(req.get("desk", "")).strip().lower(),
        str(req.get("kind", "")).strip().lower(),
        str(req.get("ticker", "")).strip().upper(),
        str(req.get("series_id", "")).strip(),
        _normalize_query_text(req.get("query", "")),
    )


def _merge_requests(*request_lists: list[dict]) -> list[dict]:
    out: list[dict] = []
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


def _merge_actions(*action_lists: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen = set()
    for actions in action_lists:
        for action in actions or []:
            if not isinstance(action, dict):
                continue
            k = (
                action.get("type", ""),
                action.get("detail", ""),
                json.dumps(action.get("params", {}), sort_keys=True, ensure_ascii=False),
            )
            if k in seen:
                continue
            seen.add(k)
            out.append(action)
    return out


def _merge_limitations(*limitation_lists: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for limitations in limitation_lists:
        for lim in limitations or []:
            text = str(lim).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
    return out


def _get_desk_task(
    state: dict,
    desk: str,
    default_horizon: int,
) -> tuple[int, list[str], str | None]:
    directives = state.get("orchestrator_directives", {}) or {}
    desk_tasks = directives.get("desk_tasks", {}) if isinstance(directives, dict) else {}
    task = desk_tasks.get(desk, {}) if isinstance(desk_tasks, dict) else {}

    try:
        horizon_days = int(task.get("horizon_days", default_horizon))
    except (TypeError, ValueError):
        horizon_days = default_horizon
    horizon_days = max(1, min(365, horizon_days))

    raw_focus = task.get("focus_areas", []) if isinstance(task, dict) else []
    focus_areas = [str(x).strip() for x in raw_focus if str(x).strip()] if isinstance(raw_focus, list) else []

    risk_budget = None
    if isinstance(task, dict) and task.get("risk_budget"):
        risk_budget = str(task.get("risk_budget")).strip() or None

    return horizon_days, focus_areas, risk_budget


def macro_analyst_node(state: InvestmentState) -> dict:
    """② Macro Analyst."""
    if state.get("completed_tasks", {}).get("macro", False):
        return {}
    _log(state, "macro", "enter")

    mode = state.get("mode", "mock")
    seed = _make_seed(state)
    as_of = state.get("as_of", "")
    ticker = state.get("target_ticker", "AAPL")
    horizon_days, focus_areas, _ = _get_desk_task(state, "macro", 30)

    hub = DataHub(run_id=state.get("run_id", ""), as_of=as_of, mode=mode)
    indicators, _, meta = hub.get_macro_indicators(ticker, seed=seed)
    output = macro_analyst_run(
        ticker, indicators,
        run_id=state.get("run_id", ""), as_of=as_of,
        horizon_days=horizon_days,
        focus_areas=focus_areas,
        state=state,
        source_name="mock" if mode == "mock" else "FRED",
    )
    rerun_reason = _rerun_reason_for_desk(state, "macro")
    prev_macro = state.get("macro_analysis", {}) if isinstance(state.get("macro_analysis"), dict) else {}
    output["limitations"] = _merge_limitations(
        output.get("limitations", []),
        meta.get("limitations", []) if isinstance(meta, dict) else [],
    )
    output = _attach_decision_change_log(
        desk="macro",
        output=output,
        state=state,
        prev_output=prev_macro,
        rerun_reason=rerun_reason,
        ticker=ticker,
    )
    if isinstance(meta, dict):
        output.setdefault("provider_meta", {})["macro"] = meta

    print(f"\n② MACRO ANALYST  (iter #{state.get('iteration_count', 1)})")
    print(f"   Regime: {output['macro_regime']}, GDP: {indicators.get('gdp_growth')}")
    print(f"   Why called: {rerun_reason}")
    print(f"   Basis: {_list_preview(output.get('key_drivers', []), max_items=2, max_len=160) or 'n/a'}")

    _ops_log(
        state,
        "macro",
        "desk output",
        [
            f"why={rerun_reason}",
            f"regime={output.get('macro_regime')} confidence={output.get('confidence')}",
            f"decision_change={output.get('decision_change_log', {}).get('status')}",
            f"key_drivers={_list_preview(output.get('key_drivers', []), max_items=3, max_len=180) or 'n/a'}",
            f"open_questions={len(output.get('open_questions', []))} evidence_requests={len(output.get('evidence_requests', []))}",
            _react_trace_preview(output),
        ],
    )

    _log(state, "macro", "exit", {"regime": output["macro_regime"]})
    return {
        "macro_analysis": output,
        "completed_tasks": {"macro": True},
        "evidence_requests": output.get("evidence_requests", []),
    }


def fundamental_analyst_node(state: InvestmentState) -> dict:
    """③ Fundamental Analyst."""
    if state.get("completed_tasks", {}).get("fundamental", False):
        return {}
    _log(state, "fundamental", "enter")

    mode = state.get("mode", "mock")
    seed = _make_seed(state) + 1
    as_of = state.get("as_of", "")
    ticker = state.get("target_ticker", "AAPL")
    asset_type = str((state.get("asset_type_by_ticker", {}) or {}).get(ticker, "")).strip().upper() or _infer_asset_type(ticker)
    horizon_days, focus_areas, _ = _get_desk_task(state, "fundamental", 90)

    hub = DataHub(run_id=state.get("run_id", ""), as_of=as_of, mode=mode)
    financials, _, fmeta = hub.get_fundamentals(ticker, seed=seed)
    sec_data, _, smeta = hub.get_sec_flags(ticker)
    output = fundamental_analyst_run(
        ticker, financials, sec_data=sec_data,
        run_id=state.get("run_id", ""), as_of=as_of,
        horizon_days=horizon_days,
        focus_areas=focus_areas,
        state=state,
        asset_type=asset_type,
        source_name="mock" if mode == "mock" else "FMP/SEC",
    )
    rerun_reason = _rerun_reason_for_desk(state, "fundamental")
    prev_funda = state.get("fundamental_analysis", {}) if isinstance(state.get("fundamental_analysis"), dict) else {}
    output["asset_type"] = asset_type
    output["evidence_requests"] = _sanitize_fundamental_requests_for_asset_type(
        output.get("evidence_requests", []),
        ticker=ticker,
        asset_type=asset_type,
    )
    output["limitations"] = _merge_limitations(
        output.get("limitations", []),
        fmeta.get("limitations", []) if isinstance(fmeta, dict) else [],
        smeta.get("limitations", []) if isinstance(smeta, dict) else [],
    )
    output = _attach_decision_change_log(
        desk="fundamental",
        output=output,
        state=state,
        prev_output=prev_funda,
        rerun_reason=rerun_reason,
        ticker=ticker,
    )
    output.setdefault("provider_meta", {})["fundamentals"] = fmeta
    output.setdefault("provider_meta", {})["sec"] = smeta

    print(f"\n③ FUNDAMENTAL ANALYST  (iter #{state.get('iteration_count', 1)})")
    print(f"   Structural Risk: {output['structural_risk_flag']}, Decision: {output['primary_decision']}")
    print(f"   Why called: {rerun_reason}")
    print(f"   Basis: {_list_preview(output.get('key_drivers', []), max_items=2, max_len=160) or 'n/a'}")

    _ops_log(
        state,
        "fundamental",
        "desk output",
        [
            f"why={rerun_reason}",
            f"decision={output.get('primary_decision')} structural_risk={output.get('structural_risk_flag')} confidence={output.get('confidence')}",
            f"decision_change={output.get('decision_change_log', {}).get('status')}",
            f"key_drivers={_list_preview(output.get('key_drivers', []), max_items=3, max_len=180) or 'n/a'}",
            f"open_questions={len(output.get('open_questions', []))} evidence_requests={len(output.get('evidence_requests', []))}",
            _react_trace_preview(output),
        ],
    )

    _log(state, "fundamental", "exit", {"structural_risk": output["structural_risk_flag"]})
    return {
        "fundamental_analysis": output,
        "completed_tasks": {"fundamental": True},
        "evidence_requests": output.get("evidence_requests", []),
    }


def sentiment_analyst_node(state: InvestmentState) -> dict:
    """④ Sentiment Analyst."""
    if state.get("completed_tasks", {}).get("sentiment", False):
        return {}
    _log(state, "sentiment", "enter")

    mode = state.get("mode", "mock")
    seed = _make_seed(state) + 2
    as_of = state.get("as_of", "")
    ticker = state.get("target_ticker", "AAPL")
    horizon_days, focus_areas, _ = _get_desk_task(state, "sentiment", 7)

    hub = DataHub(run_id=state.get("run_id", ""), as_of=as_of, mode=mode)
    indicators, _, meta = hub.get_news_sentiment(ticker, seed=seed)
    output = sentiment_analyst_run(
        ticker, indicators,
        run_id=state.get("run_id", ""), as_of=as_of,
        horizon_days=horizon_days,
        focus_areas=focus_areas,
        state=state,
        source_name="mock" if mode == "mock" else "NewsAPI",
    )
    rerun_reason = _rerun_reason_for_desk(state, "sentiment")
    prev_sentiment = state.get("sentiment_analysis", {}) if isinstance(state.get("sentiment_analysis"), dict) else {}
    output["limitations"] = _merge_limitations(
        output.get("limitations", []),
        meta.get("limitations", []) if isinstance(meta, dict) else [],
    )
    output = _attach_decision_change_log(
        desk="sentiment",
        output=output,
        state=state,
        prev_output=prev_sentiment,
        rerun_reason=rerun_reason,
        ticker=ticker,
    )
    if isinstance(meta, dict):
        output.setdefault("provider_meta", {})["sentiment"] = meta

    print(f"\n④ SENTIMENT ANALYST  (iter #{state.get('iteration_count', 1)})")
    print(f"   Regime: {output['sentiment_regime']}, Tilt: {output['tilt_factor']}")
    print(f"   Why called: {rerun_reason}")
    print(f"   Basis: {_list_preview(output.get('key_drivers', []), max_items=2, max_len=160) or 'n/a'}")

    _ops_log(
        state,
        "sentiment",
        "desk output",
        [
            f"why={rerun_reason}",
            f"regime={output.get('sentiment_regime')} tilt={output.get('tilt_factor')} confidence={output.get('confidence')}",
            f"decision_change={output.get('decision_change_log', {}).get('status')}",
            f"key_drivers={_list_preview(output.get('key_drivers', []), max_items=3, max_len=180) or 'n/a'}",
            f"open_questions={len(output.get('open_questions', []))} evidence_requests={len(output.get('evidence_requests', []))}",
            _react_trace_preview(output),
        ],
    )

    _log(state, "sentiment", "exit", {"tilt": output["tilt_factor"]})
    return {
        "sentiment_analysis": output,
        "completed_tasks": {"sentiment": True},
        "evidence_requests": output.get("evidence_requests", []),
    }


def _quant_select_pair_ticker(*, ticker: str, intent: str, asset_type: str) -> str:
    t = str(ticker or "").strip().upper()
    it = str(intent or "").strip().lower()
    at = str(asset_type or "").strip().upper()
    if it in {"market_outlook", "event_risk", "hedge_design"}:
        if t == "XLE":
            return "SPY"
        if t in {"SPY", "QQQ", "IWM"} or at in {"ETF", "INDEX"}:
            return "XLE"
        return "GLD"
    if at in {"ETF", "INDEX"} and t != "SPY":
        return "SPY"
    return "MSFT"


def _safe_pct_change(prices: Any, days: int) -> float | None:
    if not hasattr(prices, "__len__") or len(prices) <= days:
        return None
    try:
        prev = float(prices[-(days + 1)])
        cur = float(prices[-1])
    except (TypeError, ValueError, IndexError):
        return None
    if abs(prev) < 1e-12:
        return None
    return (cur / prev) - 1.0


def _safe_ann_vol(prices: Any, window: int = 20) -> float | None:
    if not hasattr(prices, "__len__") or len(prices) < window + 2:
        return None
    try:
        p = np.asarray(prices, dtype=float)
        rets = np.diff(np.log(p))
        tail = rets[-window:]
        if len(tail) == 0:
            return None
        return float(np.std(tail, ddof=0) * np.sqrt(252))
    except Exception:
        return None


def _safe_corr(a: Any, b: Any, window: int = 60) -> float | None:
    if not hasattr(a, "__len__") or not hasattr(b, "__len__"):
        return None
    n = min(len(a), len(b))
    if n < window + 2:
        return None
    try:
        aa = np.asarray(a, dtype=float)[-n:]
        bb = np.asarray(b, dtype=float)[-n:]
        ra = np.diff(np.log(aa))[-window:]
        rb = np.diff(np.log(bb))[-window:]
        if len(ra) == 0 or len(rb) == 0:
            return None
        corr = np.corrcoef(ra, rb)[0, 1]
        if np.isnan(corr):
            return None
        return float(corr)
    except Exception:
        return None


def _safe_beta_to_main(hedge_prices: Any, main_prices: Any, window: int = 60) -> float | None:
    if not hasattr(hedge_prices, "__len__") or not hasattr(main_prices, "__len__"):
        return None
    n = min(len(hedge_prices), len(main_prices))
    if n < window + 2:
        return None
    try:
        hp = np.asarray(hedge_prices, dtype=float)[-n:]
        mp = np.asarray(main_prices, dtype=float)[-n:]
        rh = np.diff(np.log(hp))[-window:]
        rm = np.diff(np.log(mp))[-window:]
        if len(rh) == 0 or len(rm) == 0:
            return None
        var_main = float(np.var(rm, ddof=0))
        if var_main < 1e-12:
            return None
        cov = float(np.cov(rh, rm, ddof=0)[0, 1])
        return cov / var_main
    except Exception:
        return None


def _safe_downside_capture(hedge_prices: Any, main_prices: Any, window: int = 60) -> float | None:
    if not hasattr(hedge_prices, "__len__") or not hasattr(main_prices, "__len__"):
        return None
    n = min(len(hedge_prices), len(main_prices))
    if n < window + 2:
        return None
    try:
        hp = np.asarray(hedge_prices, dtype=float)[-n:]
        mp = np.asarray(main_prices, dtype=float)[-n:]
        rh = np.diff(np.log(hp))[-window:]
        rm = np.diff(np.log(mp))[-window:]
        if len(rh) == 0 or len(rm) == 0:
            return None
        mask = rm < 0
        if int(np.sum(mask)) < 3:
            return None
        return float(np.mean(rh[mask]))
    except Exception:
        return None


def _safe_tail_proxy(prices: Any, window: int = 60) -> float | None:
    if not hasattr(prices, "__len__") or len(prices) < window + 2:
        return None
    try:
        p = np.asarray(prices, dtype=float)
        rets = np.diff(np.log(p))
        tail = rets[-window:]
        if len(tail) == 0:
            return None
        return float(np.percentile(tail, 5))
    except Exception:
        return None


def _liquidity_proxy_score(ticker: str, asset_type_by_ticker: dict | None = None) -> float:
    asset_type = str((asset_type_by_ticker or {}).get(ticker, "")).strip().upper() or _infer_asset_type(ticker)
    if asset_type in {"ETF", "INDEX"}:
        return 0.90
    if asset_type in {"BOND", "COMMODITY"}:
        return 0.80
    if asset_type == "EQUITY":
        return 0.60
    return 0.50


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


def _normalize_long_only_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {str(k): max(0.0, float(v)) for k, v in (weights or {}).items() if v is not None}
    s = float(sum(cleaned.values()))
    if s <= 1e-12:
        return {}
    return {k: v / s for k, v in cleaned.items()}


def hedge_lite_builder_node(state: InvestmentState) -> dict:
    """B 모드에서 메인+헤지 후보를 경량 분석해 positions_proposed를 생성."""
    analysis_mode = str(state.get("analysis_execution_mode", "")).strip()
    universe = [str(t).strip().upper() for t in (state.get("universe", []) or []) if str(t).strip()]
    if not analysis_mode.startswith("B_") or len(universe) < 2:
        return {}

    _log(state, "hedge_lite_builder", "enter")
    as_of = state.get("as_of", "")
    seed = _make_seed(state) + 17
    mode = state.get("mode", "mock")
    intent = str(state.get("intent", "")).strip().lower()
    main = str(state.get("target_ticker", "")).strip().upper() or universe[0]
    if main not in universe:
        universe = [main] + [t for t in universe if t != main]
    hedges = [t for t in universe if t != main]

    hub = DataHub(run_id=state.get("run_id", ""), as_of=as_of, mode=mode)
    main_prices, main_evidence, _ = hub.get_price_series(main, lookback_days=260, seed=seed)
    market_prices, market_evidence, _ = hub.get_market_series("SPY", lookback_days=260, seed=seed + 1)

    main_metrics = {
        "ret_5d": _safe_pct_change(main_prices, 5),
        "trend_20d": _safe_pct_change(main_prices, 20),
        "vol_20d_ann": _safe_ann_vol(main_prices, window=20),
        "vol_60d_ann": _safe_ann_vol(main_prices, window=60),
        "corr_with_market_60d": _safe_corr(main_prices, market_prices, window=60),
    }
    v20 = main_metrics.get("vol_20d_ann")
    v60 = main_metrics.get("vol_60d_ann")
    main_metrics["vol_shift_20d_vs_60d"] = float(v20) / float(v60) if v20 is not None and v60 not in (None, 0) else None

    hedge_rows: dict[str, dict] = {}
    evidence = list(main_evidence) + list(market_evidence)
    for i, hedge in enumerate(hedges):
        hp, hp_ev, _ = hub.get_price_series(hedge, lookback_days=260, seed=seed + (i + 1) * 37)
        evidence.extend(hp_ev)
        row = {
            "ticker": hedge,
            "ret_5d": _safe_pct_change(hp, 5),
            "trend_20d": _safe_pct_change(hp, 20),
            "vol_20d_ann": _safe_ann_vol(hp, window=20),
            "vol_60d_ann": _safe_ann_vol(hp, window=60),
            "corr_with_main_60d": _safe_corr(hp, main_prices, window=60),
            "corr_with_market_60d": _safe_corr(hp, market_prices, window=60),
            "beta_to_main_60d": _safe_beta_to_main(hp, main_prices, window=60),
            "downside_capture_60d": _safe_downside_capture(hp, main_prices, window=60),
            "tail_proxy_5pct_60d": _safe_tail_proxy(hp, window=60),
            "liquidity_proxy": _liquidity_proxy_score(hedge, state.get("asset_type_by_ticker", {})),
        }
        hv20 = row.get("vol_20d_ann")
        hv60 = row.get("vol_60d_ann")
        row["vol_shift_20d_vs_60d"] = float(hv20) / float(hv60) if hv20 is not None and hv60 not in (None, 0) else None

        protect = _clip(-float(row["corr_with_main_60d"])) if row.get("corr_with_main_60d") is not None else 0.0
        downside = _clip(float(row["downside_capture_60d"]) * 120.0) if row.get("downside_capture_60d") is not None else 0.0
        stability = _clip(1.0 - abs(float(row["vol_shift_20d_vs_60d"]) - 1.0)) if row.get("vol_shift_20d_vs_60d") is not None else 0.0
        trend = _clip(float(row["trend_20d"]) * 8.0) if row.get("trend_20d") is not None else 0.0
        tail = _clip(-float(row["tail_proxy_5pct_60d"]) * 20.0) if row.get("tail_proxy_5pct_60d") is not None else 0.0
        liquidity = _clip((float(row["liquidity_proxy"]) - 0.5) * 2.0)

        score = (
            0.35 * protect
            + 0.20 * downside
            + 0.20 * stability
            + 0.10 * trend
            + 0.10 * tail
            + 0.05 * liquidity
        )
        row["score_components"] = {
            "protect": round(protect, 6),
            "downside": round(downside, 6),
            "stability": round(stability, 6),
            "trend": round(trend, 6),
            "tail": round(tail, 6),
            "liquidity": round(liquidity, 6),
        }
        row["score"] = round(float(score), 6)
        row["status"] = "ok" if row.get("corr_with_main_60d") is not None and row.get("vol_20d_ann") is not None else "insufficient_data"
        row["selected"] = False
        hedge_rows[hedge] = row

        for metric in ("ret_5d", "vol_shift_20d_vs_60d", "corr_with_main_60d", "beta_to_main_60d"):
            mv = row.get(metric)
            if mv is None:
                continue
            evidence.append(
                make_evidence(
                    metric=f"{hedge.lower()}_{metric}",
                    value=mv,
                    source_name="hedge_lite",
                    source_type="model",
                    quality=0.8,
                    as_of=as_of,
                )
            )

    valid = [t for t, row in hedge_rows.items() if row.get("status") == "ok"]
    sorted_valid = sorted(valid, key=lambda t: float(hedge_rows[t].get("score", -999.0)), reverse=True)
    top_k = min(2, len(sorted_valid))
    selected = sorted_valid[:top_k]
    for t in selected:
        hedge_rows[t]["selected"] = True

    hedge_budget = 0.15
    main_vol_shift = main_metrics.get("vol_shift_20d_vs_60d")
    if intent in {"event_risk", "hedge_design"} or (main_vol_shift is not None and float(main_vol_shift) > 1.2):
        hedge_budget = 0.25
    if not selected:
        hedge_budget = 0.0

    hedge_raw_weights: dict[str, float] = {}
    for t in selected:
        score = max(0.0, float(hedge_rows[t].get("score", 0.0)))
        vol = hedge_rows[t].get("vol_20d_ann")
        inv_vol = 1.0 / max(float(vol), 1e-6) if vol not in (None, 0) else 1.0
        hedge_raw_weights[t] = (score if score > 0 else 0.25) * inv_vol
    hedge_norm = _normalize_long_only_weights(hedge_raw_weights)
    if selected and not hedge_norm:
        hedge_norm = _normalize_long_only_weights({t: 1.0 for t in selected})

    proposed_raw = {main: max(0.0, 1.0 - hedge_budget)}
    for t in hedges:
        proposed_raw.setdefault(t, 0.0)
    for t, w in hedge_norm.items():
        proposed_raw[t] = hedge_budget * float(w)

    positions_proposed = _normalize_long_only_weights(proposed_raw)
    if not positions_proposed:
        positions_proposed = {main: 1.0}
    for t in universe:
        positions_proposed.setdefault(t, 0.0)
    total = float(sum(float(v) for v in positions_proposed.values()))
    if total > 1e-12:
        positions_proposed = {t: float(v) / total for t, v in positions_proposed.items()}
    residual = 1.0 - float(sum(float(v) for v in positions_proposed.values()))
    if abs(residual) > 1e-12:
        positions_proposed[main] = max(0.0, float(positions_proposed.get(main, 0.0)) + residual)
        positions_proposed = _normalize_long_only_weights(positions_proposed) or {main: 1.0}
    for t in universe:
        positions_proposed.setdefault(t, 0.0)

    status = "ok" if selected else "insufficient_data"
    reason = "selected_top_scores" if selected else "no_hedge_with_minimum_data"
    hedge_lite = {
        "status": status,
        "reason": reason,
        "analysis_execution_mode": analysis_mode,
        "main": main,
        "universe": universe,
        "hedges": hedge_rows,
        "selected_hedges": selected,
        "hedge_budget": hedge_budget,
        "main_weight": positions_proposed.get(main, 0.0),
        "main_metrics": main_metrics,
        "weights_proposed": {t: positions_proposed.get(t, 0.0) for t in universe},
        "intent": intent,
        "timestamp": as_of,
        "seed": seed,
        "build_version": "hedge_lite_v1",
        "evidence": evidence,
    }

    audit = dict(state.get("audit", {}) or {})
    audit_paths = dict(audit.get("paths", {}) or {})
    audit_paths["hedge_lite_builder"] = {
        "status": status,
        "reason": reason,
        "seed": seed,
        "selected_hedges": selected,
        "timestamp": as_of,
    }
    audit["paths"] = audit_paths

    _ops_log(
        state,
        "hedge_lite_builder",
        "hedge lite output",
        [
            f"mode={analysis_mode} main={main} hedges={hedges}",
            f"status={status} selected={selected}",
            f"hedge_budget={hedge_budget:.2f} weights={positions_proposed}",
        ],
    )
    _log(state, "hedge_lite_builder", "exit", {"status": status, "selected": selected})
    return {
        "hedge_lite": hedge_lite,
        "positions_proposed": positions_proposed,
        "audit": audit,
    }


def _event_regime_quant_decision(payload: dict) -> dict:
    sig = dict(payload.get("event_regime_signals", {}) or {})
    trend20 = sig.get("trend_20d")
    trend5 = sig.get("event_return_5d")
    vol_shift = sig.get("vol_shift_20d_vs_60d")
    corr_market = sig.get("corr_with_market_60d")
    rel = sig.get("pair_relative_strength_20d")
    regime_prob = (
        (payload.get("market_regime_context", {}) or {})
        .get("state_probabilities", {})
        .get("regime_2_high_vol", 0.0)
    )

    cot = [
        f"[Event/Regime] 5d={trend5}, 20d={trend20}, vol_shift={vol_shift}, corr60={corr_market}, rel20={rel}",
        f"[RegimeProb] regime_2_high_vol={regime_prob}",
    ]

    risk_off = (regime_prob or 0.0) > 0.5 or ((vol_shift or 0.0) > 1.25 and (trend5 or 0.0) < 0)
    bullish = ((trend20 or 0.0) > 0.01 and (trend5 or 0.0) >= 0 and (rel or 0.0) >= -0.02 and not risk_off)
    bearish = ((trend20 or 0.0) < -0.01 and (trend5 or 0.0) < 0 and (vol_shift or 0.0) > 1.1)

    if bullish:
        decision = "LONG"
        alloc = 0.06
        cot.append("[Decision] trend/relative strength 우호 + 고변동 리스크 제한적 → LONG")
    elif bearish or risk_off:
        decision = "HOLD"
        alloc = 0.0
        cot.append("[Decision] 이벤트/변동성 리스크 우세 → HOLD")
    else:
        decision = "HOLD"
        alloc = 0.0
        cot.append("[Decision] 신호 혼조 → HOLD")

    return {
        "cot_reasoning": " ".join(cot),
        "decision": decision,
        "final_allocation_pct": alloc,
    }


def quant_analyst_node(state: InvestmentState) -> dict:
    """⑤ Quant Analyst (wrapper: data_provider → quant_engine → mock decision)."""
    if state.get("completed_tasks", {}).get("quant", False):
        return {}
    _log(state, "quant", "enter")

    mode = state.get("mode", "mock")
    seed = _make_seed(state) + 3
    as_of = state.get("as_of", "")
    ticker = state.get("target_ticker", "AAPL")
    intent = str(state.get("intent", "")).strip().lower() or "single_name"
    asset_type = str((state.get("asset_type_by_ticker", {}) or {}).get(ticker, "")).strip().upper() or _infer_asset_type(ticker)
    horizon_days, focus_areas, risk_budget = _get_desk_task(state, "quant", 10)
    pair_ticker = _quant_select_pair_ticker(ticker=ticker, intent=intent, asset_type=asset_type)
    analysis_mode = "event_regime" if intent in {"market_outlook", "event_risk", "hedge_design"} else "statarb"
    rerun_reason = _rerun_reason_for_desk(state, "quant")
    prev_quant = state.get("technical_analysis", {}) if isinstance(state.get("technical_analysis"), dict) else {}

    # Data Provider → 가격 배열 (via DataHub)
    hub = DataHub(run_id=state.get("run_id", ""), as_of=as_of, mode=mode)
    prices, p_ev, _ = hub.get_price_series(ticker, seed=seed)
    pair_prices, _, _ = hub.get_price_series(pair_ticker, seed=seed + 100)
    market_prices, _, _ = hub.get_market_series("SPY", seed=seed + 200)

    # Engine → 순수 연산
    print(f"\n⑤ QUANT ANALYST  (iter #{state.get('iteration_count', 1)})")
    print(f"   Computing quant payload for {ticker}... (mode={analysis_mode}, pair={pair_ticker})")
    payload = generate_quant_payload(ticker, prices, pair_prices, market_prices, pair_ticker=pair_ticker)

    event_regime_signals = {
        "event_return_5d": _safe_pct_change(prices, 5),
        "trend_20d": _safe_pct_change(prices, 20),
        "vol_20d_ann": _safe_ann_vol(prices, window=20),
        "vol_60d_ann": _safe_ann_vol(prices, window=60),
        "corr_with_market_60d": _safe_corr(prices, market_prices, window=60),
        "pair_relative_strength_20d": None,
    }
    if event_regime_signals["trend_20d"] is not None:
        pair_trend20 = _safe_pct_change(pair_prices, 20)
        if pair_trend20 is not None:
            event_regime_signals["pair_relative_strength_20d"] = (
                float(event_regime_signals["trend_20d"]) - float(pair_trend20)
            )
    v20 = event_regime_signals["vol_20d_ann"]
    v60 = event_regime_signals["vol_60d_ann"]
    if v20 is not None and v60 not in (None, 0):
        event_regime_signals["vol_shift_20d_vs_60d"] = float(v20) / float(v60)
    else:
        event_regime_signals["vol_shift_20d_vs_60d"] = None
    payload["event_regime_signals"] = event_regime_signals

    # Intent-aware decision mode:
    # market_outlook/event_risk: event/regime/vol
    # relative_value/single_name: statarb
    if analysis_mode == "event_regime":
        decision = _event_regime_quant_decision(payload)
    else:
        decision = mock_quant_decision(payload)
    z = ((payload.get("alpha_signals", {}).get("statistical_arbitrage", {})
          .get("execution", {})).get("current_z_score"))
    cvar = (payload.get("portfolio_risk_parameters", {}).get("asset_cvar_99_daily"))

    print(f"   Decision: {decision['decision']} | Alloc: {decision['final_allocation_pct']}")
    print(f"   Why called: {rerun_reason}")

    _ops_log(
        state,
        "quant",
        "desk output",
        [
            f"why={rerun_reason}",
            f"decision={decision.get('decision')} alloc={decision.get('final_allocation_pct')} mode={analysis_mode}",
            f"pair={pair_ticker} z_score={z} cvar99={cvar}",
            (
                "event_regime="
                f"ret5={event_regime_signals.get('event_return_5d')} "
                f"trend20={event_regime_signals.get('trend_20d')} "
                f"vol_shift={event_regime_signals.get('vol_shift_20d_vs_60d')}"
            ),
        ],
    )

    # Evidence
    evidence = list(p_ev)
    if z is not None:
        evidence.append(make_evidence(metric="z_score", value=z, source_name="quant_engine", source_type="model", quality=0.9, as_of=as_of))
    if cvar is not None:
        evidence.append(make_evidence(metric="asset_cvar_99_daily", value=cvar, source_name="quant_engine", source_type="model", quality=0.9, as_of=as_of))
    for k in ("event_return_5d", "trend_20d", "vol_shift_20d_vs_60d"):
        v = event_regime_signals.get(k)
        if v is None:
            continue
        evidence.append(make_evidence(metric=k, value=v, source_name="quant_event_regime", source_type="model", quality=0.85, as_of=as_of))

    output = {
        "agent_type": "quant",
        "run_id": state.get("run_id", ""),
        "generated_at": as_of,
        "as_of": as_of,
        "ticker": ticker,
        "horizon_days": horizon_days,
        "focus_areas": focus_areas,
        "risk_budget": risk_budget,
        "intent": intent,
        "analysis_mode": analysis_mode,
        "pair_ticker": pair_ticker,
        "asset_type": asset_type,
        "decision": decision["decision"],
        "final_allocation_pct": decision["final_allocation_pct"],
        "z_score": z,
        "asset_cvar_99_daily": cvar,
        "event_regime_signals": event_regime_signals,
        "quant_payload": payload,
        "llm_decision": decision,
        "evidence": evidence,
        "summary": (
            f"Quant({analysis_mode}): {decision['decision']} alloc={decision['final_allocation_pct']}, "
            f"Z={z}, ret5={event_regime_signals.get('event_return_5d')}"
        ),
        "status": "ok",
        "data_ok": payload.get("_data_ok", True),
        "quant_indicators": (
            ["event_return_5d", "trend_20d", "vol_shift_20d_vs_60d", "corr_with_market_60d"]
            if analysis_mode == "event_regime"
            else ["adf_pvalue", "z_score", "asset_cvar_99_daily"]
        ),
        "primary_decision": {
            "LONG": "bullish", "SHORT": "bearish",
            "HOLD": "hold", "CLEAR": "neutral",
        }.get(decision["decision"], "hold"),
        "recommendation": "allow" if decision["decision"] in ("LONG", "SHORT") else "allow_with_limits",
        "confidence": 0.6 if payload.get("_data_ok") else 0.35,
        "risk_flags": [],
        "limitations": ["Mock data" if mode == "mock" else ""],
    }
    output = _attach_decision_change_log(
        desk="quant",
        output=output,
        state=state,
        prev_output=prev_quant,
        rerun_reason=rerun_reason,
        ticker=ticker,
    )
    _ops_log(
        state,
        "quant",
        "decision trace",
        [f"decision_change={output.get('decision_change_log', {}).get('status')}"],
    )

    _log(state, "quant", "exit", {"decision": decision["decision"]})
    return {"technical_analysis": output, "completed_tasks": {"quant": True}, "evidence_requests": []}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Barrier + Risk Manager + Report Writer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def barrier_node(state: InvestmentState) -> dict:
    """Fan-in barrier. Marks unfinished desks as skipped with iteration_generated."""
    ct = dict(state.get("completed_tasks", {}))
    iteration = state.get("iteration_count", 0)
    updates: dict = {}
    for desk in ALL_DESKS:
        key = _DESK_TO_STATE_KEY[desk]
        desk_out = state.get(key)
        if not ct.get(desk, False):
            # Desk didn't run this iteration
            ct[desk] = True
            if not desk_out:
                updates[key] = {
                    "summary": f"{desk} skipped",
                    "evidence": [],
                    "status": "skipped",
                    "iteration_generated": iteration,
                }
            else:
                # Previous result exists but was not re-run this iteration
                merged = dict(desk_out)
                merged["status"] = "skipped"
                merged["stale_previous"] = True
                merged["iteration_generated"] = merged.get("iteration_generated", iteration - 1)
                updates[key] = merged
        else:
            # Desk ran: ensure iteration_generated is set
            if desk_out and desk_out.get("iteration_generated") is None:
                merged = dict(desk_out)
                merged["iteration_generated"] = iteration
                updates[key] = merged
    updates["completed_tasks"] = ct
    return updates


def _desk_outputs_for_policy(state: dict) -> dict:
    return {
        "macro": state.get("macro_analysis", {}),
        "fundamental": state.get("fundamental_analysis", {}),
        "sentiment": state.get("sentiment_analysis", {}),
        "quant": state.get("technical_analysis", {}),
    }


def _merge_request_sources(state: dict) -> list[dict]:
    reqs = list(state.get("evidence_requests", []))
    for key in ("macro_analysis", "fundamental_analysis", "sentiment_analysis"):
        reqs.extend((state.get(key, {}) or {}).get("evidence_requests", []) or [])
    return _merge_requests(reqs)


def _default_kind_for_desk(desk: str) -> str:
    if desk == "macro":
        return "macro_headline_context"
    if desk == "fundamental":
        return "valuation_context"
    if desk == "sentiment":
        return "catalyst_event_detail"
    return "web_search"


def _kind_to_swarm_bucket(kind: str) -> str:
    return _SWARM_KIND_TO_BUCKET.get((kind or "").lower(), "other")


def _default_impacted_desks(kind: str, desk: str = "") -> list[str]:
    kind_key = (kind or "").lower()
    impacted = list(_SWARM_KIND_TO_IMPACTED.get(kind_key, ()))
    if not impacted and desk in ("macro", "fundamental", "sentiment"):
        impacted = [desk]
    deduped = []
    seen = set()
    for d in impacted:
        if d == "quant":
            continue
        if d in ("macro", "fundamental", "sentiment") and d not in seen:
            seen.add(d)
            deduped.append(d)
    return deduped


def _stable_request_id(req: dict) -> str:
    raw = json.dumps(
        {
            "desk": req.get("desk", ""),
            "kind": req.get("kind", ""),
            "ticker": req.get("ticker", ""),
            "series_id": req.get("series_id", ""),
            "query": req.get("query", ""),
            "priority": req.get("priority", 3),
            "recency_days": req.get("recency_days", 30),
            "max_items": req.get("max_items", 5),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _request_sort_key(req: dict) -> tuple[int, str, str, str]:
    source_rank = {
        "open_questions": 0,
        "desk_rule": 1,
        "risk_feedback": 2,
        "disagreement": 3,
        "seed": 4,
        "llm_extra": 5,
    }
    try:
        priority = int(req.get("priority", 3))
    except (TypeError, ValueError):
        priority = 3
    tag = str(req.get("source_tag", "")).strip().lower()
    return (
        source_rank.get(tag, 9),
        max(1, min(5, priority)),
        str(req.get("kind", "")).strip().lower(),
        str(req.get("ticker", "")).strip().upper(),
        _normalize_query_text(req.get("query", "")),
    )


def _sanitize_swarm_request(req: dict, *, default_source_tag: str) -> dict | None:
    if not isinstance(req, dict):
        return None

    kind = str(req.get("kind", "")).strip().lower()
    if not kind:
        return None

    out = dict(req)
    out["kind"] = kind

    desk = str(out.get("desk", "orchestrator")).strip().lower()
    if desk not in ("macro", "fundamental", "sentiment", "quant", "orchestrator"):
        desk = "orchestrator"
    out["desk"] = desk

    ticker = str(out.get("ticker", "")).strip().upper()
    if ticker and ticker != "__GLOBAL__":
        ticker = "".join(ch for ch in ticker if ch.isalnum() or ch in ("-", "_", "."))[:8]
    out["ticker"] = ticker

    series_id = str(out.get("series_id", "")).strip()
    out["series_id"] = series_id

    query = str(out.get("query", "")).strip()
    out["query"] = query[:120]

    if not (out["ticker"] or out["series_id"] or out["query"]):
        return None

    try:
        priority = int(out.get("priority", 3))
    except (TypeError, ValueError):
        priority = 3
    out["priority"] = max(1, min(5, priority))

    try:
        recency_days = int(out.get("recency_days", 30))
    except (TypeError, ValueError):
        recency_days = 30
    out["recency_days"] = max(1, min(365, recency_days))

    try:
        max_items = int(out.get("max_items", 5))
    except (TypeError, ValueError):
        max_items = 5
    out["max_items"] = max(1, min(20, max_items))

    rationale = str(out.get("rationale", "")).strip()
    out["rationale"] = rationale or "swarm_request"

    source_tag = str(out.get("source_tag", "")).strip() or default_source_tag
    out["source_tag"] = source_tag
    out["expected_bucket"] = str(out.get("expected_bucket", "")).strip() or _kind_to_swarm_bucket(kind)

    impacted = out.get("impacted_desks", [])
    if not isinstance(impacted, list):
        impacted = []
    impacted_list = []
    seen = set()
    for d in impacted:
        ds = str(d).strip().lower()
        if ds in ("macro", "fundamental", "sentiment") and ds not in seen:
            seen.add(ds)
            impacted_list.append(ds)
    if not impacted_list:
        impacted_list = _default_impacted_desks(kind, desk=desk)
    out["impacted_desks"] = impacted_list

    out["request_id"] = str(out.get("request_id", "")).strip() or _stable_request_id(out)
    return out


def _dedupe_requests(requests: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen = set()
    for req in requests:
        if not isinstance(req, dict):
            continue
        key = _request_key(req)
        if key in seen:
            continue
        seen.add(key)
        out.append(req)
    return sorted(out, key=_request_sort_key)


def _covered_buckets_from_store(evidence_store: dict) -> set[str]:
    covered: set[str] = set()
    for item in (evidence_store or {}).values():
        if not isinstance(item, dict):
            continue
        bucket = _kind_to_swarm_bucket(str(item.get("kind", "")))
        if bucket in _SWARM_REQUIRED_BUCKETS:
            covered.add(bucket)
    return covered


def _seed_requests_for_missing_buckets(state: dict, missing_buckets: list[str]) -> list[dict]:
    bucket_to_kind = {
        "earnings": "press_release_or_ir",
        "macro": "macro_headline_context",
        "ownership": "ownership_identity",
        "valuation": "valuation_context",
    }
    seeds = []
    baseline = _baseline_seed_requests(state)
    needed = missing_buckets or list(_SWARM_REQUIRED_BUCKETS)
    for bucket in needed:
        target_kind = bucket_to_kind.get(bucket)
        if not target_kind:
            continue
        chosen = next((r for r in baseline if str(r.get("kind", "")).lower() == target_kind), None)
        if not chosen:
            continue
        req = dict(chosen)
        req["source_tag"] = "seed"
        req["expected_bucket"] = bucket
        req["impacted_desks"] = _default_impacted_desks(target_kind, desk=str(req.get("desk", "")))
        seeds.append(req)
    return seeds


def _decision_sign(output: dict) -> int:
    decision = str((output or {}).get("primary_decision", "neutral")).strip().lower()
    if decision == "bullish":
        return 1
    if decision in ("bearish", "avoid"):
        return -1
    return 0


def _disagreement_requests(state: dict, desk_outputs: dict, disagreement_score: float) -> list[dict]:
    if disagreement_score <= 0.5:
        return []
    ticker = state.get("target_ticker", "") or "AAPL"
    macro = desk_outputs.get("macro", {}) or {}
    fundamental = desk_outputs.get("fundamental", {}) or {}
    sentiment = desk_outputs.get("sentiment", {}) or {}
    quant = desk_outputs.get("quant", {}) or {}

    macro_sign = _decision_sign(macro)
    quant_sign = _decision_sign(quant)
    if macro_sign and quant_sign and macro_sign != quant_sign:
        return [{
            "desk": "macro",
            "kind": "macro_headline_context",
            "ticker": "__GLOBAL__",
            "query": "latest macro release drivers and policy stance",
            "priority": 1,
            "recency_days": 7,
            "max_items": 3,
            "rationale": f"disagreement_macro_vs_quant(score={disagreement_score:.3f})",
            "source_tag": "disagreement",
            "expected_bucket": "macro",
            "impacted_desks": ["macro", "sentiment"],
        }]

    if bool(fundamental.get("structural_risk_flag")) and quant_sign > 0:
        return [{
            "desk": "fundamental",
            "kind": "sec_filing",
            "ticker": ticker,
            "query": f"{ticker} latest SEC 10-Q 10-K material disclosures",
            "priority": 1,
            "recency_days": 365,
            "max_items": 3,
            "rationale": f"disagreement_structural_vs_quant(score={disagreement_score:.3f})",
            "source_tag": "disagreement",
            "expected_bucket": "other",
            "impacted_desks": ["fundamental"],
        }]

    if str(sentiment.get("catalyst_risk_level", "")).lower() in ("high", "elevated"):
        return [{
            "desk": "sentiment",
            "kind": "catalyst_event_detail",
            "ticker": ticker,
            "query": f"{ticker} catalyst event details and schedule",
            "priority": 1,
            "recency_days": 14,
            "max_items": 3,
            "rationale": f"disagreement_sentiment_event(score={disagreement_score:.3f})",
            "source_tag": "disagreement",
            "expected_bucket": "other",
            "impacted_desks": ["sentiment", "fundamental"],
        }]

    return [{
        "desk": "orchestrator",
        "kind": "web_search",
        "ticker": ticker,
        "query": f"{ticker} conflicting analyst narrative evidence",
        "priority": 2,
        "recency_days": 30,
        "max_items": 3,
        "rationale": f"disagreement_generic(score={disagreement_score:.3f})",
        "source_tag": "disagreement",
        "expected_bucket": "other",
        "impacted_desks": ["fundamental"],
    }]


def _bounded_swarm_plan_step(
    state: dict,
    *,
    desk_outputs: dict,
    raw_requests: list[dict],
    question_requests: list[dict],
    recovery_requests: list[dict],
    disagreement_score: float,
) -> dict:
    candidates: list[dict] = []
    for req in raw_requests:
        sreq = _sanitize_swarm_request(req, default_source_tag="desk_rule")
        if sreq:
            candidates.append(sreq)
    for req in question_requests:
        r = dict(req)
        r["source_tag"] = "open_questions"
        sreq = _sanitize_swarm_request(r, default_source_tag="open_questions")
        if sreq:
            candidates.append(sreq)

    covered = _covered_buckets_from_store(state.get("evidence_store", {}))
    missing_buckets = [b for b in _SWARM_REQUIRED_BUCKETS if b not in covered]

    if missing_buckets or not candidates:
        candidates.extend(
            s for s in (
                _sanitize_swarm_request(req, default_source_tag="seed")
                for req in _seed_requests_for_missing_buckets(state, missing_buckets)
            )
            if s is not None
        )

    for req in recovery_requests:
        r = dict(req)
        r["source_tag"] = str(r.get("source_tag", "")).strip() or "risk_feedback"
        sreq = _sanitize_swarm_request(r, default_source_tag="risk_feedback")
        if sreq:
            candidates.append(sreq)

    candidates.extend(
        s for s in (
            _sanitize_swarm_request(req, default_source_tag="disagreement")
            for req in _disagreement_requests(state, desk_outputs, disagreement_score)
        )
        if s is not None
    )

    ranked = _dedupe_requests(candidates)[:_MAX_SWARM_CANDIDATES]
    return {
        "candidates": ranked,
        "planned": ranked,
        "covered_buckets": sorted(covered),
        "missing_buckets": missing_buckets,
    }


def _select_rerun_desks(
    *,
    executed_requests: list[dict],
    desk_outputs: dict,
    k: int = _MAX_RERUN_DESKS,
) -> dict:
    if not executed_requests:
        return {"selected_desks": [], "k": k, "reasons": {}, "executed_kinds": []}

    executed_kinds = sorted({
        str(req.get("kind", "")).strip().lower()
        for req in executed_requests
        if str(req.get("kind", "")).strip()
    })
    kind_set = set(executed_kinds)
    candidates: set[str] = set()
    for req in executed_requests:
        impacted = req.get("impacted_desks", [])
        if not isinstance(impacted, list):
            impacted = []
        if not impacted:
            impacted = _default_impacted_desks(str(req.get("kind", "")), desk=str(req.get("desk", "")))
        for desk in impacted:
            if desk in ("macro", "fundamental", "sentiment"):
                candidates.add(desk)
        req_desk = str(req.get("desk", "")).strip().lower()
        if req_desk in ("macro", "fundamental", "sentiment"):
            candidates.add(req_desk)

    if not candidates:
        return {"selected_desks": [], "k": k, "reasons": {}, "executed_kinds": executed_kinds}

    risk_kind_bonus = {
        "fundamental": {"sec_filing", "ownership_identity", "press_release_or_ir"},
        "macro": {"macro_headline_context"},
        "sentiment": {"catalyst_event_detail"},
    }

    scored: list[tuple[str, float, list[str]]] = []
    for desk in sorted(candidates):
        evidence_relevance = 0
        for req in executed_requests:
            impacted = req.get("impacted_desks", [])
            if not isinstance(impacted, list):
                impacted = []
            if desk in impacted:
                evidence_relevance += 1

        open_question_match = 0
        questions = (desk_outputs.get(desk, {}) or {}).get("open_questions", [])
        if isinstance(questions, list):
            for q in questions:
                if not isinstance(q, dict):
                    continue
                qkind = str(q.get("kind", "")).strip().lower()
                if qkind and qkind in kind_set:
                    open_question_match += 1

        risk_relevance = sum(1 for kind in kind_set if kind in risk_kind_bonus.get(desk, set()))
        score = 1.0 * evidence_relevance + 0.6 * open_question_match + 0.2 * risk_relevance

        reasons = []
        if evidence_relevance > 0:
            reasons.append(f"impacted_requests={evidence_relevance}")
        if open_question_match > 0:
            reasons.append(f"open_question_match={open_question_match}")
        if risk_relevance > 0:
            reasons.append(f"risk_relevance={risk_relevance}")
        scored.append((desk, score, reasons))

    ranked = sorted(scored, key=lambda it: (-it[1], it[0]))
    selected = [desk for desk, _, _ in ranked[: max(1, k)] if desk in ("macro", "fundamental", "sentiment")]
    reason_map = {desk: reasons for desk, _, reasons in ranked if desk in selected}
    return {
        "selected_desks": selected,
        "k": max(1, k),
        "reasons": reason_map,
        "executed_kinds": executed_kinds,
    }


def _open_questions_to_requests(state: dict, desk_outputs: dict) -> list[dict]:
    ticker = state.get("target_ticker", "")
    out: list[dict] = []
    for desk in ("macro", "fundamental", "sentiment"):
        output = desk_outputs.get(desk, {}) or {}
        questions = output.get("open_questions", [])
        if not isinstance(questions, list):
            continue
        for q in questions[:5]:
            if not isinstance(q, dict):
                continue
            query_text = str(q.get("q", "")).strip()[:120]
            if not query_text:
                continue
            try:
                priority = int(q.get("priority", 3))
            except (TypeError, ValueError):
                priority = 3
            try:
                recency = int(q.get("recency_days", 30))
            except (TypeError, ValueError):
                recency = 30
            out.append({
                "desk": desk,
                "kind": str(q.get("kind") or _default_kind_for_desk(desk)),
                "ticker": ticker,
                "query": query_text,
                "priority": max(1, min(5, priority)),
                "recency_days": max(1, min(365, recency)),
                "max_items": 3,
                "rationale": str(q.get("why", "open_question_generated")).strip() or "open_question_generated",
                "source_tag": "open_questions",
            })
    return out


def _desk_followups_to_actions(desk_outputs: dict) -> list[dict]:
    out: list[dict] = []
    for desk in ("macro", "fundamental", "sentiment"):
        output = desk_outputs.get(desk, {}) or {}
        followups = output.get("followups", [])
        if not isinstance(followups, list):
            continue
        for fu in followups[:5]:
            if not isinstance(fu, dict):
                continue
            action_type = str(fu.get("type", "")).strip()
            detail = str(fu.get("detail", "")).strip()
            params = fu.get("params") if isinstance(fu.get("params"), dict) else {}
            if action_type and detail:
                out.append({"type": action_type, "detail": detail, "params": params})
    return out


def _derive_user_handoff(
    state: dict,
    *,
    decision: dict,
    recovery: dict,
    run_research: bool,
) -> tuple[bool, list[dict]]:
    if run_research:
        return False, []

    reason = str(decision.get("reason", "")).strip()
    issues = recovery.get("issues", []) if isinstance(recovery, dict) else []
    if not isinstance(issues, list):
        issues = []

    hard_stop = reason in {
        "max_research_rounds",
        "run_budget_exhausted",
        "low_added_evidence_delta",
        "low_information_gain",
        "unresolved_core_questions",
    }
    if not hard_stop:
        return False, []

    items: list[dict] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        code = str(issue.get("code", "")).strip()
        if code not in _BLOCKING_ISSUE_CODES and code != "risk_llm_enrichment_failed":
            continue
        items.append(
            {
                "code": code,
                "desk": str(issue.get("desk", "")).strip() or "unknown",
                "detail": str(issue.get("detail", "")).strip()[:200],
                "suggested_action": _USER_ACTION_HINTS.get(code, "운영자가 수동 점검 필요"),
            }
        )

    # 근거 점수가 낮은 상태로 라운드 종료되면 수동 점검 필요로 본다.
    if not items and int(state.get("evidence_score", 0)) < 55:
        items.append(
            {
                "code": "low_evidence_after_stop",
                "desk": "orchestrator",
                "detail": f"research stop={reason}, evidence_score={state.get('evidence_score', 0)}",
                "suggested_action": "API 키/네트워크/리서치 예산을 확인 후 재실행",
            }
        )

    return bool(items), items[:5]


def _baseline_seed_requests(state: dict) -> list[dict]:
    ticker = state.get("target_ticker", "") or "AAPL"
    asset_type = str((state.get("asset_type_by_ticker", {}) or {}).get(ticker, "")).strip().upper() or _infer_asset_type(ticker)
    base = [
        {
            "desk": "fundamental",
            "kind": "press_release_or_ir",
            "ticker": ticker,
            "query": f"{ticker} investor relations press release earnings date",
            "priority": 2,
            "recency_days": 30,
            "max_items": 3,
            "rationale": "baseline_seed_earnings",
        },
        {
            "desk": "macro",
            "kind": "macro_headline_context",
            "ticker": "__GLOBAL__",
            "query": "federal reserve treasury macro release market context",
            "priority": 2,
            "recency_days": 7,
            "max_items": 3,
            "rationale": "baseline_seed_macro",
        },
        {
            "desk": "fundamental",
            "kind": "valuation_context",
            "ticker": ticker,
            "query": f"{ticker} valuation historical peers context",
            "priority": 2,
            "recency_days": 365,
            "max_items": 3,
            "rationale": "baseline_seed_valuation",
        },
    ]
    if asset_type in {"ETF", "INDEX"}:
        base.extend(
            [
                {
                    "desk": "fundamental",
                    "kind": "web_search",
                    "ticker": ticker,
                    "query": f"{ticker} holdings top 10 sector weights",
                    "priority": 2,
                    "recency_days": 30,
                    "max_items": 3,
                    "rationale": "baseline_seed_etf_holdings",
                },
                {
                    "desk": "fundamental",
                    "kind": "web_search",
                    "ticker": ticker,
                    "query": f"{ticker} etf flows creation redemption tracking error liquidity",
                    "priority": 2,
                    "recency_days": 30,
                    "max_items": 3,
                    "rationale": "baseline_seed_etf_flows",
                },
            ]
        )
    else:
        base.extend(
            [
                {
                    "desk": "fundamental",
                    "kind": "ownership_identity",
                    "ticker": ticker,
                    "query": f"{ticker} Form 4 13F 13D 13G ownership",
                    "priority": 2,
                    "recency_days": 90,
                    "max_items": 3,
                    "rationale": "baseline_seed_ownership",
                },
                {
                    "desk": "fundamental",
                    "kind": "sec_filing",
                    "ticker": ticker,
                    "query": f"{ticker} latest SEC 10-Q 10-K filing",
                    "priority": 5,
                    "recency_days": 365,
                    "max_items": 3,
                    "rationale": "baseline_seed_sec_filing",
                },
            ]
        )
    return base


def research_router_node(state: InvestmentState) -> dict:
    """Research trigger router. plan_additional_research는 여기서만 호출."""
    _log(state, "research_router", "enter")
    desk_outputs = _desk_outputs_for_policy(state)
    raw_requests = _merge_request_sources(state)
    question_requests = _open_questions_to_requests(state, desk_outputs)
    followup_actions = _desk_followups_to_actions(desk_outputs)
    disagreement = compute_disagreement_score(desk_outputs)

    score_block = compute_evidence_score(
        state.get("evidence_store", {}),
        state.get("as_of", ""),
    )

    recovery = plan_runtime_recovery(state, desk_outputs)
    merged_backlog = _merge_actions(state.get("task_backlog", []), followup_actions, recovery.get("actions", []))
    if recovery.get("issues") or recovery.get("actions"):
        telemetry.log_event(
            state.get("run_id", ""),
            node_name="autonomy_planner",
            iteration=state.get("iteration_count", 0),
            phase="exit",
            outputs_summary={
                "issues": len(recovery.get("issues", [])),
                "actions": len(recovery.get("actions", [])),
                "added_requests": len(recovery.get("evidence_requests", [])),
                "notes": recovery.get("notes", []),
            },
        )

    swarm = _bounded_swarm_plan_step(
        state,
        desk_outputs=desk_outputs,
        raw_requests=raw_requests,
        question_requests=question_requests,
        recovery_requests=list(recovery.get("evidence_requests", [])),
        disagreement_score=disagreement,
    )
    swarm_candidates = list(swarm.get("candidates", []))[:_MAX_SWARM_CANDIDATES]
    planned = list(swarm_candidates)
    covered_buckets = list(swarm.get("covered_buckets", []))
    missing_buckets = list(swarm.get("missing_buckets", []))

    if _plan_research_impl is not None and swarm_candidates:
        llm_planned = _plan_research_impl(
            swarm_candidates,
            desk_outputs,
            state.get("user_request", ""),
            policy_state={
                "max_web_queries_per_run": MAX_WEB_QUERIES_PER_RUN,
                "max_web_queries_per_ticker": MAX_WEB_QUERIES_PER_TICKER,
            },
        )
        planner_base = llm_planned if llm_planned is not None else swarm_candidates
        normalized_planned: list[dict] = []
        for req in planner_base:
            req_copy = dict(req)
            if not req_copy.get("source_tag"):
                req_copy["source_tag"] = "llm_extra"
            sreq = _sanitize_swarm_request(req_copy, default_source_tag="llm_extra")
            if sreq:
                normalized_planned.append(sreq)
        planned = _dedupe_requests(normalized_planned) if normalized_planned else list(swarm_candidates)

    planned = _dedupe_requests(planned)

    rid = state.get("run_id", "")
    if rid:
        source_tag_counts: dict[str, int] = {}
        for req in planned:
            tag = str(req.get("source_tag", "unknown"))
            source_tag_counts[tag] = int(source_tag_counts.get(tag, 0)) + 1
        selected_kinds = sorted({str(req.get("kind", "")) for req in planned if req.get("kind")})
        telemetry.log_event(
            rid,
            node_name="bounded_swarm_planner",
            iteration=state.get("iteration_count", 0),
            phase="exit",
            outputs_summary={
                "research_round": int(state.get("research_round", 0)),
                "evidence_score": int(score_block.get("score", 0)),
                "missing_buckets": missing_buckets,
                "covered_buckets": covered_buckets,
                "candidate_count": len(swarm_candidates),
                "selected_count": len(planned),
                "selected_kinds": selected_kinds[:6],
                "source_tag_counts": source_tag_counts,
                "budget": {
                    "per_run": MAX_WEB_QUERIES_PER_RUN,
                    "per_ticker": MAX_WEB_QUERIES_PER_TICKER,
                },
            },
        )

    policy_input = dict(state)
    policy_input["evidence_score"] = score_block["score"]
    decision = should_run_web_research(
        state=policy_input,
        desk_outputs=desk_outputs,
        disagreement_score=disagreement,
        evidence_requests=planned,
        user_request=state.get("user_request", ""),
        max_web_queries_per_run=MAX_WEB_QUERIES_PER_RUN,
        max_web_queries_per_ticker=MAX_WEB_QUERIES_PER_TICKER,
    )

    run_research = bool(decision.get("run", False))
    decision_reason = str(decision.get("reason", "no_trigger"))
    need_score = decision.get("research_need_score")
    impact_score = decision.get("impact_score")
    uncertainty_score = decision.get("uncertainty_score")
    allowed_count = len(decision.get("allowed_requests", []) or [])

    print(f"\n🧭 RESEARCH ROUTER  (iter #{state.get('iteration_count', 1)})")
    print(
        "   Decision: "
        f"run={run_research} | reason={decision_reason} | "
        f"need={need_score} (impact={impact_score}, uncertainty={uncertainty_score})"
    )
    print(
        "   Evidence Score: "
        f"{score_block.get('score', 0)} "
        f"(coverage={score_block.get('coverage', 0)}, freshness={score_block.get('freshness', 0)}, "
        f"trust={score_block.get('source_trust', 0)}, penalty={score_block.get('contradiction_penalty', 0)})"
    )
    print(
        "   Plan: "
        f"candidates={len(swarm_candidates)} planned={len(planned)} allowed={allowed_count} "
        f"missing={missing_buckets if missing_buckets else '[]'}"
    )
    _ops_log(
        state,
        "research_router",
        "research decision",
        [
            f"run={run_research} reason={decision_reason}",
            f"need_score={need_score} (impact={impact_score}, uncertainty={uncertainty_score}, disagreement={disagreement:.3f})",
            (
                "evidence_score="
                f"{score_block.get('score', 0)} "
                f"(coverage={score_block.get('coverage', 0)}, freshness={score_block.get('freshness', 0)}, "
                f"trust={score_block.get('source_trust', 0)}, penalty={score_block.get('contradiction_penalty', 0)})"
            ),
            (
                f"plan=candidates:{len(swarm_candidates)} planned:{len(planned)} allowed:{allowed_count} "
                f"missing_buckets={missing_buckets}"
            ),
            f"selected_kinds={_list_preview(sorted({str(req.get('kind', '')) for req in planned if req.get('kind')}), max_items=6, max_len=180) or 'n/a'}",
        ],
    )
    user_action_required, user_action_items = _derive_user_handoff(
        state,
        decision=decision,
        recovery=recovery,
        run_research=run_research,
    )
    out = {
        "evidence_score": score_block["score"],
        "evidence_requests": planned,
        "task_backlog": merged_backlog,
        "_swarm_candidates": swarm_candidates[:_MAX_SWARM_CANDIDATES],
        "_swarm_plan": planned,
        "_covered_buckets": covered_buckets,
        "_run_research": run_research,
        "_research_plan": decision.get("allowed_requests", []),
        "research_stop_reason": "" if run_research else decision.get("reason", "no_trigger"),
        "user_action_required": user_action_required,
        "user_action_items": user_action_items,
        "trace": [{
            "node": "research_router",
            "run": run_research,
            "reason": decision.get("reason", ""),
            "planned_requests": len(planned),
            "followup_actions": len(followup_actions),
            "autonomy_issues": len(recovery.get("issues", [])),
            "autonomy_actions": len(recovery.get("actions", [])),
            "research_need_score": decision.get("research_need_score"),
            "impact_score": decision.get("impact_score"),
            "uncertainty_score": decision.get("uncertainty_score"),
            "covered_buckets": covered_buckets,
            "missing_buckets": missing_buckets,
            "swarm_candidates": len(swarm_candidates),
            "user_action_required": user_action_required,
        }],
    }
    if _graph_trace_enabled():
        allowed = list(out.get("_research_plan", []))
        kinds = sorted({str(req.get("kind", "")) for req in allowed if req.get("kind")})
        reason = out["research_stop_reason"] if out["research_stop_reason"] else str(decision.get("reason", ""))
        _graph_trace(
            "research_router decision: "
            f"run_research={run_research}, reason={reason}, "
            f"planned={len(planned)}, allowed={len(allowed)}, allowed_kinds={kinds[:6]}"
        )
    if user_action_required:
        telemetry.log_event(
            state.get("run_id", ""),
            node_name="human_handoff",
            iteration=state.get("iteration_count", 0),
            phase="exit",
            outputs_summary={
                "reason": out["research_stop_reason"],
                "items": user_action_items,
            },
        )
    if not run_research:
        telemetry.log_event(
            state.get("run_id", ""),
            node_name="research_round",
            iteration=state.get("iteration_count", 0),
            phase="exit",
            outputs_summary={
                "research_round": state.get("research_round", 0),
                "queries_executed": 0,
                "last_research_delta": state.get("last_research_delta", 0),
                "evidence_score": score_block["score"],
                "stop_reason": out["research_stop_reason"],
            },
        )
    _log(state, "research_router", "exit", {"run": run_research, "reason": out["trace"][0]["reason"]})
    return out


def _allowlist_for_kind(kind: str) -> list[str]:
    kind = (kind or "").lower()
    if kind == "macro_headline_context":
        return list(WebResearchProvider.ALLOWLIST_MACRO)
    if kind in ("ownership_identity", "sec_filing"):
        return list(WebResearchProvider.ALLOWLIST_FILINGS)
    if kind in ("press_release_or_ir", "web_search"):
        return list(WebResearchProvider.ALLOWLIST_EARNINGS)
    return list(WebResearchProvider.ALLOWLIST_EARNINGS + WebResearchProvider.ALLOWLIST_FILINGS + WebResearchProvider.ALLOWLIST_MACRO)


def _reindex_store_by_canonical(store: dict) -> tuple[dict[str, dict], set[str]]:
    out: dict[str, dict] = {}
    seen_canonical: set[str] = set()
    for item in (store or {}).values():
        if not isinstance(item, dict):
            continue
        canonical = _canonicalize_url(str(item.get("url", "")))
        if not canonical or canonical in seen_canonical:
            continue
        key = _evidence_store_key_from_url(canonical)
        if not key:
            continue
        row = dict(item)
        row["canonical_url"] = canonical
        row["url"] = canonical
        row["hash"] = key
        out[key] = row
        seen_canonical.add(canonical)
    return out, seen_canonical


def _quality_filter_items_for_request(req: dict, items: list[dict]) -> tuple[list[dict], dict]:
    accepted: list[dict] = []
    rejected_landing = 0
    rejected_mismatch = 0
    for item in items or []:
        if not isinstance(item, dict):
            continue
        canonical = _canonicalize_url(str(item.get("url", "")))
        if not canonical:
            continue
        row = dict(item)
        row["canonical_url"] = canonical
        row["url"] = canonical
        row["hash"] = _evidence_store_key_from_url(canonical)
        match = _item_matches_request(row, req)
        landing = _is_landing_page_candidate(canonical, str(row.get("title", "")))
        published = str(row.get("published_at", "")).strip()
        snippet = str(row.get("snippet", "")).strip()
        has_min_doc = bool(published) and len(snippet) >= 40
        if landing and (not match or not has_min_doc):
            rejected_landing += 1
            continue
        if not match:
            rejected_mismatch += 1
            continue
        accepted.append(row)
    return accepted, {
        "rejected_landing": rejected_landing,
        "rejected_mismatch": rejected_mismatch,
        "accepted": len(accepted),
    }


def _resolve_request_with_priority(
    req: dict,
    *,
    sec: SECEdgarProvider,
    web: WebResearchProvider,
    perplexity: PerplexitySearchProvider | None,
    as_of: str,
) -> tuple[list[dict], str]:
    kind = str(req.get("kind", "web_search"))
    ticker = str(req.get("ticker", "")).upper()
    query = str(req.get("query", "")).strip()
    desk = str(req.get("desk", "orchestrator"))
    recency_days = int(req.get("recency_days", 30))
    max_items = int(req.get("max_items", 5))
    allowlist = _allowlist_for_kind(kind)

    if kind == "ownership_identity":
        sec_out = sec.get_ownership_identity(ticker, as_of=as_of)
        if sec_out.get("data_ok") and sec_out.get("items"):
            return list(sec_out["items"])[:max_items], "sec_forms"
        items = web.collect_evidence(
            kind=kind, ticker=ticker, query=query, recency_days=recency_days, max_items=max_items,
            allowlist=allowlist, desk=desk, resolver_path="web_fallback_ownership",
        )
        if items:
            return items, "web_fallback_ownership"
        if perplexity is not None:
            items = perplexity.collect_evidence(
                kind=kind, ticker=ticker, query=query, recency_days=recency_days, max_items=max_items,
                allowlist=allowlist, desk=desk, resolver_path="perplexity_fallback_ownership",
            )
            if items:
                return items, "perplexity_fallback_ownership"
        return [], "web_fallback_ownership"

    if kind == "press_release_or_ir":
        sec_out = sec.get_8k_exhibits(ticker, as_of=as_of)
        if sec_out.get("data_ok") and sec_out.get("items"):
            return list(sec_out["items"])[:max_items], "sec_8k"
        ir_urls = []
        if req.get("ir_domain"):
            ir_urls.append(f"https://{str(req['ir_domain']).strip('/')}")
        items = web.collect_evidence(
            kind=kind, ticker=ticker, query=query, recency_days=recency_days, max_items=max_items,
            allowlist=allowlist, official_urls=ir_urls, desk=desk, resolver_path="ir_domain",
        )
        if items:
            return items, "ir_domain"
        items = web.collect_evidence(
            kind=kind, ticker=ticker, query=query, recency_days=recency_days, max_items=max_items,
            allowlist=allowlist, desk=desk, resolver_path="newsapi",
        )
        if items:
            return items, "newsapi"
        if perplexity is not None:
            items = perplexity.collect_evidence(
                kind=kind, ticker=ticker, query=query, recency_days=recency_days, max_items=max_items,
                allowlist=allowlist, desk=desk, resolver_path="perplexity_fallback_ir",
            )
            if items:
                return items, "perplexity_fallback_ir"
        return [], "newsapi"

    if kind == "macro_headline_context":
        query_l = _normalize_query_text(query)
        event_like = any(
            tok in query_l
            for tok in (
                "iran", "hormuz", "airstrike", "attack", "geopolitical", "wti", "brent", "oil", "war", "conflict",
                "지정학", "호르무즈", "공습", "원유", "유가",
            )
        )
        if event_like:
            items = web.collect_evidence(
                kind=kind, ticker=ticker, query=query, recency_days=recency_days, max_items=max_items,
                allowlist=allowlist, desk=desk, resolver_path="event_news_priority",
            )
            if items:
                return items, "event_news_priority"
        official_urls = [
            "https://www.federalreserve.gov/newsevents/pressreleases.htm",
            "https://home.treasury.gov/news/press-releases",
            "https://www.bls.gov/news.release/",
            "https://www.bea.gov/news",
            "https://www.nyfed.org/newsevents",
        ]
        items = web.collect_evidence(
            kind=kind, ticker=ticker, query=query, recency_days=recency_days, max_items=max_items,
            allowlist=allowlist, official_urls=official_urls, desk=desk, resolver_path="official_release",
        )
        if items:
            return items, "official_release"
        items = web.collect_evidence(
            kind=kind, ticker=ticker, query=query, recency_days=recency_days, max_items=max_items,
            allowlist=allowlist, desk=desk, resolver_path="newsapi_supplement",
        )
        if items:
            return items, "newsapi_supplement"
        if perplexity is not None:
            items = perplexity.collect_evidence(
                kind=kind, ticker=ticker, query=query, recency_days=recency_days, max_items=max_items,
                allowlist=allowlist, desk=desk, resolver_path="perplexity_fallback_macro",
            )
            if items:
                return items, "perplexity_fallback_macro"
        return [], "newsapi_supplement"

    items = web.collect_evidence(
        kind=kind, ticker=ticker, query=query, recency_days=recency_days, max_items=max_items,
        allowlist=allowlist, desk=desk, resolver_path="default_web",
    )
    if items:
        return items, "default_web"
    if perplexity is not None:
        items = perplexity.collect_evidence(
            kind=kind, ticker=ticker, query=query, recency_days=recency_days, max_items=max_items,
            allowlist=allowlist, desk=desk, resolver_path="perplexity_fallback_default",
        )
        if items:
            return items, "perplexity_fallback_default"
    return [], "default_web"


def research_executor_node(state: InvestmentState) -> dict:
    """
    Execute web research requests.
    last_research_delta = new unique hashes added in this executor round.
    """
    _log(state, "research_executor", "enter")
    plan = list(state.get("_research_plan", []))
    if not plan:
        _log(state, "research_executor", "exit", {"queries_executed": 0, "delta": 0})
        return {
            "last_research_delta": 0,
            "_run_research": False,
            "_executed_requests": [],
            "_rerun_plan": {"selected_desks": [], "k": _MAX_RERUN_DESKS, "reasons": {}, "executed_kinds": []},
            "_evidence_delta_kinds": {},
        }

    as_of = state.get("as_of", "")
    mode = state.get("mode", "mock")
    store, existing_canonical = _reindex_store_by_canonical(dict(state.get("evidence_store", {})))
    before = set(store.keys())
    before_buckets = _covered_buckets_from_store(store)

    sec = SECEdgarProvider()
    web = WebResearchProvider(mode=mode)
    perplexity = PerplexitySearchProvider(mode=mode)

    queries_executed = 0
    executed_requests: list[dict] = []
    new_hashes: set[str] = set()
    new_canonical: set[str] = set()
    evidence_delta_kinds: dict[str, int] = {}
    resolved_requests = 0
    if _graph_trace_enabled():
        request_preview = [
            f"{str(req.get('desk', ''))}:{str(req.get('kind', ''))}:{_short_text(req.get('query', ''), 90)}"
            for req in plan[:6]
        ]
        _graph_trace(f"research_executor start: requests={len(plan)}, preview={request_preview}")

    for req in sorted(plan, key=lambda r: r.get("priority", 5)):
        items, resolver_path = _resolve_request_with_priority(
            req,
            sec=sec,
            web=web,
            perplexity=perplexity,
            as_of=as_of,
        )
        queries_executed += 1
        accepted_items, quality_meta = _quality_filter_items_for_request(req, items)
        if accepted_items:
            resolved_requests += 1

        req_copy = dict(req)
        req_copy["resolver_path"] = resolver_path
        req_copy["kind"] = str(req_copy.get("kind", "web_search")).strip().lower()
        if not req_copy.get("request_id"):
            req_copy["request_id"] = _stable_request_id(req_copy)
        impacted = req_copy.get("impacted_desks")
        if not isinstance(impacted, list) or not impacted:
            req_copy["impacted_desks"] = _default_impacted_desks(
                str(req_copy.get("kind", "")),
                desk=str(req_copy.get("desk", "")),
            )
        try:
            req_priority = int(req_copy.get("priority", 3))
        except (TypeError, ValueError):
            req_priority = 3
        executed_requests.append(
            {
                "request_id": req_copy.get("request_id"),
                "desk": str(req_copy.get("desk", "")).strip().lower() or "orchestrator",
                "kind": req_copy["kind"],
                "ticker": str(req_copy.get("ticker", "")).strip().upper(),
                "priority": max(1, min(5, req_priority)),
                "resolver_path": resolver_path,
                "n_items_raw": int(len(items)),
                "n_items": int(len(accepted_items)),
                "rejected_landing": int(quality_meta.get("rejected_landing", 0)),
                "rejected_mismatch": int(quality_meta.get("rejected_mismatch", 0)),
                "impacted_desks": list(req_copy.get("impacted_desks", [])),
            }
        )

        for item in accepted_items:
            canonical = str(item.get("canonical_url", "")).strip()
            h = str(item.get("hash", "")).strip() or _evidence_store_key_from_url(canonical)
            if not h:
                continue
            if canonical and canonical in existing_canonical:
                continue
            is_new = h not in before and h not in new_hashes
            item["resolver_path"] = item.get("resolver_path") or resolver_path
            store[h] = item
            if is_new:
                new_hashes.add(h)
                if canonical:
                    existing_canonical.add(canonical)
                    new_canonical.add(canonical)
                kind = str(item.get("kind", req_copy.get("kind", "web_search"))).strip().lower() or "web_search"
                evidence_delta_kinds[kind] = int(evidence_delta_kinds.get(kind, 0)) + 1

    delta = len(new_canonical)
    after_buckets = _covered_buckets_from_store(store)
    bucket_delta = len(set(after_buckets) - set(before_buckets))
    resolved_request_ratio = (resolved_requests / queries_executed) if queries_executed > 0 else 0.0
    score_block = compute_evidence_score(store, as_of)
    research_round = int(state.get("research_round", 0)) + 1

    rerun_plan = _select_rerun_desks(
        executed_requests=executed_requests,
        desk_outputs=_desk_outputs_for_policy(state),
        k=_MAX_RERUN_DESKS,
    )
    rerun_desks = set(rerun_plan.get("selected_desks", []))

    ct = dict(state.get("completed_tasks", {}))
    for desk in rerun_desks:
        ct[desk] = False

    audit = dict(state.get("audit", {}))
    audit_research = dict((audit.get("research") or {}))
    by_ticker = dict(audit_research.get("web_queries_by_ticker", {}))
    for req in plan:
        t = str(req.get("ticker", "")).upper() or "__GLOBAL__"
        by_ticker[t] = int(by_ticker.get(t, 0)) + 1
    audit_research["web_queries_total"] = int(audit_research.get("web_queries_total", 0)) + queries_executed
    audit_research["web_queries_by_ticker"] = by_ticker
    audit["research"] = audit_research

    rid = state.get("run_id", "")
    if rid:
        telemetry.log_event(
            rid,
            node_name="rerun_selector",
            iteration=state.get("iteration_count", 0),
            phase="exit",
            outputs_summary={
                "selected_desks": sorted(rerun_desks),
                "executed_requests": len(executed_requests),
                "executed_kinds": rerun_plan.get("executed_kinds", []),
                "k": _MAX_RERUN_DESKS,
            },
        )

    stop_reason = ""
    if score_block["score"] >= 75:
        stop_reason = "evidence_score_enough"
    elif delta <= 0 and bucket_delta <= 0:
        stop_reason = "low_information_gain"
    elif resolved_request_ratio <= 0.0:
        stop_reason = "unresolved_core_questions"

    print(f"\n🔎 RESEARCH EXECUTOR  (iter #{state.get('iteration_count', 1)})")
    print(
        "   Executed: "
        f"{queries_executed} requests | delta={delta} | bucket_delta={bucket_delta} | "
        f"resolved={resolved_requests}/{queries_executed} | evidence_score={score_block['score']} | round={research_round}"
    )
    print(
        "   Kinds: "
        f"{rerun_plan.get('executed_kinds', []) if rerun_plan.get('executed_kinds') else '[]'}"
    )
    print(
        "   Rerun: "
        f"selected={sorted(rerun_desks)} (k={_MAX_RERUN_DESKS}) "
        f"reasons={rerun_plan.get('reasons', {})}"
    )
    _ops_log(
        state,
        "research_executor",
        "execution result",
        [
            f"executed_requests={queries_executed} delta={delta} research_round={research_round}",
            f"bucket_delta={bucket_delta} resolved_request_ratio={resolved_request_ratio:.2f}",
            f"executed_kinds={rerun_plan.get('executed_kinds', [])}",
            f"rerun_selected={sorted(rerun_desks)} k={_MAX_RERUN_DESKS}",
            f"rerun_reasons={rerun_plan.get('reasons', {})}",
            f"evidence_delta_kinds={evidence_delta_kinds}",
            f"stop_reason={stop_reason or 'continue'}",
        ],
    )
    if _graph_trace_enabled():
        _graph_trace(
            "research_executor result: "
            f"round={research_round}, executed={queries_executed}, delta={delta}, "
            f"bucket_delta={bucket_delta}, resolved_ratio={resolved_request_ratio:.2f}, "
            f"score={score_block['score']}, rerun_desks={sorted(rerun_desks)}, "
            f"rerun_reasons={rerun_plan.get('reasons', {})}, executed_kinds={rerun_plan.get('executed_kinds', [])}"
        )

    telemetry.log_event(
        state.get("run_id", ""),
        node_name="research_round",
        iteration=state.get("iteration_count", 0),
        phase="exit",
        outputs_summary={
            "research_round": research_round,
            "queries_executed": queries_executed,
            "last_research_delta": delta,
            "bucket_delta": bucket_delta,
            "resolved_request_ratio": round(resolved_request_ratio, 4),
            "evidence_score": score_block["score"],
            "rerun_desks": sorted(rerun_desks),
            "stop_reason": stop_reason,
        },
    )

    out = {
        "evidence_store": store,
        "evidence_score": score_block["score"],
        "last_research_delta": delta,
        "research_round": research_round,
        "research_stop_reason": stop_reason,
        "_research_info_gain": {
            "unique_canonical_delta": delta,
            "bucket_delta": bucket_delta,
            "resolved_requests": resolved_requests,
            "resolved_request_ratio": round(resolved_request_ratio, 4),
        },
        "completed_tasks": ct,
        "audit": audit,
        "_run_research": False,
        "_research_plan": [],
        "_executed_requests": executed_requests[:MAX_WEB_QUERIES_PER_RUN],
        "_rerun_plan": rerun_plan,
        "_evidence_delta_kinds": evidence_delta_kinds,
        "trace": [{
            "node": "research_executor",
            "research_round": research_round,
            "queries_executed": queries_executed,
            "last_research_delta": delta,
            "bucket_delta": bucket_delta,
            "resolved_request_ratio": round(resolved_request_ratio, 4),
            "evidence_score": score_block["score"],
            "rerun_desks": sorted(rerun_desks),
        }],
    }
    _log(state, "research_executor", "exit", {"queries_executed": queries_executed, "delta": delta, "score": score_block["score"]})
    return out


def research_barrier_node(state: InvestmentState) -> dict:
    """
    Carry-forward barrier:
      - rerun되지 않은 desk 결과는 건드리지 않음
      - evidence_requests는 reducer(append+dedupe)에 의해 유지/병합
    """
    return {}


def macro_analyst_research_node(state: InvestmentState) -> dict:
    return macro_analyst_node(state)


def fundamental_analyst_research_node(state: InvestmentState) -> dict:
    return fundamental_analyst_node(state)


def sentiment_analyst_research_node(state: InvestmentState) -> dict:
    return sentiment_analyst_node(state)


def quant_analyst_research_node(state: InvestmentState) -> dict:
    return quant_analyst_node(state)


def risk_manager_node(state: InvestmentState) -> dict:
    """⑥ Risk Manager (5-Gate, barrier 후, iteration 중복 방지)."""
    _log(state, "risk_manager", "enter")
    iteration = state.get("iteration_count", 0)

    existing = state.get("risk_assessment", {})
    if existing.get("_iteration_evaluated") == iteration:
        _log(state, "risk_manager", "exit", {"skipped": True})
        return {}

    if _risk_node_impl is not None:
        result = _risk_node_impl(state)
    else:
        result = {
            "risk_assessment": {
                "grade": "Low",
                "risk_decision": {
                    "per_ticker_decisions": {},
                    "portfolio_actions": {},
                    "orchestrator_feedback": {"required": False, "reasons": [], "detail": "Mock pass."},
                },
                "summary": "Mock risk — all clear.",
                "evidence": [],
                "status": "ok",
            }
        }

    ra = result.get("risk_assessment", {})
    ra["_iteration_evaluated"] = iteration
    ra.setdefault("summary", "Risk assessment completed.")
    ra.setdefault("evidence", [])
    ra.setdefault("status", "ok")
    result["risk_assessment"] = ra
    rd = ra.get("risk_decision", {}) if isinstance(ra.get("risk_decision"), dict) else {}
    fb = rd.get("orchestrator_feedback", {}) if isinstance(rd.get("orchestrator_feedback"), dict) else {}

    print(f"\n🛡️ RISK SUMMARY  (iter #{iteration})")
    print(
        "   Grade: "
        f"{ra.get('grade', 'N/A')} | feedback_required={fb.get('required', False)} | "
        f"feedback_reasons={fb.get('reasons', []) if fb.get('reasons') else '[]'}"
    )
    _ops_log(
        state,
        "risk_manager",
        "risk decision",
        [
            f"grade={ra.get('grade', 'N/A')}",
            f"feedback_required={fb.get('required', False)} reasons={fb.get('reasons', []) if fb.get('reasons') else []}",
            f"summary={_short_text(ra.get('summary', ''), max_len=180)}",
        ],
    )

    _log(
        state,
        "risk_manager",
        "exit",
        {
            "grade": ra.get("grade"),
            "llm_enrichment_status": (ra.get("risk_decision", {}) or {}).get("_llm_enrichment_status", ""),
        },
    )
    return result


def report_writer_node(state: InvestmentState) -> dict:
    """⑦ Report Writer — IC Memo + Red Team."""
    _log(state, "report_writer", "enter")

    if _report_node_impl is not None:
        result = _report_node_impl(state)
    else:
        ticker = state.get("target_ticker", "N/A")
        result = {"final_report": f"# Mock IC Memo — {ticker}\n\n(Mock mode)"}

    report = str(result.get("final_report", "") or "")
    first_line = report.splitlines()[0] if report else ""
    print(
        "   [REPORT] "
        f"len={len(report)} title={_short_text(first_line, max_len=120) or 'n/a'}"
    )
    _ops_log(
        state,
        "report_writer",
        "report generated",
        [
            f"report_len={len(report)}",
            f"title={_short_text(first_line, max_len=180) or 'n/a'}",
        ],
    )

    _log(state, "report_writer", "exit", {"report_len": len(result.get("final_report", ""))})
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Router
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def research_router(state: InvestmentState) -> Literal["research_executor", "risk_manager"]:
    if state.get("_run_research", False):
        reason = ""
        for item in reversed(state.get("trace", []) or []):
            if isinstance(item, dict) and item.get("node") == "research_router":
                reason = str(item.get("reason", ""))
                break
        print(
            "   [ROUTER] research_router -> research_executor "
            f"(reason={reason}, allowed_requests={len(state.get('_research_plan', []) or [])})"
        )
        _graph_trace(
            "route research_router -> research_executor "
            f"(reason={reason}, allowed_requests={len(state.get('_research_plan', []) or [])})"
        )
        return "research_executor"
    reason = str(state.get("research_stop_reason", ""))
    print(f"   [ROUTER] research_router -> risk_manager (stop_reason={reason})")
    _graph_trace(f"route research_router -> risk_manager (stop_reason={reason})")
    return "risk_manager"


def risk_router(state: InvestmentState) -> Literal["orchestrator", "report_writer"]:
    risk = state.get("risk_assessment", {})
    grade = risk.get("grade", "Low")
    iteration = state.get("iteration_count", 0)
    rd = risk.get("risk_decision", risk)
    fb = rd.get("orchestrator_feedback", {})

    if grade == "High" and fb.get("required", False) and iteration < MAX_ITERATIONS:
        print(
            "   [ROUTER] risk_router -> orchestrator "
            f"(grade={grade}, iteration={iteration}, reasons={fb.get('reasons', [])})"
        )
        _graph_trace(
            "route risk_router -> orchestrator "
            f"(grade={grade}, iteration={iteration}, feedback_reasons={fb.get('reasons', [])})"
        )
        return "orchestrator"
    print(
        "   [ROUTER] risk_router -> report_writer "
        f"(grade={grade}, iteration={iteration}, feedback_required={fb.get('required', False)})"
    )
    _graph_trace(
        "route risk_router -> report_writer "
        f"(grade={grade}, iteration={iteration}, feedback_required={fb.get('required', False)})"
    )
    return "report_writer"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Graph Assembly
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_investment_graph() -> StateGraph:
    """
    V4 Graph:
      START → orchestrator → [4 desks parallel] → barrier → hedge_lite_builder → research_router
        research_router -> risk_manager
        research_router -> research_executor -> [4 desks(parallel,research)] -> research_barrier
                         -> hedge_lite_builder -> research_router
      risk_manager -> router -> orchestrator/report_writer
    """
    g = StateGraph(InvestmentState)

    g.add_node("orchestrator", orchestrator_node)
    g.add_node("macro_analyst", macro_analyst_node)
    g.add_node("fundamental_analyst", fundamental_analyst_node)
    g.add_node("sentiment_analyst", sentiment_analyst_node)
    g.add_node("quant_analyst", quant_analyst_node)
    g.add_node("barrier", barrier_node)
    g.add_node("hedge_lite_builder", hedge_lite_builder_node)
    g.add_node("research_router", research_router_node)
    g.add_node("research_executor", research_executor_node)
    g.add_node("research_barrier", research_barrier_node)
    g.add_node("macro_analyst_research", macro_analyst_research_node)
    g.add_node("fundamental_analyst_research", fundamental_analyst_research_node)
    g.add_node("sentiment_analyst_research", sentiment_analyst_research_node)
    g.add_node("quant_analyst_research", quant_analyst_research_node)
    g.add_node("risk_manager", risk_manager_node)
    g.add_node("report_writer", report_writer_node)

    g.add_edge(START, "orchestrator")

    # Fan-out: orchestrator → 4 desks
    for desk in ["macro_analyst", "fundamental_analyst", "sentiment_analyst", "quant_analyst"]:
        g.add_edge("orchestrator", desk)

    # Fan-in: 4 desks → barrier
    for desk in ["macro_analyst", "fundamental_analyst", "sentiment_analyst", "quant_analyst"]:
        g.add_edge(desk, "barrier")

    g.add_edge("barrier", "hedge_lite_builder")
    g.add_edge("hedge_lite_builder", "research_router")

    g.add_conditional_edges("research_router", research_router, {
        "research_executor": "research_executor",
        "risk_manager": "risk_manager",
    })

    for desk in ["macro_analyst_research", "fundamental_analyst_research", "sentiment_analyst_research", "quant_analyst_research"]:
        g.add_edge("research_executor", desk)
        g.add_edge(desk, "research_barrier")

    g.add_edge("research_barrier", "hedge_lite_builder")

    g.add_conditional_edges("risk_manager", risk_router, {
        "orchestrator": "orchestrator",
        "report_writer": "report_writer",
    })

    g.add_edge("report_writer", END)

    return g


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main(
    mode: str = "mock",
    seed: int | None = 42,
    user_request: str | None = None,
) -> dict:
    print("🚀 7-Agent AI Investment Team V3 (Final Integration)")
    print("=" * 60)
    print(f"   Mode: {mode} | Seed: {seed}")

    state = create_initial_state(
        user_request=user_request or "애플(AAPL) 주식을 지금 매수해도 괜찮을까요? 6개월 투자 관점에서 분석해 주세요.",
        mode=mode,
        seed=seed,
    )
    run_id = state["run_id"]
    print(f"   Run ID: {run_id}")

    run_dir = telemetry.init_run(run_id, mode)
    print(f"   Run Dir: {run_dir}")

    graph = build_investment_graph()
    app = graph.compile()
    final_state = app.invoke(state)

    telemetry.save_final_state(run_id, final_state)
    operator_summary_path = _write_operator_summary(run_id)

    print("\n" + "=" * 60)
    print("✅ Pipeline Complete")
    print("=" * 60)

    # Summary
    macro = final_state.get("macro_analysis", {})
    funda = final_state.get("fundamental_analysis", {})
    senti = final_state.get("sentiment_analysis", {})
    quant = final_state.get("technical_analysis", {})
    risk = final_state.get("risk_assessment", {})

    print(f"   Run ID:       {run_id}")
    print(f"   Iterations:   {final_state.get('iteration_count', 0)}")
    print(f"   Macro Regime: {macro.get('macro_regime', 'N/A')}")
    print(f"   Structural:   {funda.get('structural_risk_flag', 'N/A')}")
    print(f"   Sentiment:    {senti.get('sentiment_regime', 'N/A')} (tilt={senti.get('tilt_factor', 'N/A')})")
    print(f"   Quant:        {quant.get('decision', 'N/A')} (alloc={quant.get('final_allocation_pct', 'N/A')})")
    print(f"   Risk Grade:   {risk.get('grade', 'N/A')}")
    print(f"   Events:       runs/{run_id}/events.jsonl")
    print(f"   State:        runs/{run_id}/final_state.json")
    print(f"   Ops Timeline: runs/{run_id}/operator_timeline.log")
    if operator_summary_path is not None:
        print(f"   Ops Summary:  {operator_summary_path}")

    report = final_state.get("final_report", "")
    if report:
        print(f"\n{'─' * 60}")
        print("📄 Final Report (first 30 lines)")
        print(f"{'─' * 60}")
        for line in report.split("\n")[:30]:
            print(line)

    print(f"\n⚠️  Confirm: No Broker API / No order placement code added. (R0)")
    return final_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="7-Agent AI Investment Team V3")
    parser.add_argument("--mode", default="mock", choices=["mock", "live"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question", type=str, default=None, help="사용자 질문 텍스트")
    args = parser.parse_args()
    main(mode=args.mode, seed=args.seed, user_request=args.question)
