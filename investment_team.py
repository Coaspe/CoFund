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
from typing import Any, Literal, Optional
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
from visualization.agent_empire import write_run_dashboard

# ── Data Providers ────────────────────────────────────────────────────────
from data_providers.data_hub import DataHub
from data_providers.web_research_provider import WebResearchProvider
from data_providers.sec_edgar_provider import SECEdgarProvider
from data_providers.tavily_search_provider import TavilySearchProvider
from data_providers.exa_search_provider import ExaSearchProvider
from data_providers.perplexity_search_provider import PerplexitySearchProvider

# ── Agents ────────────────────────────────────────────────────────────────
from agents.macro_agent import macro_analyst_run
from agents.fundamental_agent import fundamental_analyst_run
from agents.sentiment_agent import sentiment_analyst_run
from agents.autonomy_planner import plan_runtime_recovery

# ── Engines (for quant) ──────────────────────────────────────────────────
from engines.quant_engine import generate_quant_payload, mock_quant_decision
from engines.research_policy import compute_evidence_score, should_run_web_research

try:
    from llm.router import get_llm_with_cache
    from langchain_core.messages import HumanMessage, SystemMessage

    HAS_FRONTDOOR_LLM = True
except Exception:
    HAS_FRONTDOOR_LLM = False
    get_llm_with_cache = None  # type: ignore
    HumanMessage = SystemMessage = None  # type: ignore

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
_QUESTION_TICKER_RE = re.compile(r"(?<![A-Z0-9])((?:[A-Z]{1,5}|[0-9]{4,6})(?:[.-][A-Z0-9]{1,6})?)(?![A-Z0-9])")
_QUESTION_TICKER_STOPWORDS = {"AI", "ETF", "PM", "CEO", "CIO", "IPO", "USD", "KRW"}
_QUESTION_KR_TICKER_MAP = {
    "애플": "AAPL",
    "마이크로소프트": "MSFT",
    "엔비디아": "NVDA",
    "테슬라": "TSLA",
    "구글": "GOOGL",
    "알파벳": "GOOGL",
    "아마존": "AMZN",
    "메타": "META",
    "나스닥": "QQQ",
    "미국증시": "SPY",
    "미국 시장": "SPY",
}


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


def _normalize_primary_tickers(values: Any) -> list[str]:
    out: list[str] = []
    seen = set()
    for raw in values or []:
        ticker = str(raw or "").strip().upper()
        if not ticker or ticker in _QUESTION_TICKER_STOPWORDS or ticker in seen:
            continue
        out.append(ticker)
        seen.add(ticker)
    return out


_POSITION_REVIEW_EXPLICIT_HINTS = (
    "평단", "보유", "shares", "주 ", "주,", "avg cost", "매입가", "리밸런", "비중",
)
_POSITION_REVIEW_SELL_HINTS = (
    "매도", "익절", "차익실현", "정리", "팔까", "팔아", "sell", "trim", "take profit",
)
_POSITION_REVIEW_PROFIT_HINTS = (
    "수익", "수익률", "profit", "gain", "gains",
)
_POSITION_REVIEW_TIMING_HINTS = (
    "언제", "when", "지금", "now", "timing",
)
_CANONICAL_FRONTDOOR_INTENTS = {
    "single_name",
    "position_review",
    "portfolio_rebalance",
    "hedge_design",
    "market_outlook",
    "relative_value",
}
_LEGACY_PLANNER_INTENT_TO_CANONICAL = {
    "single_ticker_entry": "single_name",
    "compare_tickers": "relative_value",
}
_PLANNER_INTENT_SCENARIO_TAGS = {
    "event_risk": "event_risk",
    "overheated_check": "overheated_check",
}


def _looks_like_position_review_request(user_request: str, universe: list[str]) -> bool:
    text = _normalize_query_text(user_request)
    if any(k in text for k in _POSITION_REVIEW_EXPLICIT_HINTS):
        return True
    symbols = [str(t).strip().upper() for t in (universe or []) if str(t).strip()]
    if len(symbols) != 1:
        return False
    if any(k in text for k in _POSITION_REVIEW_SELL_HINTS):
        return True
    has_profit = any(k in text for k in _POSITION_REVIEW_PROFIT_HINTS)
    has_timing = any(k in text for k in _POSITION_REVIEW_TIMING_HINTS)
    return has_profit and has_timing


def _detect_output_language(user_request: str) -> str:
    text = str(user_request or "")
    if re.search(r"[가-힣]", text):
        return "ko"
    return "en"


def _infer_intent_from_request(user_request: str, universe: list[str]) -> str:
    text = _normalize_query_text(user_request)
    if _looks_like_position_review_request(user_request, universe):
        if len(universe or []) >= 2:
            return "portfolio_rebalance"
        if len(universe or []) == 1:
            return "position_review"
    if any(k in text for k in ("시장", "macro", "market outlook", "전망", "지정학", "event risk", "이벤트")):
        return "market_outlook"
    if any(k in text for k in ("헤지", "hedge")):
        return "hedge_design"
    if len(universe or []) >= 2:
        return "relative_value"
    return "single_name"


def _normalize_scenario_tags(values: Any) -> list[str]:
    out: list[str] = []
    seen = set()
    for raw in values or []:
        tag = str(raw or "").strip().lower()
        if not tag or tag in seen:
            continue
        out.append(tag)
        seen.add(tag)
    return out


def _canonicalize_runtime_intent(
    raw_intent: Any,
    *,
    preferred_intent: Any = "",
    user_request: str = "",
    universe: Optional[list[str]] = None,
) -> tuple[str, list[str]]:
    raw = str(raw_intent or "").strip().lower()
    preferred = str(preferred_intent or "").strip().lower()
    scenario_tag = _PLANNER_INTENT_SCENARIO_TAGS.get(raw)
    scenario_tags = [scenario_tag] if scenario_tag else []

    if preferred in _CANONICAL_FRONTDOOR_INTENTS:
        canonical = preferred
    elif raw in _CANONICAL_FRONTDOOR_INTENTS:
        canonical = raw
    elif raw in _LEGACY_PLANNER_INTENT_TO_CANONICAL:
        canonical = _LEGACY_PLANNER_INTENT_TO_CANONICAL[raw]
    else:
        canonical = ""

    if canonical not in _CANONICAL_FRONTDOOR_INTENTS:
        canonical = _infer_intent_from_request(
            user_request,
            [str(t).strip().upper() for t in (universe or []) if str(t).strip()],
        ) if user_request else "single_name"
    if canonical not in _CANONICAL_FRONTDOOR_INTENTS:
        canonical = "single_name"
    return canonical, scenario_tags


def _extract_question_tickers(user_request: str) -> list[str]:
    text = str(user_request or "")
    out: list[str] = []
    seen = set()
    for match in _QUESTION_TICKER_RE.findall(text):
        t = str(match or "").strip().upper()
        if t in _QUESTION_TICKER_STOPWORDS:
            continue
        if t not in seen:
            out.append(t)
            seen.add(t)
    if out:
        return out
    for kr, ticker in _QUESTION_KR_TICKER_MAP.items():
        if kr in text and ticker not in seen:
            out.append(ticker)
            seen.add(ticker)
    return out


def _parse_numeric_phrase(value: Any) -> Optional[float]:
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    multiplier = 1.0
    if "억" in raw:
        multiplier *= 100_000_000.0
    if "만" in raw:
        multiplier *= 10_000.0
    if raw.endswith("k"):
        multiplier *= 1_000.0
    elif raw.endswith("m"):
        multiplier *= 1_000_000.0
    elif raw.endswith("b"):
        multiplier *= 1_000_000_000.0
    cleaned = re.sub(r"[^0-9.+-]", "", raw)
    if not cleaned:
        return None
    try:
        return float(cleaned) * multiplier
    except ValueError:
        return None


def _normalize_currency(text: Any) -> str:
    raw = str(text or "").strip().upper()
    if not raw:
        return "USD"
    if "₩" in raw or "KRW" in raw or "원" in raw:
        return "KRW"
    return "USD"


def _normalize_holdings(holdings_raw: Any) -> list[dict]:
    out: list[dict] = []
    for item in holdings_raw or []:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker", "")).strip().upper()
        shares = _parse_numeric_phrase(item.get("shares"))
        if not ticker or shares is None or shares <= 0:
            continue
        avg_cost = _parse_numeric_phrase(item.get("avg_cost"))
        out.append({
            "ticker": ticker,
            "shares": round(float(shares), 6),
            "avg_cost": round(float(avg_cost), 6) if avg_cost is not None else None,
            "currency": _normalize_currency(item.get("currency")),
        })
    return out


def _portfolio_context_intake_seed(portfolio_context: Any) -> dict:
    ctx = portfolio_context if isinstance(portfolio_context, dict) else {}
    holdings = _normalize_holdings(ctx.get("holdings") or [])
    primary_tickers = _normalize_primary_tickers(
        ctx.get("primary_tickers")
        or [item.get("ticker") for item in holdings]
    )
    cash = ctx.get("cash") if isinstance(ctx.get("cash"), dict) else None
    account_value = ctx.get("account_value") if isinstance(ctx.get("account_value"), dict) else None
    return {
        "holdings": holdings,
        "primary_tickers": primary_tickers,
        "cash": cash,
        "account_value": account_value,
    }


def _question_understanding_rules(user_request: str) -> dict:
    text = str(user_request or "")
    lower = text.lower()
    tickers = _extract_question_tickers(text)
    holdings: list[dict] = []
    holding_pat = re.compile(
        r"\b(?P<ticker>[A-Z]{1,5})\b[\s:,-]*"
        r"(?P<shares>[0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:주|shares?)"
        r"(?:[\s,/-]*(?:평단|avg(?:\s*cost)?|cost|매입가|at)\s*"
        r"(?P<avg>[$₩]?[0-9][0-9,]*(?:\.[0-9]+)?(?:만|억|k|m|b)?))?",
        re.IGNORECASE,
    )
    seen = set()
    for match in holding_pat.finditer(text):
        ticker = str(match.group("ticker") or "").strip().upper()
        shares = _parse_numeric_phrase(match.group("shares"))
        avg_cost = _parse_numeric_phrase(match.group("avg"))
        if not ticker or shares is None or shares <= 0 or ticker in seen:
            continue
        holdings.append({
            "ticker": ticker,
            "shares": round(float(shares), 6),
            "avg_cost": round(float(avg_cost), 6) if avg_cost is not None else None,
            "currency": "USD",
        })
        seen.add(ticker)
    cash_match = re.search(
        r"(?:현금|cash)\s*(?:은|는|이|가)?\s*(?P<amount>[$₩]?[0-9][0-9,]*(?:\.[0-9]+)?(?:만|억|k|m|b)?)",
        text,
        re.IGNORECASE,
    )
    account_match = re.search(
        r"(?:총\s*자산|계좌\s*규모|계좌\s*가치|portfolio\s*value|account\s*value)\s*(?:은|는|이|가)?\s*(?P<amount>[$₩]?[0-9][0-9,]*(?:\.[0-9]+)?(?:만|억|k|m|b)?)",
        text,
        re.IGNORECASE,
    )
    if holdings:
        tickers = [item["ticker"] for item in holdings]
    looks_like_position_review = _looks_like_position_review_request(text, tickers)
    question_type = "single_name_analysis"
    if (
        any(k in lower for k in ("리밸런", "rebalance", "비중", "allocation", "포트폴리오"))
        or len(holdings) >= 2
        or (looks_like_position_review and len(tickers) >= 2)
    ):
        question_type = "portfolio_rebalance"
    elif holdings or looks_like_position_review:
        question_type = "single_position_review"
    elif any(k in lower for k in ("헤지", "hedge")):
        question_type = "hedge_request"
    elif any(k in lower for k in ("시장", "전망", "macro", "outlook")) and not tickers:
        question_type = "market_outlook"
    elif len(tickers) >= 2:
        question_type = "compare_tickers"

    intent_map = {
        "portfolio_rebalance": "portfolio_rebalance",
        "single_position_review": "position_review",
        "hedge_request": "hedge_design",
        "market_outlook": "market_outlook",
        "compare_tickers": "relative_value",
        "single_name_analysis": "single_name",
    }
    missing_fields: list[str] = []
    if holdings and any(item.get("avg_cost") is None for item in holdings):
        missing_fields.append("avg_cost")
    if question_type == "portfolio_rebalance" and not cash_match and not account_match:
        missing_fields.append("account_value_or_cash")
    return {
        "question_type": question_type,
        "intent": intent_map.get(question_type, "single_name"),
        "primary_tickers": tickers[:8],
        "holdings": holdings,
        "cash": {
            "amount": _parse_numeric_phrase(cash_match.group("amount")),
            "currency": _normalize_currency(cash_match.group("amount")),
        } if cash_match else None,
        "account_value": {
            "amount": _parse_numeric_phrase(account_match.group("amount")),
            "currency": _normalize_currency(account_match.group("amount")),
        } if account_match else None,
        "constraints": {
            "horizon_days": 30,
            "risk_tolerance": None,
        },
        "user_goal": "review_or_rebalance" if question_type in {"single_position_review", "portfolio_rebalance"} else "analyze",
        "missing_fields": missing_fields,
        "assumption_policy": "limited_answer_without_fabrication",
        "confidence": (
            0.92 if holdings
            else (0.84 if question_type == "single_position_review" else (0.80 if question_type == "portfolio_rebalance" else (0.78 if tickers else 0.55)))
        ),
        "source": "rules",
    }


def _parse_json_dict_maybe_fenced(text: str) -> dict:
    s = str(text or "").strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, TypeError):
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(s[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise json.JSONDecodeError("No JSON object found", s, 0)


def _validate_question_understanding(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    if not isinstance(result.get("question_type"), str):
        return False
    if not isinstance(result.get("intent"), str):
        return False
    if result.get("holdings") is not None and not isinstance(result.get("holdings"), list):
        return False
    return True


_QUESTION_UNDERSTANDING_SYSTEM_PROMPT = """
You are a frontdoor parser for an investment research system.
Return JSON only.
Do not guess missing facts.
If the user did not provide a value, use null.
Infer the user's underlying investment task semantically, not with keyword matching.
Resolve company/common-language names to canonical tradable tickers when unambiguous.
If ticker resolution is ambiguous, leave primary_tickers empty and add "ticker_confirmation" to missing_fields.
If the user is asking about when to sell, trim, or protect gains on an existing position, classify it as single_position_review.
Extract the user's question type, tickers, holdings, cash, account value, and constraints.
Valid question_type values:
- single_name_analysis
- single_position_review
- portfolio_rebalance
- hedge_request
- market_outlook
- compare_tickers
Valid intent values:
- single_name
- position_review
- portfolio_rebalance
- hedge_design
- market_outlook
- relative_value
Output keys:
question_type, intent, primary_tickers, holdings, cash, account_value, constraints,
user_goal, missing_fields, assumption_policy, confidence.
Each holding must be: {ticker, shares, avg_cost, currency}.
""".strip()


def _apply_question_understanding_guardrails(
    llm_out: dict,
    rules: dict,
) -> dict:
    merged = dict(llm_out or {})
    merged["primary_tickers"] = _normalize_primary_tickers(
        merged.get("primary_tickers") or rules.get("primary_tickers") or []
    )
    merged["holdings"] = _normalize_holdings(merged.get("holdings") or rules.get("holdings") or [])
    merged["cash"] = merged.get("cash") if isinstance(merged.get("cash"), dict) else rules.get("cash")
    merged["account_value"] = (
        merged.get("account_value")
        if isinstance(merged.get("account_value"), dict)
        else rules.get("account_value")
    )
    merged["constraints"] = (
        merged.get("constraints")
        if isinstance(merged.get("constraints"), dict)
        else rules.get("constraints")
    )
    merged["missing_fields"] = list(dict.fromkeys([
        str(x).strip()
        for x in [*(merged.get("missing_fields") or []), *(rules.get("missing_fields") or [])]
        if str(x).strip()
    ]))
    merged["assumption_policy"] = str(
        merged.get("assumption_policy") or rules.get("assumption_policy") or "limited_answer_without_fabrication"
    ).strip()
    try:
        merged["confidence"] = max(0.0, min(1.0, float(merged.get("confidence", rules.get("confidence", 0.0)) or 0.0)))
    except (TypeError, ValueError):
        merged["confidence"] = float(rules.get("confidence", 0.0) or 0.0)

    question_type = str(merged.get("question_type", "")).strip()
    intent = str(merged.get("intent", "")).strip()
    user_goal = str(merged.get("user_goal", "")).strip()
    low_confidence = float(merged.get("confidence", 0.0) or 0.0) < 0.70

    if merged["holdings"]:
        if len(merged["holdings"]) >= 2:
            question_type = "portfolio_rebalance"
            intent = "portfolio_rebalance"
        else:
            question_type = "single_position_review"
            intent = "position_review"
        user_goal = "review_or_rebalance"
    elif low_confidence:
        rules_question_type = str(rules.get("question_type", "")).strip()
        if rules_question_type in {"single_position_review", "portfolio_rebalance"}:
            question_type = rules_question_type
            intent = str(rules.get("intent", "")).strip()
            user_goal = str(rules.get("user_goal", "")).strip()

    if not question_type:
        question_type = str(rules.get("question_type", "")).strip()
    if not intent:
        intent = str(rules.get("intent", "")).strip()
    if not user_goal:
        user_goal = str(rules.get("user_goal", "")).strip()

    merged["question_type"] = question_type
    merged["intent"] = intent
    merged["user_goal"] = user_goal
    if not merged["primary_tickers"]:
        merged["primary_tickers"] = _normalize_primary_tickers(rules.get("primary_tickers") or [])
    if str(merged.get("source", "")).strip() != "rules":
        merged["source"] = "llm"
    return merged


def _merge_understanding_with_intake_seed(understanding: dict, intake_seed: dict) -> dict:
    merged = dict(understanding or {})
    holdings = intake_seed.get("holdings") or []
    primary_tickers = _normalize_primary_tickers(
        intake_seed.get("primary_tickers")
        or merged.get("primary_tickers")
        or [item.get("ticker") for item in holdings]
    )
    if holdings:
        merged["holdings"] = holdings
        primary_tickers = [item["ticker"] for item in holdings]
        if len(holdings) >= 2:
            merged["question_type"] = "portfolio_rebalance"
            merged["intent"] = "portfolio_rebalance"
        else:
            merged["question_type"] = "single_position_review"
            merged["intent"] = "position_review"
        merged["user_goal"] = "review_or_rebalance"
    if primary_tickers:
        merged["primary_tickers"] = primary_tickers
    if intake_seed.get("cash") is not None:
        merged["cash"] = intake_seed.get("cash")
    if intake_seed.get("account_value") is not None:
        merged["account_value"] = intake_seed.get("account_value")
    missing_fields = [str(x).strip() for x in (merged.get("missing_fields") or []) if str(x).strip()]
    if holdings and all(item.get("avg_cost") is not None for item in holdings):
        missing_fields = [item for item in missing_fields if item != "avg_cost"]
    if primary_tickers:
        missing_fields = [item for item in missing_fields if item != "ticker_confirmation"]
    merged["missing_fields"] = list(dict.fromkeys(missing_fields))
    return merged


def _call_question_understanding_llm(user_request: str) -> dict:
    rules = _question_understanding_rules(user_request)
    if not HAS_FRONTDOOR_LLM or get_llm_with_cache is None or SystemMessage is None or HumanMessage is None:
        return rules
    human_msg = (
        "Parse the following investment question into structured JSON.\n"
        "Do not infer missing portfolio facts.\n\n"
        f"User question:\n{user_request}"
    )
    try:
        llm, cached = get_llm_with_cache("orchestrator", human_msg)
        if isinstance(cached, dict) and _validate_question_understanding(cached):
            out = dict(cached)
        else:
            if llm is None:
                return rules
            raw = llm.invoke([
                SystemMessage(content=_QUESTION_UNDERSTANDING_SYSTEM_PROMPT),
                HumanMessage(content=human_msg),
            ])
            out = _parse_json_dict_maybe_fenced(getattr(raw, "content", ""))
        if not _validate_question_understanding(out):
            return rules
        return _apply_question_understanding_guardrails(out, rules)
    except Exception:
        return rules


def _merge_portfolio_context(existing: dict, understanding: dict, intake: dict, normalized: dict) -> dict:
    ctx = dict(existing or {})
    ctx.setdefault("question_type", understanding.get("question_type"))
    ctx.setdefault("frontdoor_intent", understanding.get("intent"))
    ctx.setdefault("primary_tickers", understanding.get("primary_tickers", []))
    ctx.setdefault("user_goal", understanding.get("user_goal"))
    ctx.setdefault("frontdoor_confidence", understanding.get("confidence"))
    ctx.setdefault("frontdoor_source", understanding.get("source"))
    if intake.get("holdings"):
        ctx.setdefault("holdings", intake.get("holdings"))
    if intake.get("cash") is not None:
        ctx.setdefault("cash", intake.get("cash"))
    if intake.get("account_value") is not None:
        ctx.setdefault("account_value", intake.get("account_value"))
    if normalized:
        ctx["normalized_portfolio_snapshot"] = normalized
    missing = understanding.get("missing_fields", [])
    if missing:
        ctx["missing_fields"] = list(dict.fromkeys([*ctx.get("missing_fields", []), *missing]))
    return ctx


def _resolve_frontdoor_targets(
    state: InvestmentState,
    understanding: dict,
    intake: dict,
) -> tuple[str, list[str]]:
    target = str(state.get("target_ticker", "")).strip().upper()
    primary_tickers = _normalize_primary_tickers(understanding.get("primary_tickers") or [])
    universe = [str(t).strip().upper() for t in (state.get("universe") or []) if str(t).strip()]
    if not universe:
        universe = primary_tickers[:]
    if not universe:
        universe = [str(item.get("ticker", "")).strip().upper() for item in intake.get("holdings", []) if str(item.get("ticker", "")).strip()]
    if not target and primary_tickers:
        target = primary_tickers[0]
    if not target and universe:
        target = universe[0]
    return target, universe


def _build_frontdoor_bundle(
    state: InvestmentState,
    *,
    compute_normalized: bool = True,
) -> dict[str, Any]:
    if state.get("_frontdoor_prepared") and isinstance(state.get("question_understanding"), dict) and state.get("question_understanding"):
        understanding = dict(state.get("question_understanding", {}) or {})
        intake = dict(state.get("portfolio_intake", {}) or {})
        normalized = dict(state.get("normalized_portfolio_snapshot", {}) or {})
        if compute_normalized and not normalized:
            normalized = _normalize_portfolio_snapshot(state, intake)
        merged_ctx = _merge_portfolio_context(
            state.get("portfolio_context", {}) if isinstance(state.get("portfolio_context"), dict) else {},
            understanding,
            intake,
            normalized,
        )
        target, universe = _resolve_frontdoor_targets(state, understanding, intake)
    else:
        user_request = str(state.get("user_request", "") or "")
        understanding = _call_question_understanding_llm(user_request)
        intake_seed = _portfolio_context_intake_seed(state.get("portfolio_context", {}))
        understanding = _merge_understanding_with_intake_seed(understanding, intake_seed)
        intake = {
            "holdings": _normalize_holdings(understanding.get("holdings") or []),
            "cash": understanding.get("cash"),
            "account_value": understanding.get("account_value"),
            "missing_fields": list(understanding.get("missing_fields", []) or []),
        }
        normalized = _normalize_portfolio_snapshot(state, intake) if compute_normalized else {}
        existing_ctx = state.get("portfolio_context", {}) if isinstance(state.get("portfolio_context"), dict) else {}
        merged_ctx = _merge_portfolio_context(existing_ctx, understanding, intake, normalized)
        target, universe = _resolve_frontdoor_targets(state, understanding, intake)

    out: dict[str, Any] = {
        "question_understanding": understanding,
        "portfolio_intake": intake,
        "normalized_portfolio_snapshot": normalized,
        "portfolio_context": merged_ctx,
        "_frontdoor_prepared": True,
    }
    if universe and not state.get("universe"):
        out["universe"] = universe
    if target and not state.get("target_ticker"):
        out["target_ticker"] = target
    if understanding.get("intent") and not state.get("intent"):
        out["intent"] = str(understanding.get("intent"))
    if not (state.get("positions_final") or state.get("positions_proposed")) and normalized.get("weights"):
        out["positions_final"] = dict(normalized.get("weights", {}))
    return out


def _build_position_review_clarification(understanding: dict, intake: dict) -> dict:
    intent = str(understanding.get("intent", "")).strip()
    if intent != "position_review":
        return {"required": False, "fields": [], "message": "", "target_ticker": "", "currency": "USD"}
    holdings = _normalize_holdings(intake.get("holdings") or [])
    primary_tickers = _normalize_primary_tickers(understanding.get("primary_tickers") or [])
    target_ticker = primary_tickers[0] if primary_tickers else (holdings[0]["ticker"] if holdings else "")
    fields: list[str] = []
    if not target_ticker:
        fields.append("ticker")
    if not holdings:
        fields.extend(["shares", "avg_cost"])
    else:
        if holdings[0].get("avg_cost") is None:
            fields.append("avg_cost")
    message = ""
    if fields:
        message = "position review에서 매도 시점과 규모를 정밀하게 계산하려면 보유 수량과 평단이 필요합니다."
        if "ticker" in fields:
            message = "position review를 정확히 진행하려면 검토할 종목과 보유 수량, 평단이 필요합니다."
    return {
        "required": bool(fields),
        "fields": fields,
        "message": message,
        "target_ticker": target_ticker,
        "currency": "USD",
    }


def preview_launch_requirements(
    user_request: str,
    *,
    portfolio_context: dict | None = None,
    mode: str = "mock",
    seed: int | None = 42,
) -> dict[str, Any]:
    state = create_initial_state(
        user_request=user_request,
        mode=mode,
        portfolio_context=portfolio_context or {},
        seed=seed,
    )
    bundle = _build_frontdoor_bundle(state, compute_normalized=False)
    understanding = dict(bundle.get("question_understanding", {}) or {})
    intake = dict(bundle.get("portfolio_intake", {}) or {})
    clarification = _build_position_review_clarification(understanding, intake)
    return {
        "question_understanding": understanding,
        "portfolio_intake": intake,
        "target_ticker": str(bundle.get("target_ticker", "")),
        "universe": list(bundle.get("universe", []) or []),
        "needs_clarification": bool(clarification.get("required")),
        "clarification": clarification,
    }


def _prompt_position_review_inputs_if_tty(state: InvestmentState, preview: dict[str, Any]) -> dict[str, Any]:
    clarification = preview.get("clarification", {}) if isinstance(preview.get("clarification"), dict) else {}
    if not clarification.get("required"):
        return preview
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return preview

    fields = list(clarification.get("fields", []) or [])
    target_ticker = str(clarification.get("target_ticker", "") or "").strip().upper()
    currency = str(clarification.get("currency", "USD") or "USD").strip().upper() or "USD"

    print("\n📝 Position Review Input Required")
    print("   매도 시점과 규모를 정밀하게 계산하려면 보유 종목 정보를 입력해 주세요.")

    if "ticker" in fields:
        raw = input(f"   Ticker [{target_ticker or '예: NVDA'}]: ").strip().upper()
        if raw:
            target_ticker = raw
    if "shares" in fields:
        raw = input("   Shares [예: 25]: ").strip()
        shares = _parse_numeric_phrase(raw)
    else:
        shares = None
    if "avg_cost" in fields:
        raw = input(f"   Avg Cost ({currency}) [예: 132.5]: ").strip()
        avg_cost = _parse_numeric_phrase(raw)
    else:
        avg_cost = None
    raw_currency = input(f"   Currency [{currency}]: ").strip().upper()
    if raw_currency:
        currency = raw_currency

    if target_ticker and shares is not None and shares > 0:
        portfolio_context = dict(state.get("portfolio_context", {}) if isinstance(state.get("portfolio_context"), dict) else {})
        portfolio_context["holdings"] = [{
            "ticker": target_ticker,
            "shares": float(shares),
            "avg_cost": float(avg_cost) if avg_cost is not None else None,
            "currency": currency,
        }]
        portfolio_context["primary_tickers"] = [target_ticker]
        state["portfolio_context"] = portfolio_context
        return preview_launch_requirements(
            str(state.get("user_request", "") or ""),
            portfolio_context=portfolio_context,
            mode=str(state.get("mode", "mock") or "mock"),
            seed=state.get("run_context", {}).get("seed"),
        )
    return preview


def _has_position_review_snapshot(state: InvestmentState) -> bool:
    frontdoor = state.get("question_understanding", {}) if isinstance(state.get("question_understanding"), dict) else {}
    if str(frontdoor.get("intent", "")).strip() != "position_review":
        return True
    normalized = state.get("normalized_portfolio_snapshot", {}) if isinstance(state.get("normalized_portfolio_snapshot"), dict) else {}
    weights = normalized.get("weights", {}) if isinstance(normalized.get("weights"), dict) else {}
    if weights:
        return True
    intake = state.get("portfolio_intake", {}) if isinstance(state.get("portfolio_intake"), dict) else {}
    if _normalize_holdings(intake.get("holdings") or []):
        return True
    positions_final = state.get("positions_final", {}) if isinstance(state.get("positions_final"), dict) else {}
    return bool(positions_final) and not bool(state.get("positions_proposed"))


def _frontdoor_intent(state: InvestmentState) -> str:
    frontdoor = state.get("question_understanding", {}) if isinstance(state.get("question_understanding"), dict) else {}
    frontdoor_intent = str(frontdoor.get("intent", "")).strip().lower()
    if frontdoor_intent in _CANONICAL_FRONTDOOR_INTENTS:
        return frontdoor_intent
    return _state_canonical_intent(state)


def _state_canonical_intent(state: InvestmentState) -> str:
    frontdoor = state.get("question_understanding", {}) if isinstance(state.get("question_understanding"), dict) else {}
    frontdoor_intent = str(frontdoor.get("intent", "")).strip()
    frontdoor_tickers = _normalize_primary_tickers(frontdoor.get("primary_tickers") or [])
    universe = [str(t).strip().upper() for t in (state.get("universe") or frontdoor_tickers) if str(t).strip()]
    canonical, _ = _canonicalize_runtime_intent(
        state.get("intent", ""),
        preferred_intent=frontdoor_intent,
        user_request=str(state.get("user_request", "") or ""),
        universe=universe,
    )
    return canonical


def _state_scenario_tags(state: InvestmentState) -> list[str]:
    tags = _normalize_scenario_tags(state.get("scenario_tags") or [])
    legacy_tag = _PLANNER_INTENT_SCENARIO_TAGS.get(str(state.get("intent", "")).strip().lower())
    if legacy_tag and legacy_tag not in tags:
        tags.append(legacy_tag)
    return tags


def _state_has_scenario_tag(state: InvestmentState, tag: str) -> bool:
    return str(tag or "").strip().lower() in set(_state_scenario_tags(state))


def _normalize_portfolio_snapshot(state: InvestmentState, intake: dict) -> dict:
    holdings = _normalize_holdings(intake.get("holdings") or [])
    if not holdings:
        return {"status": "no_holdings", "holdings": [], "weights": {}, "basis": "none", "missing_fields": []}
    seed = _make_seed(state) + 401
    as_of = state.get("as_of", "")
    mode = state.get("mode", "mock")
    hub = DataHub(run_id=state.get("run_id", ""), as_of=as_of, mode=mode)
    cash_payload = intake.get("cash") or {}
    account_value_payload = intake.get("account_value") or {}
    cash_amount = _parse_numeric_phrase(cash_payload.get("amount"))
    account_value = _parse_numeric_phrase(account_value_payload.get("amount"))
    enriched: list[dict] = []
    unresolved: list[str] = []
    price_evidence: list[dict] = []
    total_holdings_value = 0.0
    for idx, item in enumerate(holdings):
        ticker = item["ticker"]
        try:
            prices, ev, _ = hub.get_price_series(ticker, lookback_days=90, seed=seed + idx * 17)
            px = float(prices[-1]) if len(prices) else None
            px_as_of = str((ev[-1].get("as_of") if ev else "") or as_of)
            price_evidence.extend(ev or [])
        except Exception:
            px = None
            px_as_of = as_of
        if px is None or px <= 0:
            unresolved.append(ticker)
            enriched.append({
                **item,
                "current_price": None,
                "price_as_of": px_as_of,
                "market_value": None,
                "unrealized_pnl": None,
                "unrealized_pnl_pct": None,
            })
            continue
        market_value = float(item["shares"]) * px
        total_holdings_value += market_value
        avg_cost = item.get("avg_cost")
        pnl = None
        pnl_pct = None
        if avg_cost is not None and avg_cost > 0:
            pnl = (px - float(avg_cost)) * float(item["shares"])
            pnl_pct = ((px / float(avg_cost)) - 1.0) * 100.0
        enriched.append({
            **item,
            "current_price": round(px, 6),
            "price_as_of": px_as_of,
            "market_value": round(market_value, 6),
            "unrealized_pnl": round(pnl, 6) if pnl is not None else None,
            "unrealized_pnl_pct": round(pnl_pct, 4) if pnl_pct is not None else None,
        })
    denominator = None
    basis = "known_assets_only"
    if account_value is not None and account_value > 0:
        denominator = float(account_value)
        basis = "account_value"
    elif total_holdings_value > 0:
        denominator = total_holdings_value + max(0.0, float(cash_amount or 0.0))
        basis = "holdings_plus_cash" if cash_amount is not None else "known_assets_only"
    weights: dict[str, float] = {}
    if denominator and denominator > 0:
        for item in enriched:
            mv = item.get("market_value")
            if mv is None:
                continue
            weights[item["ticker"]] = float(mv) / denominator
        weights = _normalize_long_only_weights(weights)
    return {
        "status": "ok" if weights else ("insufficient_price_data" if unresolved else "no_weights"),
        "holdings": enriched,
        "weights": weights,
        "basis": basis,
        "known_holdings_value": round(total_holdings_value, 6),
        "cash_amount": round(float(cash_amount), 6) if cash_amount is not None else None,
        "account_value": round(float(account_value), 6) if account_value is not None else None,
        "price_unresolved": unresolved,
        "missing_fields": list(intake.get("missing_fields", []) or []),
        "evidence_count": len(price_evidence),
    }


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
# ① Frontdoor + Orchestrator Node
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def question_understanding_node(state: InvestmentState) -> dict:
    """Frontdoor node: diverse user questions -> structured intake + normalized portfolio snapshot."""
    _log(state, "question_understanding", "enter")
    out = _build_frontdoor_bundle(state, compute_normalized=True)
    understanding = out.get("question_understanding", {}) if isinstance(out.get("question_understanding"), dict) else {}
    intake = out.get("portfolio_intake", {}) if isinstance(out.get("portfolio_intake"), dict) else {}
    normalized = out.get("normalized_portfolio_snapshot", {}) if isinstance(out.get("normalized_portfolio_snapshot"), dict) else {}
    primary_tickers = _normalize_primary_tickers(understanding.get("primary_tickers") or [])

    _ops_log(
        state,
        "question_understanding",
        "frontdoor parse",
        [
            f"type={understanding.get('question_type', 'unknown')} intent={understanding.get('intent', 'unknown')}",
            f"tickers={_list_preview(primary_tickers, max_items=6, max_len=120) or 'n/a'} holdings={len(intake.get('holdings', []))}",
            f"normalization={normalized.get('status', 'unknown')} basis={normalized.get('basis', 'n/a')}",
            f"missing={_list_preview(understanding.get('missing_fields', []), max_items=6, max_len=120) or 'none'}",
        ],
    )
    _log(
        state,
        "question_understanding",
        "exit",
        {
            "question_type": understanding.get("question_type"),
            "holdings": len(intake.get("holdings", [])),
        },
    )
    return out

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
        result["orchestrator_directives"] = directives
    frontdoor = state.get("question_understanding", {}) if isinstance(state.get("question_understanding"), dict) else {}
    frontdoor_tickers = [str(t).strip().upper() for t in (frontdoor.get("primary_tickers") or []) if str(t).strip()]
    frontdoor_intent = str(frontdoor.get("intent", "")).strip() or str(state.get("intent", "")).strip()
    universe = _extract_universe_from_directives(directives)
    if not universe and state.get("universe"):
        universe = [str(t).strip().upper() for t in state.get("universe", []) if str(t).strip()]
    if not universe and frontdoor_tickers:
        universe = frontdoor_tickers[:]
    target_ticker = str(result.get("target_ticker", "")).strip().upper()
    if not target_ticker:
        target_ticker = str(state.get("target_ticker", "")).strip().upper()
    if not target_ticker and frontdoor_tickers:
        target_ticker = frontdoor_tickers[0]
    if not universe and target_ticker:
        universe = [target_ticker]

    output_language = str(state.get("output_language", "")).strip().lower() or _detect_output_language(user_request)
    brief = directives.get("investment_brief", {}) if isinstance(directives.get("investment_brief"), dict) else {}
    directive_universe = universe[:]
    action_type = str(directives.get("action_type", "")).strip()
    if frontdoor_intent == "position_review" and frontdoor_tickers:
        normalized_universe = frontdoor_tickers[:]
        target_ticker = frontdoor_tickers[0]
        needs_directive_update = (
            normalized_universe != directive_universe
            or action_type == "pivot_strategy"
            or str(directives.get("intent", "")).strip() not in {"", "position_review"}
        )
        if needs_directive_update:
            directives = dict(directives)
            brief = dict(brief)
            note = "포지션 리뷰 요청이므로 원보유 종목의 유지/축소/청산 판단에 집중."
            rationale = str(brief.get("rationale", "")).strip()
            if note not in rationale:
                brief["rationale"] = f"{rationale} {note}".strip() if rationale else note
            brief["target_universe"] = normalized_universe
            directives["investment_brief"] = brief
            directives["intent"] = "position_review"
            if action_type == "pivot_strategy":
                directives["action_type"] = "scale_down"
            result["orchestrator_directives"] = directives
        universe = normalized_universe
    elif universe:
        target_ticker = universe[0]

    raw_directive_intent = str(directives.get("intent", "")).strip()
    raw_directive_scenario_tags = _normalize_scenario_tags(directives.get("scenario_tags") or [])
    preferred_intent = frontdoor_intent or _infer_intent_from_request(user_request, universe)
    intent, derived_scenario_tags = _canonicalize_runtime_intent(
        raw_directive_intent or preferred_intent,
        preferred_intent=preferred_intent,
        user_request=user_request,
        universe=universe,
    )
    if frontdoor_intent in {"position_review", "portfolio_rebalance"} and raw_directive_intent in {"", "single_name", "single_ticker_entry"}:
        intent = frontdoor_intent
    scenario_tags = _normalize_scenario_tags([
        *_state_scenario_tags(state),
        *raw_directive_scenario_tags,
        *derived_scenario_tags,
    ])
    directives = dict(directives)
    directives["intent"] = intent
    directives["scenario_tags"] = scenario_tags
    result["orchestrator_directives"] = directives
    execution_mode = "single_main" if frontdoor_intent == "position_review" else ("B_main_plus_hedge_lite" if len(universe) >= 2 else "single_main")
    asset_type_by_ticker = _build_asset_type_map(universe)

    result["target_ticker"] = target_ticker
    result["universe"] = universe
    result["asset_type_by_ticker"] = asset_type_by_ticker
    result["intent"] = intent
    result["scenario_tags"] = scenario_tags
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
        "scenario_tags": scenario_tags,
        "execution_mode": execution_mode,
        "universe_size": len(universe),
        "portfolio_mandate_applied": bool(((directives.get("portfolio_mandate") or {}) if isinstance(directives, dict) else {}).get("applied")),
        "active_idea_count": len(result.get("active_ideas", {}) or {}),
        "open_review_count": len(result.get("monitoring_backlog", []) or []),
        "capital_competition_count": len(result.get("capital_competition", []) or []),
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
            f"mandate_applied={trace_entry.get('portfolio_mandate_applied')}",
            f"active_ideas={trace_entry.get('active_idea_count')} open_reviews={trace_entry.get('open_review_count')}",
            f"capital_candidates={trace_entry.get('capital_competition_count')}",
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
    macro_events, macro_event_evidence, event_meta = hub.get_macro_event_calendar()
    output = macro_analyst_run(
        ticker, indicators,
        run_id=state.get("run_id", ""), as_of=as_of,
        horizon_days=horizon_days,
        focus_areas=focus_areas,
        state=state,
        macro_events=macro_events,
        source_name="mock" if mode == "mock" else "FRED",
    )
    rerun_reason = _rerun_reason_for_desk(state, "macro")
    prev_macro = state.get("macro_analysis", {}) if isinstance(state.get("macro_analysis"), dict) else {}
    output["limitations"] = _merge_limitations(
        output.get("limitations", []),
        meta.get("limitations", []) if isinstance(meta, dict) else [],
        event_meta.get("limitations", []) if isinstance(event_meta, dict) else [],
    )
    output["evidence"] = list(output.get("evidence", [])) + list(macro_event_evidence or [])
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
    if isinstance(event_meta, dict):
        output.setdefault("provider_meta", {})["macro_event_calendar"] = event_meta

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
    peer_context, _, pmeta = hub.get_peer_context(ticker)
    ownership_items, _, ometa = hub.get_ownership_identity(ticker)
    catalyst_items, _, ckmeta = hub.get_8k_exhibits(ticker)
    ir_events, _, irmeta = hub.get_ir_press_release_events(ticker)
    estimate_revision, _, ermeta = hub.get_estimate_revision(ticker)
    ownership_snapshot, _, ysmeta = hub.get_structured_ownership(ticker)
    history_snapshot, _, yhmeta = hub.get_fundamental_history(ticker)
    if isinstance(estimate_revision, dict):
        financials = {**financials, **estimate_revision}
        yahoo_targets = estimate_revision.get("analyst_price_targets_yahoo") or {}
        if financials.get("price_target_consensus") is None and yahoo_targets.get("mean") is not None:
            financials["price_target_consensus"] = yahoo_targets.get("mean")
            financials["price_target_median"] = yahoo_targets.get("median")
            financials["price_target_high"] = yahoo_targets.get("high")
            financials["price_target_low"] = yahoo_targets.get("low")
            current_price = financials.get("current_price")
            target = financials.get("price_target_consensus")
            if current_price not in (None, 0) and target is not None:
                financials["price_target_upside_pct"] = round((target / current_price - 1) * 100, 2)
        if financials.get("next_earnings_date") is None and estimate_revision.get("next_earnings_date_yahoo"):
            financials["next_earnings_date"] = estimate_revision.get("next_earnings_date_yahoo")
    if isinstance(ownership_snapshot, dict):
        financials = {**financials, **ownership_snapshot}
    if isinstance(history_snapshot, dict):
        financials = {**financials, **history_snapshot}
    output = fundamental_analyst_run(
        ticker, financials, sec_data=sec_data,
        run_id=state.get("run_id", ""), as_of=as_of,
        horizon_days=horizon_days,
        focus_areas=focus_areas,
        state=state,
        history=(history_snapshot or {}).get("valuation_history_real"),
        peers=(peer_context or {}).get("peers"),
        ownership_items=ownership_items,
        ownership_snapshot=ownership_snapshot,
        catalyst_items=list(catalyst_items or []) + list(ir_events or []),
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
        pmeta.get("limitations", []) if isinstance(pmeta, dict) else [],
        ometa.get("limitations", []) if isinstance(ometa, dict) else [],
        ckmeta.get("limitations", []) if isinstance(ckmeta, dict) else [],
        irmeta.get("limitations", []) if isinstance(irmeta, dict) else [],
        ermeta.get("limitations", []) if isinstance(ermeta, dict) else [],
        ysmeta.get("limitations", []) if isinstance(ysmeta, dict) else [],
        yhmeta.get("limitations", []) if isinstance(yhmeta, dict) else [],
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
    output.setdefault("provider_meta", {})["peers"] = pmeta
    output.setdefault("provider_meta", {})["ownership"] = ometa
    output.setdefault("provider_meta", {})["sec_8k"] = ckmeta
    output.setdefault("provider_meta", {})["ir_press_release"] = irmeta
    output.setdefault("provider_meta", {})["estimate_revision"] = ermeta
    output.setdefault("provider_meta", {})["structured_ownership"] = ysmeta
    output.setdefault("provider_meta", {})["fundamental_history"] = yhmeta

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


def _collect_sentiment_upcoming_events(state: InvestmentState, ticker: str, as_of: str) -> list[dict]:
    ticker = str(ticker).strip().upper()
    if isinstance(state.get("event_calendar"), list) and state.get("event_calendar"):
        raw_events = state.get("event_calendar", []) or []
    else:
        raw_events = []
        macro = state.get("macro_analysis", {}) if isinstance(state.get("macro_analysis"), dict) else {}
        raw_events.extend(macro.get("macro_event_calendar", []) or [])
        fundamental = state.get("fundamental_analysis", {}) if isinstance(state.get("fundamental_analysis"), dict) else {}
        raw_events.extend(fundamental.get("catalyst_calendar", []) or [])

    out: list[dict] = []
    for raw in raw_events:
        if not isinstance(raw, dict):
            continue
        raw_ticker = str(raw.get("ticker", "")).strip().upper()
        if raw_ticker and raw_ticker not in {ticker, "__GLOBAL__"}:
            continue
        days = raw.get("days_to_event")
        if not isinstance(days, int):
            days = _event_days_to(as_of, raw.get("date"))
        status = str(raw.get("status", "")).strip().lower() or "upcoming"
        if days is not None and 0 <= days <= 7:
            status = "imminent"
        elif days is not None and days < 0:
            status = "stale"
        out.append(
            {
                "type": str(raw.get("type", "")).strip().lower() or "event",
                "subtype": str(raw.get("subtype", "")).strip() or str(raw.get("title", "")).strip(),
                "title": str(raw.get("title", "")).strip() or str(raw.get("subtype", "")).strip(),
                "date": str(raw.get("date", "")).strip(),
                "days_to_event": days,
                "status": status,
                "confirmed": bool(raw.get("confirmed")) or str(raw.get("source_classification", "")).strip().lower() == "confirmed",
                "source_classification": str(raw.get("source_classification", "")).strip().lower() or ("confirmed" if raw.get("confirmed") else "inferred"),
                "source": str(raw.get("source", "")).strip() or str(raw.get("resolver_path", "")).strip() or "event_calendar",
                "notes": str(raw.get("notes", "")).strip() or str(raw.get("expected_scenario", "")).strip(),
            }
        )
    out.sort(key=lambda item: (item.get("days_to_event") is None, item.get("days_to_event") if item.get("days_to_event") is not None else 9999))
    return out[:8]


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
    indicators, news_evidence, meta = hub.get_news_sentiment(ticker, seed=seed)
    market_snapshot, market_evidence, market_meta = hub.get_sentiment_market_snapshot(ticker)
    ownership_snapshot, ownership_evidence, ownership_meta = hub.get_structured_ownership(ticker)
    merged_indicators = dict(indicators or {})
    merged_indicators.update({k: v for k, v in (market_snapshot or {}).items() if v is not None})
    if isinstance(ownership_snapshot, dict):
        for src_key, dst_key in (
            ("institutions_percent_held", "institutions_percent_held"),
            ("institutional_top10_pct", "institutional_top10_pct"),
            ("crowding_risk", "ownership_crowding_risk"),
            ("insider_net_activity", "insider_net_activity"),
            ("incremental_buyer_seller_map", "incremental_buyer_seller_map"),
        ):
            value = ownership_snapshot.get(src_key)
            if value not in (None, "", [], {}):
                merged_indicators[dst_key] = value
    merged_indicators["upcoming_events"] = _collect_sentiment_upcoming_events(state, ticker, as_of)
    output = sentiment_analyst_run(
        ticker, merged_indicators,
        run_id=state.get("run_id", ""), as_of=as_of,
        horizon_days=horizon_days,
        focus_areas=focus_areas,
        state=state,
        source_name="mock" if mode == "mock" else "multi_source",
    )
    rerun_reason = _rerun_reason_for_desk(state, "sentiment")
    prev_sentiment = state.get("sentiment_analysis", {}) if isinstance(state.get("sentiment_analysis"), dict) else {}
    output["limitations"] = _merge_limitations(
        output.get("limitations", []),
        meta.get("limitations", []) if isinstance(meta, dict) else [],
        market_meta.get("limitations", []) if isinstance(market_meta, dict) else [],
        ownership_meta.get("limitations", []) if isinstance(ownership_meta, dict) else [],
    )
    output["evidence"] = list(output.get("evidence", [])) + list(news_evidence or []) + list(market_evidence or []) + list(ownership_evidence or [])
    output = _attach_decision_change_log(
        desk="sentiment",
        output=output,
        state=state,
        prev_output=prev_sentiment,
        rerun_reason=rerun_reason,
        ticker=ticker,
    )
    if isinstance(meta, dict):
        output.setdefault("provider_meta", {})["sentiment_news"] = meta
    if isinstance(market_meta, dict):
        output.setdefault("provider_meta", {})["sentiment_market"] = market_meta
    if isinstance(ownership_meta, dict):
        output.setdefault("provider_meta", {})["sentiment_ownership"] = ownership_meta
    output.setdefault("provider_meta", {})["sentiment"] = {
        "data_ok": bool(
            (meta or {}).get("data_ok")
            or (market_meta or {}).get("data_ok")
            or (ownership_meta or {}).get("data_ok")
        ),
        "limitations": _merge_limitations(
            [],
            meta.get("limitations", []) if isinstance(meta, dict) else [],
            market_meta.get("limitations", []) if isinstance(market_meta, dict) else [],
            ownership_meta.get("limitations", []) if isinstance(ownership_meta, dict) else [],
        ),
    }

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


def _quant_select_pair_ticker(
    *,
    ticker: str,
    intent: str,
    asset_type: str,
    scenario_tags: Optional[list[str]] = None,
) -> str:
    t = str(ticker or "").strip().upper()
    it = str(intent or "").strip().lower()
    at = str(asset_type or "").strip().upper()
    tags = set(_normalize_scenario_tags(scenario_tags or []))
    if it in {"market_outlook", "hedge_design"} or "event_risk" in tags:
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


def _safe_max_drawdown(prices: Any, window: int = 60) -> float | None:
    if not hasattr(prices, "__len__") or len(prices) < window + 2:
        return None
    try:
        p = np.asarray(prices, dtype=float)[-window:]
        running_max = np.maximum.accumulate(p)
        drawdowns = (p / np.where(running_max == 0, np.nan, running_max)) - 1.0
        worst = np.nanmin(drawdowns)
        if np.isnan(worst):
            return None
        return abs(float(worst))
    except Exception:
        return None


def _safe_downside_beta(prices: Any, market_prices: Any, window: int = 60) -> float | None:
    if not hasattr(prices, "__len__") or not hasattr(market_prices, "__len__"):
        return None
    n = min(len(prices), len(market_prices))
    if n < window + 2:
        return None
    try:
        p = np.asarray(prices, dtype=float)[-n:]
        m = np.asarray(market_prices, dtype=float)[-n:]
        rp = np.diff(np.log(p))[-window:]
        rm = np.diff(np.log(m))[-window:]
        if len(rp) == 0 or len(rm) == 0:
            return None
        mask = rm < 0
        if int(np.sum(mask)) < 5:
            return None
        rm_down = rm[mask]
        rp_down = rp[mask]
        var_down = float(np.var(rm_down, ddof=0))
        if var_down < 1e-12:
            return None
        cov = float(np.cov(rp_down, rm_down, ddof=0)[0, 1])
        return cov / var_down
    except Exception:
        return None


def _safe_realized_skew(prices: Any, window: int = 60) -> float | None:
    if not hasattr(prices, "__len__") or len(prices) < window + 2:
        return None
    try:
        p = np.asarray(prices, dtype=float)
        rets = np.diff(np.log(p))[-window:]
        if len(rets) < 10:
            return None
        mean = float(np.mean(rets))
        std = float(np.std(rets, ddof=0))
        if std < 1e-12:
            return None
        centered = rets - mean
        skew = float(np.mean(centered ** 3) / (std ** 3))
        if np.isnan(skew):
            return None
        return skew
    except Exception:
        return None


def _safe_vol_forecast_gap(payload: dict, realized_vol: float | None) -> float | None:
    try:
        forecast = (
            (payload.get("market_regime_context", {}) or {})
            .get("volatility_forecast", {})
            .get("t_plus_1_volatility")
        )
        if forecast is None or realized_vol is None:
            return None
        return float(forecast) - float(realized_vol)
    except (TypeError, ValueError):
        return None


def _extract_quant_event_context(state: InvestmentState, ticker: str) -> dict:
    ticker = str(ticker or "").strip().upper()
    raw_calendar = list(state.get("event_calendar", []) or [])
    if not raw_calendar:
        macro = state.get("macro_analysis", {}) if isinstance(state.get("macro_analysis"), dict) else {}
        for raw in macro.get("macro_event_calendar", []) or []:
            if not isinstance(raw, dict):
                continue
            raw_calendar.append(
                {
                    "ticker": "__GLOBAL__",
                    "type": raw.get("type", "macro_event"),
                    "subtype": raw.get("title", raw.get("subtype", "")),
                    "status": raw.get("status", "upcoming"),
                    "days_to_event": raw.get("days_to_event"),
                    "source_classification": raw.get("source_classification", "confirmed"),
                    "confirmed": str(raw.get("source_classification", "")).strip().lower() == "confirmed",
                }
            )
        fundamental = state.get("fundamental_analysis", {}) if isinstance(state.get("fundamental_analysis"), dict) else {}
        fundamental_ticker = str(fundamental.get("ticker", "")).strip().upper() or ticker
        for raw in fundamental.get("catalyst_calendar", []) or []:
            if not isinstance(raw, dict):
                continue
            raw_calendar.append(
                {
                    "ticker": fundamental_ticker,
                    "type": raw.get("type", "catalyst"),
                    "subtype": raw.get("source_title", raw.get("type", "")),
                    "status": raw.get("status", "upcoming"),
                    "days_to_event": raw.get("days_to_event"),
                    "source_classification": raw.get("source_classification", "confirmed"),
                    "confirmed": str(raw.get("source_classification", "")).strip().lower() == "confirmed",
                }
            )
    events: list[dict] = []
    for raw in raw_calendar:
        if not isinstance(raw, dict):
            continue
        event_ticker = str(raw.get("ticker", "")).strip().upper()
        affected = {
            str(item).strip().upper()
            for item in (raw.get("affected_tickers", []) or [])
            if str(item).strip()
        }
        if event_ticker not in {"", "__GLOBAL__", ticker} and ticker not in affected:
            continue
        events.append(raw)

    nearest_days: int | None = None
    confirmed = 0
    imminent = 0
    triggered = 0
    labels: list[str] = []
    for event in events:
        days = event.get("days_to_event")
        if isinstance(days, int) and days >= 0 and (nearest_days is None or days < nearest_days):
            nearest_days = days
        status = str(event.get("status", "")).strip().lower()
        if status == "imminent":
            imminent += 1
        if status == "triggered":
            triggered += 1
        source_class = str(event.get("source_classification", "")).strip().lower()
        if source_class == "confirmed" or bool(event.get("confirmed")):
            confirmed += 1
        label = str(event.get("subtype", "")).strip() or str(event.get("type", "")).strip()
        if label and label not in labels:
            labels.append(label)

    return {
        "event_count": len(events),
        "confirmed_event_count": confirmed,
        "imminent_event_count": imminent,
        "triggered_event_count": triggered,
        "nearest_event_days": nearest_days,
        "event_labels": labels[:5],
    }


def _build_quant_data_provenance(payload: dict, signals: dict, event_context: dict) -> dict:
    sources = {
        "price_history": signals.get("event_return_5d") is not None and signals.get("trend_60d") is not None,
        "pair_history": signals.get("pair_relative_strength_20d") is not None,
        "market_history": signals.get("corr_with_market_60d") is not None and signals.get("beta_to_market_60d") is not None,
        "regime_engine": bool((payload.get("market_regime_context", {}) or {}).get("state_probabilities")),
        "risk_engine": (payload.get("portfolio_risk_parameters", {}) or {}).get("asset_cvar_99_daily") is not None,
        "event_calendar": event_context.get("event_count", 0) > 0,
    }
    raw_components = sum(1 for v in sources.values() if v)
    if raw_components >= 5:
        quality = "high"
    elif raw_components >= 3:
        quality = "medium"
    else:
        quality = "low"
    return {
        "sources": sources,
        "raw_components": raw_components,
        "coverage_score": round(raw_components / len(sources), 4),
        "quality": quality,
    }


def _build_quant_relative_value_views(
    ticker: str,
    pair_ticker: str,
    signals: dict,
    decision: dict,
) -> list[dict]:
    views: list[dict] = []
    rel = signals.get("pair_relative_strength_20d")
    if rel is not None:
        stance = "outperform" if float(rel) > 0.02 else ("underperform" if float(rel) < -0.02 else "neutral")
        views.append(
            {
                "pair": f"{ticker}/{pair_ticker}",
                "stance": stance,
                "metric": "pair_relative_strength_20d",
                "value": round(float(rel), 6),
                "confidence": decision.get("confidence", 0.5),
            }
        )
    beta = signals.get("beta_to_market_60d")
    if beta is not None:
        market_tilt = "high_beta" if float(beta) > 1.1 else ("defensive" if float(beta) < 0.8 else "market_like")
        views.append(
            {
                "pair": f"{ticker}/SPY",
                "stance": market_tilt,
                "metric": "beta_to_market_60d",
                "value": round(float(beta), 6),
                "confidence": decision.get("confidence", 0.5),
            }
        )
    return views


def _build_quant_execution_plan(
    decision: dict,
    signals: dict,
    event_context: dict,
    *,
    horizon_days: int,
    pair_ticker: str,
    risk_budget: str,
) -> dict:
    vol_shift = signals.get("vol_shift_20d_vs_60d")
    nearest_days = event_context.get("nearest_event_days")
    staged = bool((vol_shift is not None and float(vol_shift) > 1.15) or (nearest_days is not None and nearest_days <= 7))
    review_drawdown = signals.get("drawdown_60d")
    review_drawdown_pct = round(float(review_drawdown) * 100, 2) if review_drawdown is not None else None
    return {
        "action": decision.get("decision"),
        "allocation_pct": decision.get("final_allocation_pct"),
        "sizing_mode": "risk_scaled_event_aware",
        "entry_style": "staggered" if staged else "standard",
        "holding_horizon_days": horizon_days,
        "pair_reference": pair_ticker,
        "risk_budget": risk_budget,
        "review_thresholds": {
            "drawdown_pct": review_drawdown_pct,
            "vol_shift_trigger": 1.25,
            "event_window_days": 7,
        },
    }


def _build_quant_monitoring_triggers(signals: dict, event_context: dict, payload: dict) -> list[dict]:
    triggers: list[dict] = []
    regime_prob = (
        (payload.get("market_regime_context", {}) or {})
        .get("state_probabilities", {})
        .get("regime_2_high_vol", 0.0)
    )
    if float(regime_prob or 0.0) >= 0.35:
        triggers.append(
            {
                "name": "High-vol regime probability",
                "metric": "regime_2_high_vol",
                "current_value": round(float(regime_prob), 4),
                "trigger": "> 0.50",
                "action": "gross exposure 감축 또는 HOLD 유지",
                "priority": 1,
            }
        )
    vol_shift = signals.get("vol_shift_20d_vs_60d")
    if vol_shift is not None and float(vol_shift) >= 1.20:
        triggers.append(
            {
                "name": "Volatility expansion",
                "metric": "vol_shift_20d_vs_60d",
                "current_value": round(float(vol_shift), 4),
                "trigger": "> 1.25",
                "action": "allocation 축소 및 risk refresh",
                "priority": 1,
            }
        )
    downside_beta = signals.get("downside_beta_60d")
    if downside_beta is not None and float(downside_beta) >= 1.10:
        triggers.append(
            {
                "name": "Downside beta spike",
                "metric": "downside_beta_60d",
                "current_value": round(float(downside_beta), 4),
                "trigger": "> 1.20",
                "action": "beta 노출 재검토",
                "priority": 2,
            }
        )
    drawdown = signals.get("drawdown_60d")
    if drawdown is not None and float(drawdown) >= 0.08:
        triggers.append(
            {
                "name": "Drawdown breach",
                "metric": "drawdown_60d",
                "current_value": round(float(drawdown), 4),
                "trigger": "> 0.10",
                "action": "signal reset 여부 재평가",
                "priority": 1,
            }
        )
    nearest_days = event_context.get("nearest_event_days")
    if nearest_days is not None and nearest_days <= 7:
        triggers.append(
            {
                "name": "Event proximity",
                "metric": "nearest_event_days",
                "current_value": nearest_days,
                "trigger": "<= 7",
                "action": "event window 동안 sizing 보수화",
                "priority": 1,
            }
        )
    forecast_gap = signals.get("vol_forecast_gap")
    if forecast_gap is not None and float(forecast_gap) >= 0.02:
        triggers.append(
            {
                "name": "Forecast vol repricing",
                "metric": "vol_forecast_gap",
                "current_value": round(float(forecast_gap), 4),
                "trigger": "> 0.03",
                "action": "변동성 재가격 반영 여부 점검",
                "priority": 2,
            }
        )
    return triggers


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


def _safe_return_vector(prices: Any, window: int = 60) -> np.ndarray | None:
    if not hasattr(prices, "__len__") or len(prices) < window + 2:
        return None
    try:
        p = np.asarray(prices, dtype=float)
        rets = np.diff(np.log(p))
        tail = rets[-window:]
        if len(tail) < max(10, window // 2):
            return None
        return tail
    except Exception:
        return None


def _weighted_average_correlation(
    return_map: dict[str, np.ndarray | None],
    base_weights: dict[str, float],
    ticker: str,
) -> float | None:
    current = return_map.get(ticker)
    if current is None:
        return None
    weighted_sum = 0.0
    weight_sum = 0.0
    for other, base_weight in (base_weights or {}).items():
        if other == ticker or float(base_weight) <= 1e-12:
            continue
        other_ret = return_map.get(other)
        if other_ret is None:
            continue
        n = min(len(current), len(other_ret))
        if n < 20:
            continue
        corr = np.corrcoef(current[-n:], other_ret[-n:])[0, 1]
        if np.isnan(corr):
            continue
        weighted_sum += float(base_weight) * float(corr)
        weight_sum += float(base_weight)
    if weight_sum <= 1e-12:
        return None
    return weighted_sum / weight_sum


def _cap_relative_weights(weights: dict[str, float], cap: float) -> dict[str, float]:
    capped = {str(t): max(0.0, float(w)) for t, w in (weights or {}).items()}
    if not capped:
        return {}
    cap = max(0.05, min(float(cap), 1.0))
    for _ in range(6):
        overflow = 0.0
        open_names: list[str] = []
        for ticker, weight in list(capped.items()):
            if weight > cap:
                overflow += weight - cap
                capped[ticker] = cap
            else:
                open_names.append(ticker)
        if overflow <= 1e-12 or not open_names:
            break
        room = {t: max(0.0, cap - capped[t]) for t in open_names}
        room_total = float(sum(room.values()))
        if room_total <= 1e-12:
            break
        for ticker in open_names:
            capped[ticker] += overflow * (room[ticker] / room_total)
    return _normalize_long_only_weights(capped)


def _portfolio_construction_base_weights(state: InvestmentState, universe: list[str], main: str) -> dict[str, float]:
    raw = state.get("positions_proposed", {}) if isinstance(state.get("positions_proposed"), dict) else {}
    base = _normalize_long_only_weights(raw)
    if base:
        for ticker in universe:
            base.setdefault(ticker, 0.0)
        return base

    plan = state.get("book_allocation_plan", {}) if isinstance(state.get("book_allocation_plan"), dict) else {}
    weights = plan.get("weights_relative", {}) if isinstance(plan.get("weights_relative"), dict) else {}
    base = _normalize_long_only_weights(weights)
    if base:
        for ticker in universe:
            base.setdefault(ticker, 0.0)
        return base

    if universe:
        fallback = {ticker: 0.0 for ticker in universe}
        fallback[main] = 1.0
        return fallback
    return {main: 1.0} if main else {}


def _portfolio_construction_signal_snapshot(
    state: InvestmentState,
    ticker: str,
    competition_by_ticker: dict[str, dict],
    main: str,
) -> dict:
    row = competition_by_ticker.get(ticker, {}) if isinstance(competition_by_ticker, dict) else {}
    snapshot = {
        "conviction_score": float(row.get("conviction_score", 0.0) or 0.0),
        "expected_return_score": float(row.get("expected_return_score", 0.0) or 0.0),
        "downside_penalty": float(row.get("downside_penalty", 0.0) or 0.0),
        "catalyst_proximity_score": float(row.get("catalyst_proximity_score", 0.0) or 0.0),
        "source": "capital_competition" if row else "fallback",
    }
    if ticker != main:
        hedge_lite = state.get("hedge_lite", {}) if isinstance(state.get("hedge_lite"), dict) else {}
        hedge_rows = hedge_lite.get("hedges", {}) if isinstance(hedge_lite.get("hedges"), dict) else {}
        hedge_row = hedge_rows.get(ticker, {}) if isinstance(hedge_rows.get(ticker), dict) else {}
        if hedge_row:
            snapshot["conviction_score"] = float(snapshot["conviction_score"]) + 0.40 * float(hedge_row.get("score", 0.0) or 0.0)
            snapshot["source"] = "capital_competition+hedge_lite"
        return snapshot

    fundamental = state.get("fundamental_analysis", {}) if isinstance(state.get("fundamental_analysis"), dict) else {}
    technical = state.get("technical_analysis", {}) if isinstance(state.get("technical_analysis"), dict) else {}
    sentiment = state.get("sentiment_analysis", {}) if isinstance(state.get("sentiment_analysis"), dict) else {}
    expected_profile = fundamental.get("expected_return_profile", {}) if isinstance(fundamental.get("expected_return_profile"), dict) else {}
    if not row:
        anchor_upside = expected_profile.get("anchor_upside_pct")
        bear_downside = expected_profile.get("bear_downside_pct")
        if anchor_upside is not None:
            snapshot["expected_return_score"] = _clip(float(anchor_upside) / 20.0)
        if bear_downside is not None:
            snapshot["downside_penalty"] = _clip(abs(float(bear_downside)) / 20.0, 0.0, 1.0)
        f_score = float(fundamental.get("signal_strength", 0.0) or 0.0)
        q_score = float((technical.get("signal_stack", {}) or {}).get("total_score", 0.0) or 0.0)
        s_score = float(sentiment.get("signal_strength", 0.0) or 0.0)
        snapshot["conviction_score"] = _clip(0.45 * f_score + 0.40 * q_score + 0.15 * s_score)
    catalyst_calendar = fundamental.get("catalyst_calendar", []) if isinstance(fundamental.get("catalyst_calendar"), list) else []
    first_catalyst = next((item for item in catalyst_calendar if isinstance(item, dict)), None)
    if first_catalyst is not None:
        days = first_catalyst.get("days_to_event")
        if isinstance(days, int):
            if days <= 7:
                snapshot["catalyst_proximity_score"] = max(float(snapshot["catalyst_proximity_score"]), 0.80)
            elif days <= 30:
                snapshot["catalyst_proximity_score"] = max(float(snapshot["catalyst_proximity_score"]), 0.40)
    return snapshot


def _portfolio_construction_event_penalty(event_calendar: list[dict], ticker: str) -> tuple[float, dict]:
    confirmed = 0
    imminent = 0
    triggered = 0
    for item in event_calendar or []:
        if not isinstance(item, dict):
            continue
        event_ticker = str(item.get("ticker", "")).strip().upper()
        if event_ticker not in {"", "__GLOBAL__", ticker}:
            continue
        status = str(item.get("status", "")).strip().lower()
        if status == "triggered":
            triggered += 1
        elif status == "imminent":
            imminent += 1
        if bool(item.get("confirmed")) or str(item.get("source_classification", "")).strip().lower() == "confirmed":
            confirmed += 1
    penalty = min(0.40, 0.05 * confirmed + 0.08 * imminent + 0.12 * triggered)
    return penalty, {
        "confirmed_event_count": confirmed,
        "imminent_event_count": imminent,
        "triggered_event_count": triggered,
    }


def _build_portfolio_construction_monitoring_triggers(
    rows: dict[str, dict],
    turnover_estimate: float,
    diversification_score: float,
    cap_relative: float,
) -> list[dict]:
    triggers: list[dict] = []
    if turnover_estimate >= 0.28:
        triggers.append({
            "name": "Turnover budget",
            "metric": "turnover_estimate",
            "current_value": round(float(turnover_estimate), 4),
            "trigger": "> 0.35",
            "action": "construction 재균형 강도를 낮추고 catalyst 근거 재검토",
            "priority": 2,
        })
    if diversification_score <= 0.32:
        triggers.append({
            "name": "Correlation crowding",
            "metric": "diversification_score",
            "current_value": round(float(diversification_score), 4),
            "trigger": "< 0.30",
            "action": "상관 집중 포지션 축소 및 hedge sleeve 확대 검토",
            "priority": 1,
        })
    for ticker, row in rows.items():
        target_weight = float(row.get("target_weight", 0.0) or 0.0)
        if target_weight >= max(0.45, cap_relative * 0.97):
            triggers.append({
                "name": "Concentration pressure",
                "metric": f"{ticker}_target_weight",
                "current_value": round(target_weight, 4),
                "trigger": f"> {round(cap_relative, 4)}",
                "action": "single-name concentration 재조정",
                "priority": 1,
            })
        event_penalty = float(row.get("event_penalty", 0.0) or 0.0)
        if event_penalty >= 0.20:
            triggers.append({
                "name": "Event cluster risk",
                "metric": f"{ticker}_event_penalty",
                "current_value": round(event_penalty, 4),
                "trigger": "> 0.20",
                "action": "event window 동안 target weight 보수화",
                "priority": 2,
            })
    return triggers[:8]


def portfolio_construction_quant_node(state: InvestmentState) -> dict:
    """Quant portfolio construction layer: positions_proposed를 construction-aware하게 재조정."""
    if not _has_position_review_snapshot(state):
        return {}
    _log(state, "portfolio_construction_quant", "enter")

    universe = [
        str(t).strip().upper()
        for t in (state.get("universe", []) or [])
        if str(t).strip()
    ]
    main = str(state.get("target_ticker", "")).strip().upper() or (universe[0] if universe else "")
    if main and main not in universe:
        universe = [main] + [t for t in universe if t != main]
    if not universe and main:
        universe = [main]
    if not universe:
        return {}

    as_of = str(state.get("as_of", "")).strip()
    mode = str(state.get("mode", "mock")).strip() or "mock"
    seed = _make_seed(state) + 29
    base_weights = _portfolio_construction_base_weights(state, universe, main)
    if not base_weights:
        return {}

    book_plan = state.get("book_allocation_plan", {}) if isinstance(state.get("book_allocation_plan"), dict) else {}
    allocator_guidance = (
        ((state.get("orchestrator_directives", {}) or {}).get("allocator_guidance", {}))
        if isinstance(state.get("orchestrator_directives", {}), dict)
        else {}
    )
    gross_target = float(
        book_plan.get("gross_target")
        or allocator_guidance.get("target_gross_exposure")
        or 1.0
    )
    gross_target = max(0.10, min(gross_target, 1.0))
    single_name_cap = float(
        book_plan.get("single_name_cap")
        or allocator_guidance.get("single_name_cap")
        or min(0.55, gross_target)
    )
    single_name_cap = max(0.10, min(single_name_cap, gross_target))
    cap_relative = max(0.10, min(single_name_cap / max(gross_target, 1e-6), 1.0))

    capital_competition = state.get("capital_competition", []) if isinstance(state.get("capital_competition"), list) else []
    competition_by_ticker = {
        str(row.get("ticker", "")).strip().upper(): row
        for row in capital_competition
        if isinstance(row, dict) and str(row.get("ticker", "")).strip()
    }
    event_calendar = state.get("event_calendar", []) if isinstance(state.get("event_calendar"), list) else []

    hub = DataHub(run_id=state.get("run_id", ""), as_of=as_of, mode=mode)
    market_prices, market_evidence, _ = hub.get_market_series("SPY", lookback_days=260, seed=seed)
    market_returns = _safe_return_vector(market_prices, window=60)
    evidence = list(market_evidence)

    price_map: dict[str, Any] = {}
    return_map: dict[str, np.ndarray | None] = {}
    metric_rows: dict[str, dict] = {}
    vol_values: list[float] = []
    for idx, ticker in enumerate(universe):
        prices, price_evidence, _ = hub.get_price_series(ticker, lookback_days=260, seed=seed + (idx + 1) * 17)
        price_map[ticker] = prices
        return_map[ticker] = _safe_return_vector(prices, window=60)
        evidence.extend(price_evidence)
        vol_60d = _safe_ann_vol(prices, window=60)
        if vol_60d is not None:
            vol_values.append(float(vol_60d))
        metric_rows[ticker] = {
            "base_weight": round(float(base_weights.get(ticker, 0.0) or 0.0), 6),
            "vol_20d_ann": _safe_ann_vol(prices, window=20),
            "vol_60d_ann": vol_60d,
            "corr_with_market_60d": _safe_corr(prices, market_prices, window=60),
            "beta_to_market_60d": _safe_beta_to_main(prices, market_prices, window=60) if market_returns is not None else None,
            "downside_beta_60d": _safe_downside_beta(prices, market_prices, window=60) if market_returns is not None else None,
            "drawdown_60d": _safe_max_drawdown(prices, window=60),
            "liquidity_proxy": _liquidity_proxy_score(ticker, state.get("asset_type_by_ticker", {})),
        }

    median_vol = float(np.median(np.asarray(vol_values, dtype=float))) if vol_values else None
    raw_weights: dict[str, float] = {}
    diversification_components: list[float] = []
    rows: dict[str, dict] = {}
    for ticker in universe:
        signal = _portfolio_construction_signal_snapshot(state, ticker, competition_by_ticker, main)
        weighted_corr = _weighted_average_correlation(return_map, base_weights, ticker)
        event_penalty, event_counts = _portfolio_construction_event_penalty(event_calendar, ticker)
        vol_60d = metric_rows[ticker].get("vol_60d_ann")
        inv_vol_multiplier = 1.0
        if median_vol is not None and vol_60d not in (None, 0):
            inv_vol_multiplier = max(0.55, min(median_vol / float(vol_60d), 1.65))
        diversification_multiplier = 1.0
        if weighted_corr is not None:
            diversification_multiplier = max(0.45, min(1.30, 1.0 + (0.25 - float(weighted_corr)) * 0.85))
            diversification_components.append(1.0 - max(float(weighted_corr), 0.0))
        liquidity_multiplier = 0.75 + 0.35 * float(metric_rows[ticker].get("liquidity_proxy", 0.50) or 0.50)
        drawdown_penalty = max(0.0, min(float(metric_rows[ticker].get("drawdown_60d", 0.0) or 0.0) * 3.5, 0.45))
        beta_penalty = 0.0
        beta = metric_rows[ticker].get("beta_to_market_60d")
        if beta is not None:
            beta_penalty = max(0.0, min(abs(float(beta) - 1.0) * 0.12, 0.20))

        conviction = float(signal.get("conviction_score", 0.0) or 0.0)
        expected_return = float(signal.get("expected_return_score", 0.0) or 0.0)
        downside_penalty = float(signal.get("downside_penalty", 0.0) or 0.0)
        catalyst_score = float(signal.get("catalyst_proximity_score", 0.0) or 0.0)

        alpha_multiplier = max(
            0.30,
            1.0
            + 0.35 * conviction
            + 0.25 * expected_return
            - 0.20 * downside_penalty
            + 0.08 * catalyst_score,
        )
        risk_multiplier = max(
            0.15,
            inv_vol_multiplier
            * diversification_multiplier
            * liquidity_multiplier
            * (1.0 - drawdown_penalty)
            * (1.0 - beta_penalty)
            * (1.0 - event_penalty),
        )
        base_anchor = max(0.02, float(base_weights.get(ticker, 0.0) or 0.0))
        if ticker == main:
            base_anchor = max(base_anchor, 0.20)
        raw_weight = base_anchor * alpha_multiplier * risk_multiplier
        raw_weights[ticker] = raw_weight
        rows[ticker] = {
            **metric_rows[ticker],
            "weighted_avg_corr_60d": round(float(weighted_corr), 6) if weighted_corr is not None else None,
            "inv_vol_multiplier": round(float(inv_vol_multiplier), 6),
            "diversification_multiplier": round(float(diversification_multiplier), 6),
            "liquidity_multiplier": round(float(liquidity_multiplier), 6),
            "drawdown_penalty": round(float(drawdown_penalty), 6),
            "beta_penalty": round(float(beta_penalty), 6),
            "conviction_score": round(float(conviction), 6),
            "expected_return_score": round(float(expected_return), 6),
            "downside_penalty": round(float(downside_penalty), 6),
            "catalyst_proximity_score": round(float(catalyst_score), 6),
            "event_penalty": round(float(event_penalty), 6),
            "event_counts": event_counts,
            "raw_weight": round(float(raw_weight), 6),
            "signal_source": str(signal.get("source", "fallback")).strip() or "fallback",
        }
        evidence.append(
            make_evidence(
                metric=f"{ticker.lower()}_construction_raw_weight",
                value=round(float(raw_weight), 6),
                source_name="portfolio_construction_quant",
                source_type="model",
                quality=0.86,
                as_of=as_of,
            )
        )

    raw_norm = _normalize_long_only_weights(raw_weights)
    turnover_raw = 0.5 * float(sum(abs(float(raw_norm.get(t, 0.0)) - float(base_weights.get(t, 0.0))) for t in universe))
    avg_catalyst = float(np.mean([rows[t]["catalyst_proximity_score"] for t in universe])) if universe else 0.0
    turnover_blend = max(0.45, min(0.85, 0.55 + 0.60 * turnover_raw - 0.15 * avg_catalyst))

    blended = {}
    for ticker in universe:
        blended[ticker] = (
            turnover_blend * float(base_weights.get(ticker, 0.0) or 0.0)
            + (1.0 - turnover_blend) * float(raw_norm.get(ticker, 0.0) or 0.0)
        )
    blended = _normalize_long_only_weights(blended)
    final_weights = _cap_relative_weights(blended, cap_relative)
    turnover_final = 0.5 * float(sum(abs(float(final_weights.get(t, 0.0)) - float(base_weights.get(t, 0.0))) for t in universe))

    for ticker in universe:
        target_weight = float(final_weights.get(ticker, 0.0) or 0.0)
        base_weight = float(base_weights.get(ticker, 0.0) or 0.0)
        delta = target_weight - base_weight
        if base_weight <= 1e-12 and target_weight >= 0.03:
            action = "add"
        elif delta > 0.03:
            action = "scale_up"
        elif delta < -0.03:
            action = "scale_down"
        else:
            action = "hold"
        rows[ticker]["target_weight"] = round(target_weight, 6)
        rows[ticker]["delta_weight"] = round(delta, 6)
        rows[ticker]["rebalance_action"] = action

    diversification_score = max(0.0, min(float(np.mean(diversification_components)) if diversification_components else 0.50, 1.0))
    monitoring_triggers = _build_portfolio_construction_monitoring_triggers(
        rows,
        turnover_estimate=turnover_final,
        diversification_score=diversification_score,
        cap_relative=cap_relative,
    )
    event_risk_level = "high" if any(float(rows[t]["event_penalty"]) >= 0.20 for t in universe) else ("medium" if any(float(rows[t]["event_penalty"]) >= 0.10 for t in universe) else "low")

    portfolio_construction_analysis = {
        "status": "ok",
        "allocator_source": "portfolio_construction_quant_v1",
        "main_ticker": main,
        "universe": universe,
        "base_weights": {t: round(float(base_weights.get(t, 0.0)), 6) for t in universe},
        "weights_proposed": {t: round(float(final_weights.get(t, 0.0)), 6) for t in universe},
        "target_gross_exposure": round(gross_target, 6),
        "single_name_cap": round(single_name_cap, 6),
        "relative_single_name_cap": round(cap_relative, 6),
        "turnover_estimate": round(turnover_final, 6),
        "turnover_blend": round(turnover_blend, 6),
        "diversification_score": round(diversification_score, 6),
        "event_risk_level": event_risk_level,
        "covariance_window_days": 60,
        "rows": rows,
        "rebalances": [
            {
                "ticker": ticker,
                "action": rows[ticker]["rebalance_action"],
                "base_weight": rows[ticker]["base_weight"],
                "target_weight": rows[ticker]["target_weight"],
                "delta_weight": rows[ticker]["delta_weight"],
            }
            for ticker in sorted(universe)
        ],
        "monitoring_triggers": monitoring_triggers,
        "evidence": evidence,
        "build_version": "portfolio_construction_quant_v1",
        "timestamp": as_of,
        "seed": seed,
    }

    audit = dict(state.get("audit", {}) or {})
    audit_paths = dict(audit.get("paths", {}) or {})
    audit_paths["portfolio_construction_quant"] = {
        "status": "ok",
        "timestamp": as_of,
        "seed": seed,
        "turnover_estimate": round(turnover_final, 6),
        "diversification_score": round(diversification_score, 6),
    }
    audit["paths"] = audit_paths

    _ops_log(
        state,
        "portfolio_construction_quant",
        "construction output",
        [
            f"gross_target={gross_target} cap={single_name_cap} base={base_weights}",
            f"turnover={round(turnover_final, 4)} diversification={round(diversification_score, 4)}",
            f"weights={final_weights}",
        ],
    )
    _log(state, "portfolio_construction_quant", "exit", {"turnover": round(turnover_final, 4), "diversification": round(diversification_score, 4)})
    return {
        "portfolio_construction_analysis": portfolio_construction_analysis,
        "positions_proposed": final_weights,
        "audit": audit,
    }


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
    intent = _state_canonical_intent(state)
    scenario_tags = _state_scenario_tags(state)
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
    if intent == "hedge_design" or _state_has_scenario_tag(state, "event_risk") or (main_vol_shift is not None and float(main_vol_shift) > 1.2):
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
        "scenario_tags": scenario_tags,
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
    trend5 = sig.get("event_return_5d")
    trend20 = sig.get("trend_20d")
    trend60 = sig.get("trend_60d")
    vol_shift = sig.get("vol_shift_20d_vs_60d")
    corr_market = sig.get("corr_with_market_60d")
    beta_market = sig.get("beta_to_market_60d")
    downside_beta = sig.get("downside_beta_60d")
    rel = sig.get("pair_relative_strength_20d")
    drawdown = sig.get("drawdown_60d")
    vol_gap = sig.get("vol_forecast_gap")
    imminent_events = int(sig.get("imminent_event_count") or 0)
    confirmed_events = int(sig.get("confirmed_event_count") or 0)
    triggered_events = int(sig.get("triggered_event_count") or 0)
    regime_prob = (
        (payload.get("market_regime_context", {}) or {})
        .get("state_probabilities", {})
        .get("regime_2_high_vol", 0.0)
    )

    cot = [
        (
            "[Event/Regime] "
            f"5d={trend5}, 20d={trend20}, 60d={trend60}, "
            f"vol_shift={vol_shift}, corr60={corr_market}, beta60={beta_market}, "
            f"down_beta={downside_beta}, dd60={drawdown}, rel20={rel}, vol_gap={vol_gap}"
        ),
        (
            "[Events] "
            f"confirmed={confirmed_events}, imminent={imminent_events}, triggered={triggered_events}"
        ),
        f"[RegimeProb] regime_2_high_vol={regime_prob}",
    ]

    def _s(value: Any, scale: float) -> float:
        if value is None:
            return 0.0
        try:
            return _clip(float(value) * scale)
        except (TypeError, ValueError):
            return 0.0

    trend_score = (
        0.30 * _s(trend5, 12.0)
        + 0.40 * _s(trend20, 10.0)
        + 0.30 * _s(trend60, 6.0)
    )
    beta_component = 0.25 * _clip(1.0 - abs(float(beta_market) - 1.0), -1.0, 1.0) if beta_market is not None else 0.0
    corr_component = 0.15 * _clip(-abs(float(corr_market) - 0.6), -1.0, 0.0) if corr_market is not None else 0.0
    relative_value_score = (
        0.60 * _s(rel, 14.0)
        + beta_component
        + corr_component
    )
    risk_penalty = (
        0.35 * _clip(float(regime_prob or 0.0) * 2.0)
        + 0.20 * _clip(max((float(vol_shift or 1.0) - 1.0), 0.0) * 2.5)
        + 0.20 * _clip(float(drawdown or 0.0) * 6.0)
        + 0.15 * _clip(max((float(downside_beta or 1.0) - 1.0), 0.0) * 2.0)
        + 0.10 * _clip(max((float(vol_gap or 0.0)), 0.0) * 12.0)
    )
    event_penalty = min(0.35, 0.07 * confirmed_events + 0.08 * imminent_events + 0.12 * triggered_events)
    total_score = 0.55 * trend_score + 0.45 * relative_value_score - risk_penalty - event_penalty

    cot.append(
        f"[Score] trend={round(trend_score, 4)}, rv={round(relative_value_score, 4)}, "
        f"risk_penalty={round(risk_penalty, 4)}, event_penalty={round(event_penalty, 4)}, "
        f"total={round(total_score, 4)}"
    )

    risk_off = bool(
        (regime_prob or 0.0) > 0.55
        or (vol_shift or 0.0) > 1.35
        or (drawdown or 0.0) > 0.12
        or triggered_events > 0
    )
    if total_score >= 0.25 and not risk_off:
        decision = "LONG"
        alloc = min(0.10, max(0.02, round(0.10 * min(total_score, 1.0), 4)))
        cot.append("[Decision] trend/rv 우위가 리스크를 상회 → LONG")
    elif total_score <= -0.30 and not risk_off:
        decision = "SHORT"
        alloc = min(0.08, max(0.02, round(0.08 * min(abs(total_score), 1.0), 4)))
        cot.append("[Decision] 하방 trend/rv 열위가 명확 → SHORT")
    else:
        decision = "HOLD"
        alloc = 0.0
        cot.append("[Decision] 이벤트/리스크 우세 또는 신호 혼조 → HOLD")

    confidence = round(max(0.35, min(0.8, 0.55 + 0.20 * abs(total_score) - 0.10 * event_penalty)), 2)

    return {
        "cot_reasoning": " ".join(cot),
        "decision": decision,
        "final_allocation_pct": alloc,
        "confidence": confidence,
        "signal_stack": {
            "trend_score": round(trend_score, 4),
            "relative_value_score": round(relative_value_score, 4),
            "risk_penalty": round(risk_penalty, 4),
            "event_penalty": round(event_penalty, 4),
            "total_score": round(total_score, 4),
        },
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
    intent = _state_canonical_intent(state) or "single_name"
    scenario_tags = _state_scenario_tags(state)
    asset_type = str((state.get("asset_type_by_ticker", {}) or {}).get(ticker, "")).strip().upper() or _infer_asset_type(ticker)
    horizon_days, focus_areas, risk_budget = _get_desk_task(state, "quant", 10)
    pair_ticker = _quant_select_pair_ticker(
        ticker=ticker,
        intent=intent,
        asset_type=asset_type,
        scenario_tags=scenario_tags,
    )
    analysis_mode = "event_regime" if intent in {"market_outlook", "hedge_design", "position_review"} or _state_has_scenario_tag(state, "event_risk") else "statarb"
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
        "trend_60d": _safe_pct_change(prices, 60),
        "vol_20d_ann": _safe_ann_vol(prices, window=20),
        "vol_60d_ann": _safe_ann_vol(prices, window=60),
        "corr_with_market_60d": _safe_corr(prices, market_prices, window=60),
        "beta_to_market_60d": _safe_beta_to_main(prices, market_prices, window=60),
        "downside_beta_60d": _safe_downside_beta(prices, market_prices, window=60),
        "drawdown_60d": _safe_max_drawdown(prices, window=60),
        "realized_skew_60d": _safe_realized_skew(prices, window=60),
        "tail_proxy_5pct_60d": _safe_tail_proxy(prices, window=60),
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
    event_regime_signals["vol_forecast_gap"] = _safe_vol_forecast_gap(payload, event_regime_signals.get("vol_60d_ann"))
    event_context = _extract_quant_event_context(state, ticker)
    event_regime_signals.update(event_context)
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
    signal_stack = dict(decision.get("signal_stack", {}) or {})
    data_provenance = _build_quant_data_provenance(payload, event_regime_signals, event_context)
    execution_plan = _build_quant_execution_plan(
        decision,
        event_regime_signals,
        event_context,
        horizon_days=horizon_days,
        pair_ticker=pair_ticker,
        risk_budget=risk_budget,
    )
    monitoring_triggers = _build_quant_monitoring_triggers(event_regime_signals, event_context, payload)
    relative_value_views = _build_quant_relative_value_views(ticker, pair_ticker, event_regime_signals, decision)

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
                f"vol_shift={event_regime_signals.get('vol_shift_20d_vs_60d')} "
                f"events={event_regime_signals.get('imminent_event_count')}"
            ),
        ],
    )

    # Evidence
    evidence = list(p_ev)
    if z is not None:
        evidence.append(make_evidence(metric="z_score", value=z, source_name="quant_engine", source_type="model", quality=0.9, as_of=as_of))
    if cvar is not None:
        evidence.append(make_evidence(metric="asset_cvar_99_daily", value=cvar, source_name="quant_engine", source_type="model", quality=0.9, as_of=as_of))
    for k in (
        "event_return_5d",
        "trend_20d",
        "trend_60d",
        "vol_shift_20d_vs_60d",
        "beta_to_market_60d",
        "downside_beta_60d",
        "drawdown_60d",
        "vol_forecast_gap",
    ):
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
        "scenario_tags": scenario_tags,
        "analysis_mode": analysis_mode,
        "pair_ticker": pair_ticker,
        "asset_type": asset_type,
        "decision": decision["decision"],
        "final_allocation_pct": decision["final_allocation_pct"],
        "z_score": z,
        "asset_cvar_99_daily": cvar,
        "event_regime_signals": event_regime_signals,
        "signal_stack": signal_stack,
        "execution_plan": execution_plan,
        "monitoring_triggers": monitoring_triggers,
        "relative_value_views": relative_value_views,
        "data_provenance": data_provenance,
        "quant_payload": payload,
        "llm_decision": decision,
        "evidence": evidence,
        "summary": (
            f"Quant({analysis_mode}): {decision['decision']} alloc={decision['final_allocation_pct']}, "
            f"Z={z}, ret5={event_regime_signals.get('event_return_5d')}"
        ),
        "status": "ok",
        "data_ok": payload.get("_data_ok", True) and data_provenance.get("quality") != "low",
        "quant_indicators": (
            [
                "event_return_5d",
                "trend_20d",
                "trend_60d",
                "vol_shift_20d_vs_60d",
                "corr_with_market_60d",
                "beta_to_market_60d",
                "downside_beta_60d",
                "drawdown_60d",
            ]
            if analysis_mode == "event_regime"
            else ["adf_pvalue", "z_score", "asset_cvar_99_daily"]
        ),
        "primary_decision": {
            "LONG": "bullish", "SHORT": "bearish",
            "HOLD": "hold", "CLEAR": "neutral",
        }.get(decision["decision"], "hold"),
        "recommendation": "allow" if decision["decision"] in ("LONG", "SHORT") else "allow_with_limits",
        "confidence": decision.get("confidence", 0.6 if payload.get("_data_ok") else 0.35),
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


def _parse_event_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass
    if len(text) >= 10:
        try:
            return datetime.fromisoformat(text[:10])
        except ValueError:
            return None
    return None


def _event_days_to(as_of: str, value: Any) -> int | None:
    base_dt = _parse_event_datetime(as_of)
    event_dt = _parse_event_datetime(value)
    if base_dt is None or event_dt is None:
        return None
    return int((event_dt.date() - base_dt.date()).days)


def _macro_trigger_breached(trigger: dict) -> bool:
    if not isinstance(trigger, dict):
        return False
    metric = str(trigger.get("metric", "")).strip().lower()
    value = trigger.get("current_value")
    try:
        current = float(value)
    except (TypeError, ValueError):
        return False

    if metric == "hy_oas":
        bp = current * 100.0 if abs(current) < 50 else current
        return bp >= 500
    if metric == "pmi":
        return current < 48
    if metric in {"inflation_proxy", "core_cpi_yoy", "cpi_yoy"}:
        return current > 3.0
    if metric == "vix_level":
        return current > 20
    if metric == "yield_curve_spread":
        return current < -0.20 or current > 0.30
    if metric == "sofr_ff_6m_basis_bp":
        return abs(current) > 10
    if metric in {"wti/brent", "wti", "brent"}:
        return current >= 80
    return False


def _macro_trigger_value_is_plausible(metric: str, value: Any) -> bool:
    try:
        current = float(value)
    except (TypeError, ValueError):
        return False
    metric = str(metric or "").strip().lower()
    if metric == "pmi":
        return 0.0 <= current <= 100.0
    if metric == "hy_oas":
        return 0.0 <= current <= 2000.0
    if metric in {"inflation_proxy", "core_cpi_yoy", "cpi_yoy"}:
        return -10.0 <= current <= 30.0
    if metric == "vix_level":
        return 0.0 <= current <= 150.0
    if metric == "yield_curve_spread":
        return -10.0 <= current <= 10.0
    if metric == "sofr_ff_6m_basis_bp":
        return -200.0 <= current <= 200.0
    if metric in {"wti/brent", "wti", "brent"}:
        return 0.0 <= current <= 500.0
    return abs(current) <= 10_000.0


def _event_priority_rank(item: dict) -> tuple[int, int, int, str, str]:
    priority = int(item.get("priority", 3) or 3)
    status = str(item.get("status", "")).strip().lower()
    status_rank = 0 if status in {"triggered", "imminent"} else (1 if status == "upcoming" else 2)
    days = item.get("days_to_event")
    days_rank = int(days) if isinstance(days, int) else 10_000
    return priority, status_rank, days_rank, str(item.get("desk", "")), str(item.get("ticker", ""))


def _event_key(item: dict) -> str:
    desk = str(item.get("desk", "")).strip().lower()
    ticker = str(item.get("ticker", "")).strip().upper()
    event_type = str(item.get("type", "")).strip().lower()
    subtype = str(item.get("subtype", "")).strip().lower()
    date = str(item.get("date", "")).strip()
    trigger = str(item.get("trigger", "")).strip().lower()
    return "::".join(part for part in [desk, ticker, event_type, subtype, date, trigger] if part)


def _build_event_calendar(state: InvestmentState) -> list[dict]:
    as_of = str(state.get("as_of", "")).strip()
    target = str(state.get("target_ticker", "")).strip().upper()
    universe = [
        str(t).strip().upper()
        for t in (state.get("universe", []) or [])
        if str(t).strip()
    ]
    events: list[dict] = []

    macro = state.get("macro_analysis", {}) if isinstance(state.get("macro_analysis"), dict) else {}
    for trigger in macro.get("monitoring_triggers", []) or []:
        if not isinstance(trigger, dict):
            continue
        metric = str(trigger.get("metric", "")).strip()
        current_value = trigger.get("current_value")
        if not _macro_trigger_value_is_plausible(metric, current_value):
            continue
        breached = _macro_trigger_breached(trigger)
        item = {
            "desk": "macro",
            "ticker": "__GLOBAL__",
            "type": "macro_monitor",
            "subtype": str(trigger.get("name", "")).strip() or metric,
            "metric": metric,
            "priority": int(trigger.get("priority", 3) or 3),
            "status": "triggered" if breached else "watch",
            "current_value": current_value,
            "trigger": str(trigger.get("trigger", "")).strip(),
            "action": str(trigger.get("action", "")).strip(),
            "source": "macro_monitoring_triggers",
            "confirmed": False,
            "affected_tickers": universe[:6],
            "breached": breached,
        }
        item["event_key"] = _event_key(item)
        events.append(item)
    for raw_event in macro.get("macro_event_calendar", []) or []:
        if not isinstance(raw_event, dict):
            continue
        days = raw_event.get("days_to_event")
        if not isinstance(days, int):
            days = _event_days_to(as_of, raw_event.get("date"))
        if days is not None and days < 0:
            continue
        status = str(raw_event.get("status", "")).strip().lower() or "upcoming"
        if days is not None and days <= 7 and days >= 0:
            status = "imminent"
        event = {
            "desk": "macro",
            "ticker": "__GLOBAL__",
            "type": str(raw_event.get("type", "")).strip().lower() or "macro_event",
            "subtype": str(raw_event.get("title", "")).strip() or str(raw_event.get("subtype", "")).strip(),
            "date": str(raw_event.get("date", "")).strip(),
            "days_to_event": days,
            "priority": 1 if status == "imminent" else 2,
            "status": status,
            "source": str(raw_event.get("source", "")).strip() or "macro_event_calendar",
            "confirmed": str(raw_event.get("source_classification", "")).strip().lower() == "confirmed",
            "source_classification": str(raw_event.get("source_classification", "")).strip().lower() or "confirmed",
            "event_origin": str(raw_event.get("event_origin", "")).strip() or "macro_event_calendar",
            "notes": str(raw_event.get("notes", "")).strip(),
        }
        event["event_key"] = _event_key(event)
        events.append(event)

    fundamental = state.get("fundamental_analysis", {}) if isinstance(state.get("fundamental_analysis"), dict) else {}
    fundamental_ticker = str(fundamental.get("ticker", "")).strip().upper() or target
    for item in fundamental.get("catalyst_calendar", []) or []:
        if not isinstance(item, dict):
            continue
        days = item.get("days_to_event")
        if not isinstance(days, int):
            days = _event_days_to(as_of, item.get("date"))
        if days is not None and days < 0:
            continue
        status = str(item.get("status", "")).strip().lower() or "upcoming"
        if days is not None and days <= 7:
            status = "imminent"
        event = {
            "desk": "fundamental",
            "ticker": fundamental_ticker,
            "type": str(item.get("type", "")).strip().lower() or "catalyst",
            "subtype": str(item.get("source_title", "")).strip() or str(item.get("type", "")).strip(),
            "date": str(item.get("date", "")).strip(),
            "days_to_event": days,
            "priority": 1 if str(item.get("importance", "")).strip().lower() == "high" else 2,
            "status": status,
            "source": str(item.get("source_title", "")).strip() or str(item.get("resolver_path", "")).strip() or "fundamental_catalyst",
            "confirmed": str(item.get("source_classification", "")).strip().lower() == "confirmed",
            "source_classification": str(item.get("source_classification", "")).strip().lower() or "inferred",
            "expected_scenario": str(item.get("expected_scenario", "")).strip(),
            "thesis_change_trigger": str(item.get("thesis_change_trigger", "")).strip(),
        }
        event["event_key"] = _event_key(event)
        events.append(event)

    sentiment = state.get("sentiment_analysis", {}) if isinstance(state.get("sentiment_analysis"), dict) else {}
    catalyst_risk = sentiment.get("catalyst_risk", {}) if isinstance(sentiment.get("catalyst_risk"), dict) else {}
    if catalyst_risk.get("catalyst_present") or str(sentiment.get("volatility_regime", "")).strip().lower() in {"high", "elevated"}:
        event = {
            "desk": "sentiment",
            "ticker": target or "__GLOBAL__",
            "type": "sentiment_regime",
            "subtype": "catalyst_risk" if catalyst_risk.get("catalyst_present") else "volatility_regime",
            "priority": 2,
            "status": "triggered" if catalyst_risk.get("catalyst_present") else "watch",
            "source": "sentiment_overlay",
            "confirmed": False,
            "catalyst_type": list(catalyst_risk.get("catalyst_type", []) or []),
            "volatility_regime": str(sentiment.get("volatility_regime", "")).strip(),
        }
        event["event_key"] = _event_key(event)
        events.append(event)

    quant = state.get("technical_analysis", {}) if isinstance(state.get("technical_analysis"), dict) else {}
    quant_payload = quant.get("quant_payload", {}) if isinstance(quant.get("quant_payload"), dict) else {}
    regime_probs = (quant_payload.get("market_regime_context", {}) or {}).get("state_probabilities", {}) or {}
    high_vol_prob = float(regime_probs.get("regime_2_high_vol", 0.0) or 0.0)
    asset_cvar = float(quant.get("asset_cvar_99_daily", 0.0) or 0.0)
    if high_vol_prob >= 0.20 or asset_cvar >= 0.05:
        event = {
            "desk": "quant",
            "ticker": target or "__GLOBAL__",
            "type": "quant_risk",
            "subtype": "high_vol_regime" if high_vol_prob >= 0.20 else "asset_cvar",
            "priority": 1,
            "status": "triggered",
            "source": "quant_payload",
            "confirmed": True,
            "high_vol_prob": round(high_vol_prob, 4),
            "asset_cvar_99_daily": round(asset_cvar, 6),
        }
        event["event_key"] = _event_key(event)
        events.append(event)

    construction = state.get("portfolio_construction_analysis", {}) if isinstance(state.get("portfolio_construction_analysis"), dict) else {}
    for trigger in construction.get("monitoring_triggers", []) or []:
        if not isinstance(trigger, dict):
            continue
        item = {
            "desk": "portfolio_construction",
            "ticker": target or "__GLOBAL__",
            "type": "construction_monitor",
            "subtype": str(trigger.get("name", "")).strip() or str(trigger.get("metric", "")).strip(),
            "metric": str(trigger.get("metric", "")).strip(),
            "priority": int(trigger.get("priority", 2) or 2),
            "status": "triggered",
            "current_value": trigger.get("current_value"),
            "trigger": str(trigger.get("trigger", "")).strip(),
            "action": str(trigger.get("action", "")).strip(),
            "source": "portfolio_construction_quant",
            "confirmed": True,
        }
        item["event_key"] = _event_key(item)
        events.append(item)

    deduped: dict[str, dict] = {}
    for item in events:
        key = str(item.get("event_key", "")).strip()
        if not key:
            continue
        prev = deduped.get(key)
        if prev is None or _event_priority_rank(item) < _event_priority_rank(prev):
            deduped[key] = item
    return sorted(deduped.values(), key=_event_priority_rank)[:16]


def _build_decision_quality_scorecard(state: InvestmentState) -> dict:
    desk_map = {
        "macro": state.get("macro_analysis", {}),
        "fundamental": state.get("fundamental_analysis", {}),
        "sentiment": state.get("sentiment_analysis", {}),
        "quant": state.get("technical_analysis", {}),
    }
    weights = {"macro": 0.30, "fundamental": 0.35, "sentiment": 0.15, "quant": 0.20}
    desks: dict[str, dict] = {}
    weak_desks: list[str] = []
    overall = 0.0
    for desk, output in desk_map.items():
        if not isinstance(output, dict) or not output:
            score = 0.15
            desks[desk] = {
                "status": "missing",
                "quality_score": score,
                "confidence": 0.0,
                "evidence_count": 0,
                "needs_more_data": True,
                "warning_count": 1,
            }
            weak_desks.append(desk)
            overall += score * weights[desk]
            continue
        evidence_count = len(output.get("evidence", []) or []) + len(output.get("evidence_digest", []) or [])
        warnings = len(((output.get("data_quality", {}) or {}).get("warnings", []) or []))
        confidence = float(output.get("confidence", 0.0) or 0.0)
        signal_strength = abs(float(output.get("signal_strength", 0.0) or 0.0))
        data_ok = bool(output.get("data_ok", output.get("status") == "ok"))
        needs_more_data = bool(output.get("needs_more_data", False))
        score = (
            (0.25 if data_ok else 0.05)
            + 0.20 * min(evidence_count / 5.0, 1.0)
            + 0.20 * min(confidence, 1.0)
            + 0.10 * min(signal_strength, 1.0)
            + (0.15 if not needs_more_data else 0.02)
            - 0.05 * min(warnings, 3)
        )
        score = max(0.0, min(round(score, 4), 1.0))
        desks[desk] = {
            "status": "ok" if data_ok else "weak",
            "quality_score": score,
            "confidence": round(confidence, 4),
            "evidence_count": evidence_count,
            "needs_more_data": needs_more_data,
            "warning_count": warnings,
        }
        if score < 0.50 or needs_more_data:
            weak_desks.append(desk)
        overall += score * weights[desk]

    return {
        "status": "ok",
        "overall_score": round(overall, 4),
        "desks": desks,
        "weak_desks": sorted(set(weak_desks)),
    }


def _monitoring_request_from_event(item: dict) -> dict | None:
    if not isinstance(item, dict):
        return None
    desk = str(item.get("desk", "")).strip().lower()
    ticker = str(item.get("ticker", "")).strip().upper()
    event_type = str(item.get("type", "")).strip().lower()
    subtype = str(item.get("subtype", "")).strip()
    priority = 1 if str(item.get("status", "")).strip().lower() in {"triggered", "imminent"} else 2
    if desk == "macro":
        return {
            "desk": "macro",
            "kind": "macro_headline_context",
            "ticker": "__GLOBAL__",
            "query": f"{subtype or event_type} macro update market impact",
            "priority": priority,
            "recency_days": 7,
            "max_items": 3,
            "rationale": f"monitoring_event:{event_type}",
            "source_tag": "monitoring",
            "impacted_desks": ["macro", "sentiment"],
        }
    if desk == "fundamental":
        kind = "press_release_or_ir"
        if event_type in {"legal_reg", "contract_renewal"}:
            kind = "sec_filing"
        return {
            "desk": "fundamental",
            "kind": kind,
            "ticker": ticker,
            "query": f"{ticker} {event_type or subtype} latest official update",
            "priority": priority,
            "recency_days": 14,
            "max_items": 3,
            "rationale": f"monitoring_event:{event_type}",
            "source_tag": "monitoring",
            "impacted_desks": ["fundamental", "sentiment"] if event_type in {"earnings", "street_rating"} else ["fundamental"],
        }
    if desk == "sentiment":
        return {
            "desk": "sentiment",
            "kind": "catalyst_event_detail",
            "ticker": ticker or "__GLOBAL__",
            "query": f"{ticker or 'market'} sentiment catalyst volatility positioning update",
            "priority": priority,
            "recency_days": 7,
            "max_items": 3,
            "rationale": f"monitoring_event:{event_type}",
            "source_tag": "monitoring",
            "impacted_desks": ["sentiment"],
        }
    return None


def _build_monitoring_actions(
    state: InvestmentState,
    event_calendar: list[dict],
    scorecard: dict,
) -> dict:
    prior_actions = state.get("monitoring_actions", {}) if isinstance(state.get("monitoring_actions"), dict) else {}
    handled = {
        str(key).strip()
        for key in (prior_actions.get("handled_event_keys", []) or [])
        if str(key).strip()
    }
    weak_desks = set(scorecard.get("weak_desks", []) or [])

    triggered_events = [
        item for item in event_calendar
        if isinstance(item, dict) and str(item.get("status", "")).strip().lower() in {"triggered", "imminent"}
    ]
    new_triggered = [item for item in triggered_events if str(item.get("event_key", "")).strip() not in handled]

    selected_desks: set[str] = set()
    risk_refresh_required = False
    for item in new_triggered:
        desk = str(item.get("desk", "")).strip().lower()
        event_type = str(item.get("type", "")).strip().lower()
        if desk in {"macro", "fundamental", "sentiment", "quant"}:
            selected_desks.add(desk)
        if desk == "portfolio_construction":
            risk_refresh_required = True
            selected_desks.add("quant")
        if desk == "fundamental" and event_type in {"earnings", "street_rating"}:
            selected_desks.add("sentiment")
        if desk in {"macro", "quant"}:
            risk_refresh_required = True

    if new_triggered and weak_desks:
        selected_desks.update(weak_desks)

    monitoring_requests = []
    for item in new_triggered:
        req = _monitoring_request_from_event(item)
        if req:
            sanitized = _sanitize_swarm_request(req, default_source_tag="monitoring")
            if sanitized:
                monitoring_requests.append(sanitized)
    monitoring_requests = _dedupe_requests(monitoring_requests)

    task_actions = []
    for desk in sorted(selected_desks):
        task_actions.append({
            "type": "rerun_desk",
            "detail": f"{desk} desk rerun for monitoring escalation",
            "params": {"desk": desk},
        })
    if risk_refresh_required:
        task_actions.append({
            "type": "rerun_risk",
            "detail": "risk refresh for monitoring escalation",
            "params": {"source": "monitoring_router"},
        })

    handled_event_keys = sorted(handled | {str(item.get("event_key", "")).strip() for item in new_triggered if str(item.get("event_key", "")).strip()})
    return {
        "status": "ok",
        "triggered_events": triggered_events[:8],
        "new_triggered_events": new_triggered[:8],
        "selected_desks": sorted(selected_desks),
        "risk_refresh_required": risk_refresh_required,
        "monitoring_requests": monitoring_requests[:MAX_WEB_QUERIES_PER_RUN],
        "force_research": bool(monitoring_requests or selected_desks),
        "reason": "new_triggered_events" if new_triggered else ("weak_desk_monitoring" if weak_desks else "no_escalation"),
        "handled_event_keys": handled_event_keys[-64:],
        "weak_desks": sorted(weak_desks),
        "task_actions": task_actions,
    }


def monitoring_router_node(state: InvestmentState) -> dict:
    _log(state, "monitoring_router", "enter")
    event_calendar = _build_event_calendar(state)
    scorecard = _build_decision_quality_scorecard(state)
    actions = _build_monitoring_actions(state, event_calendar, scorecard)
    backlog = _merge_actions(state.get("task_backlog", []), actions.get("task_actions", []))

    print(f"\n🕒 MONITORING ROUTER  (iter #{state.get('iteration_count', 1)})")
    print(
        "   Events: "
        f"total={len(event_calendar)} triggered={len(actions.get('triggered_events', []))} "
        f"new={len(actions.get('new_triggered_events', []))}"
    )
    print(
        "   Escalation: "
        f"desks={actions.get('selected_desks', []) if actions.get('selected_desks') else '[]'} "
        f"risk_refresh={actions.get('risk_refresh_required', False)} "
        f"force_research={actions.get('force_research', False)}"
    )
    _ops_log(
        state,
        "monitoring_router",
        "monitoring escalation",
        [
            f"events_total={len(event_calendar)} triggered={len(actions.get('triggered_events', []))} new={len(actions.get('new_triggered_events', []))}",
            f"selected_desks={actions.get('selected_desks', [])}",
            f"risk_refresh_required={actions.get('risk_refresh_required', False)}",
            f"decision_quality_overall={scorecard.get('overall_score')}",
        ],
    )
    out = {
        "event_calendar": event_calendar,
        "decision_quality_scorecard": scorecard,
        "monitoring_actions": actions,
        "_monitoring_forced_desks": list(actions.get("selected_desks", [])),
        "task_backlog": backlog,
        "trace": [{
            "node": "monitoring_router",
            "events_total": len(event_calendar),
            "triggered_events": len(actions.get("triggered_events", [])),
            "new_triggered_events": len(actions.get("new_triggered_events", [])),
            "selected_desks": list(actions.get("selected_desks", [])),
            "risk_refresh_required": bool(actions.get("risk_refresh_required", False)),
            "quality_overall": scorecard.get("overall_score"),
        }],
    }
    _log(state, "monitoring_router", "exit", {"selected_desks": actions.get("selected_desks", []), "force_research": actions.get("force_research", False)})
    return out


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
    monitoring_actions = state.get("monitoring_actions", {}) if isinstance(state.get("monitoring_actions"), dict) else {}
    monitoring_requests = []
    for req in (monitoring_actions.get("monitoring_requests", []) or []):
        sreq = _sanitize_swarm_request(req, default_source_tag="monitoring")
        if sreq:
            monitoring_requests.append(sreq)
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
        recovery_requests=list(recovery.get("evidence_requests", [])) + monitoring_requests,
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
    allowed_requests = list(decision.get("allowed_requests", []) or [])

    if monitoring_actions.get("force_research"):
        run_research = True
        decision_reason = f"monitoring_escalation:{monitoring_actions.get('reason', 'event_trigger')}"
        if monitoring_requests:
            allowed_requests = _dedupe_requests(allowed_requests + monitoring_requests)[:MAX_WEB_QUERIES_PER_RUN]

    allowed_count = len(allowed_requests)

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
            f"monitoring_force_research={monitoring_actions.get('force_research', False)} monitoring_desks={monitoring_actions.get('selected_desks', [])}",
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
        "_research_plan": allowed_requests,
        "_monitoring_forced_desks": list(monitoring_actions.get("selected_desks", [])),
        "research_stop_reason": "" if run_research else decision.get("reason", "no_trigger"),
        "user_action_required": user_action_required,
        "user_action_items": user_action_items,
        "trace": [{
            "node": "research_router",
            "run": run_research,
            "reason": decision_reason,
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
            "monitoring_force_research": bool(monitoring_actions.get("force_research", False)),
            "monitoring_selected_desks": list(monitoring_actions.get("selected_desks", [])),
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
    tavily: TavilySearchProvider | None = None,
    exa: ExaSearchProvider | None = None,
    perplexity: PerplexitySearchProvider | None = None,
    as_of: str = "",
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
        if tavily is not None:
            items = tavily.collect_evidence(
                kind=kind, ticker=ticker, query=query, recency_days=recency_days, max_items=max_items,
                allowlist=allowlist, desk=desk, resolver_path="tavily_fallback_ownership",
            )
            if items:
                return items, "tavily_fallback_ownership"
        if exa is not None:
            items = exa.collect_evidence(
                kind=kind, ticker=ticker, query=query, recency_days=recency_days, max_items=max_items,
                allowlist=allowlist, desk=desk, resolver_path="exa_fallback_ownership",
            )
            if items:
                return items, "exa_fallback_ownership"
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
        if tavily is not None:
            items = tavily.collect_evidence(
                kind=kind, ticker=ticker, query=query, recency_days=recency_days, max_items=max_items,
                allowlist=allowlist, desk=desk, resolver_path="tavily_fallback_ir",
            )
            if items:
                return items, "tavily_fallback_ir"
        if exa is not None:
            items = exa.collect_evidence(
                kind=kind, ticker=ticker, query=query, recency_days=recency_days, max_items=max_items,
                allowlist=allowlist, desk=desk, resolver_path="exa_fallback_ir",
            )
            if items:
                return items, "exa_fallback_ir"
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
        if tavily is not None:
            items = tavily.collect_evidence(
                kind=kind, ticker=ticker, query=query, recency_days=recency_days, max_items=max_items,
                allowlist=allowlist, desk=desk, resolver_path="tavily_fallback_macro",
            )
            if items:
                return items, "tavily_fallback_macro"
        if exa is not None:
            items = exa.collect_evidence(
                kind=kind, ticker=ticker, query=query, recency_days=recency_days, max_items=max_items,
                allowlist=allowlist, desk=desk, resolver_path="exa_fallback_macro",
            )
            if items:
                return items, "exa_fallback_macro"
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
    if tavily is not None:
        items = tavily.collect_evidence(
            kind=kind, ticker=ticker, query=query, recency_days=recency_days, max_items=max_items,
            allowlist=allowlist, desk=desk, resolver_path="tavily_fallback_default",
        )
        if items:
            return items, "tavily_fallback_default"
    if exa is not None:
        items = exa.collect_evidence(
            kind=kind, ticker=ticker, query=query, recency_days=recency_days, max_items=max_items,
            allowlist=allowlist, desk=desk, resolver_path="exa_fallback_default",
        )
        if items:
            return items, "exa_fallback_default"
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
        forced_desks = sorted({
            str(desk).strip().lower()
            for desk in (state.get("_monitoring_forced_desks", []) or [])
            if str(desk).strip().lower() in {"macro", "fundamental", "sentiment", "quant"}
        })
        ct = dict(state.get("completed_tasks", {}))
        for desk in forced_desks:
            ct[desk] = False
        _log(state, "research_executor", "exit", {"queries_executed": 0, "delta": 0})
        return {
            "last_research_delta": 0,
            "_run_research": False,
            "_executed_requests": [],
            "_rerun_plan": {
                "selected_desks": forced_desks,
                "k": _MAX_RERUN_DESKS,
                "reasons": {desk: ["monitoring_escalation"] for desk in forced_desks},
                "executed_kinds": [],
            },
            "_evidence_delta_kinds": {},
            "completed_tasks": ct,
            "trace": [{
                "node": "research_executor",
                "research_round": int(state.get("research_round", 0)),
                "queries_executed": 0,
                "last_research_delta": 0,
                "evidence_score": int(state.get("evidence_score", 0)),
                "rerun_desks": forced_desks,
                "forced_rerun_desks": forced_desks,
            }],
        }

    as_of = state.get("as_of", "")
    mode = state.get("mode", "mock")
    store, existing_canonical = _reindex_store_by_canonical(dict(state.get("evidence_store", {})))
    before = set(store.keys())
    before_buckets = _covered_buckets_from_store(store)

    sec = SECEdgarProvider()
    web = WebResearchProvider(mode=mode)
    tavily = TavilySearchProvider(mode=mode)
    exa = ExaSearchProvider(mode=mode)
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
            tavily=tavily,
            exa=exa,
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
    forced_desks = {
        str(desk).strip().lower()
        for desk in (state.get("_monitoring_forced_desks", []) or [])
        if str(desk).strip().lower() in {"macro", "fundamental", "sentiment", "quant"}
    }
    if forced_desks:
        rerun_desks.update(forced_desks)
        reasons = dict(rerun_plan.get("reasons", {}) or {})
        for desk in sorted(forced_desks):
            reason_list = list(reasons.get(desk, []) or [])
            if "monitoring_escalation" not in reason_list:
                reason_list.append("monitoring_escalation")
            reasons[desk] = reason_list
        rerun_plan["reasons"] = reasons
    rerun_plan["selected_desks"] = sorted(rerun_desks)

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
            "forced_rerun_desks": sorted(forced_desks),
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


def post_desk_router(state: InvestmentState) -> Literal["hedge_lite_builder", "risk_manager"]:
    intent = _frontdoor_intent(state)
    if intent == "position_review":
        print("   [ROUTER] barrier -> risk_manager (frontdoor_intent=position_review)")
        _graph_trace("route barrier -> risk_manager (frontdoor_intent=position_review)")
        return "risk_manager"
    print(f"   [ROUTER] barrier -> hedge_lite_builder (frontdoor_intent={intent or 'n/a'})")
    _graph_trace(f"route barrier -> hedge_lite_builder (frontdoor_intent={intent or 'n/a'})")
    return "hedge_lite_builder"


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

def build_general_investment_graph() -> StateGraph:
    """
    V4 Graph:
      START → question_understanding → orchestrator → [4 desks parallel] → barrier → hedge_lite_builder
            → portfolio_construction_quant → monitoring_router → research_router
        research_router -> risk_manager
        research_router -> research_executor -> [4 desks(parallel,research)] -> research_barrier
                         -> hedge_lite_builder -> portfolio_construction_quant -> monitoring_router -> research_router
      risk_manager -> router -> orchestrator/report_writer
    """
    g = StateGraph(InvestmentState)

    g.add_node("question_understanding", question_understanding_node)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("macro_analyst", macro_analyst_node)
    g.add_node("fundamental_analyst", fundamental_analyst_node)
    g.add_node("sentiment_analyst", sentiment_analyst_node)
    g.add_node("quant_analyst", quant_analyst_node)
    g.add_node("barrier", barrier_node)
    g.add_node("hedge_lite_builder", hedge_lite_builder_node)
    g.add_node("portfolio_construction_quant", portfolio_construction_quant_node)
    g.add_node("monitoring_router", monitoring_router_node)
    g.add_node("research_router", research_router_node)
    g.add_node("research_executor", research_executor_node)
    g.add_node("research_barrier", research_barrier_node)
    g.add_node("macro_analyst_research", macro_analyst_research_node)
    g.add_node("fundamental_analyst_research", fundamental_analyst_research_node)
    g.add_node("sentiment_analyst_research", sentiment_analyst_research_node)
    g.add_node("quant_analyst_research", quant_analyst_research_node)
    g.add_node("risk_manager", risk_manager_node)
    g.add_node("report_writer", report_writer_node)

    g.add_edge(START, "question_understanding")
    g.add_edge("question_understanding", "orchestrator")

    # Fan-out: orchestrator → 4 desks
    for desk in ["macro_analyst", "fundamental_analyst", "sentiment_analyst", "quant_analyst"]:
        g.add_edge("orchestrator", desk)

    # Fan-in: 4 desks → barrier
    for desk in ["macro_analyst", "fundamental_analyst", "sentiment_analyst", "quant_analyst"]:
        g.add_edge(desk, "barrier")

    g.add_conditional_edges("barrier", post_desk_router, {
        "hedge_lite_builder": "hedge_lite_builder",
        "risk_manager": "risk_manager",
    })
    g.add_edge("hedge_lite_builder", "portfolio_construction_quant")
    g.add_edge("portfolio_construction_quant", "monitoring_router")
    g.add_edge("monitoring_router", "research_router")

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


def build_position_review_graph() -> StateGraph:
    """
    Dedicated position review workflow:
      START → question_understanding → orchestrator → [4 desks parallel] → barrier
            → risk_manager → router → orchestrator/report_writer
    """
    g = StateGraph(InvestmentState)

    g.add_node("question_understanding", question_understanding_node)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("macro_analyst", macro_analyst_node)
    g.add_node("fundamental_analyst", fundamental_analyst_node)
    g.add_node("sentiment_analyst", sentiment_analyst_node)
    g.add_node("quant_analyst", quant_analyst_node)
    g.add_node("barrier", barrier_node)
    g.add_node("risk_manager", risk_manager_node)
    g.add_node("report_writer", report_writer_node)

    g.add_edge(START, "question_understanding")
    g.add_edge("question_understanding", "orchestrator")

    for desk in ("macro_analyst", "fundamental_analyst", "sentiment_analyst", "quant_analyst"):
        g.add_edge("orchestrator", desk)
        g.add_edge(desk, "barrier")

    g.add_edge("barrier", "risk_manager")
    g.add_conditional_edges("risk_manager", risk_router, {
        "orchestrator": "orchestrator",
        "report_writer": "report_writer",
    })
    g.add_edge("report_writer", END)
    return g


def build_investment_graph(frontdoor_intent: str | None = None) -> StateGraph:
    intent = str(frontdoor_intent or "").strip()
    if intent == "position_review":
        return build_position_review_graph()
    return build_general_investment_graph()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main(
    mode: str = "mock",
    seed: int | None = 42,
    user_request: str | None = None,
    portfolio_context: dict | None = None,
) -> dict:
    print("🚀 7-Agent AI Investment Team V3 (Final Integration)")
    print("=" * 60)
    print(f"   Mode: {mode} | Seed: {seed}")

    state = create_initial_state(
        user_request=user_request or "애플(AAPL) 주식을 지금 매수해도 괜찮을까요? 6개월 투자 관점에서 분석해 주세요.",
        mode=mode,
        portfolio_context=portfolio_context or {},
        seed=seed,
    )
    run_id = state["run_id"]
    print(f"   Run ID: {run_id}")

    run_dir = telemetry.init_run(run_id, mode)
    print(f"   Run Dir: {run_dir}")

    preview = preview_launch_requirements(
        str(state.get("user_request", "") or ""),
        portfolio_context=state.get("portfolio_context", {}) if isinstance(state.get("portfolio_context"), dict) else {},
        mode=mode,
        seed=seed,
    )
    preview = _prompt_position_review_inputs_if_tty(state, preview)
    state["question_understanding"] = dict(preview.get("question_understanding", {}) or {})
    state["portfolio_intake"] = dict(preview.get("portfolio_intake", {}) or {})
    if preview.get("target_ticker"):
        state["target_ticker"] = str(preview.get("target_ticker", "")).strip().upper()
    if preview.get("universe"):
        state["universe"] = [str(t).strip().upper() for t in (preview.get("universe") or []) if str(t).strip()]
    state["_frontdoor_prepared"] = True
    frontdoor_bundle = _build_frontdoor_bundle(state, compute_normalized=True)
    state.update(frontdoor_bundle)
    frontdoor_intent = str(((frontdoor_bundle.get("question_understanding", {}) or {}).get("intent", ""))).strip()
    state["workflow_kind"] = "position_review" if frontdoor_intent == "position_review" else "general"

    graph = build_investment_graph(frontdoor_intent=frontdoor_intent)
    app = graph.compile()
    final_state = app.invoke(state)

    telemetry.save_final_state(run_id, final_state)
    operator_summary_path = _write_operator_summary(run_id)
    dashboard_path = write_run_dashboard(run_id)

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
    print(f"   Dashboard:    {dashboard_path}")

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
    parser.add_argument("--portfolio-context-json", type=str, default=None, help="추가 포지션/북 컨텍스트 JSON")
    args = parser.parse_args()
    portfolio_context = None
    if args.portfolio_context_json:
        portfolio_context = json.loads(args.portfolio_context_json)
    main(mode=args.mode, seed=args.seed, user_request=args.question, portfolio_context=portfolio_context)
