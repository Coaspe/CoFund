"""
orchestrator_agent.py — ① Orchestrator Agent (CIO/PM)
=====================================================
총괄 PM(Portfolio Manager) 겸 최고 투자 책임자(CIO) 역할.

핵심 기능:
  A. 초기 위임    : 사용자 요청 → 미니 IPS 수립 → 4개 데스크 태스크 할당
  B. 피드백 대응  : Risk Manager 반려 → Scale / Hedge / Pivot 결정
  C. 무한 루프 방지: iteration_count ≥ MAX → Fallback(현금 관망) 강제 종료

의존 패키지:
  pip install numpy langchain-openai langgraph pydantic

실행:
  python orchestrator_agent.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

warnings.filterwarnings("ignore")

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = object  # type: ignore

try:
    from langchain_core.messages import SystemMessage, HumanMessage
    HAS_LC = True
except ImportError:
    HAS_LC = False

from schemas.common import InvestmentState
from llm.router import get_llm_with_cache, set_cache, force_real_llm_in_tests


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 상수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MAX_ITERATIONS = 3

# 티커 추출용 간이 정규식 (NER 대용)
_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")

# 자주 언급되는 한국어→영어 티커 매핑
_KR_TICKER_MAP = {
    "애플": "AAPL", "아마존": "AMZN", "구글": "GOOGL", "알파벳": "GOOGL",
    "마이크로소프트": "MSFT", "테슬라": "TSLA", "엔비디아": "NVDA",
    "메타": "META", "넷플릭스": "NFLX",
}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _orch_trace_enabled() -> bool:
    return _env_flag("ORCH_TRACE", False)


def _orch_trace(message: str) -> None:
    if _orch_trace_enabled():
        print(f"   [ORCH TRACE] {message}", flush=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pydantic 출력 스키마
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if HAS_PYDANTIC:
    class DeskTask(BaseModel):
        """개별 데스크에 내리는 분석 지시."""
        horizon_days: int = Field(ge=1, description="분석 시간 범위 (일)")
        risk_budget: Optional[str] = Field(
            default=None,
            description="리스크 예산 수준 (Conservative / Moderate / Aggressive)"
        )
        focus_areas: List[str] = Field(
            default_factory=list,
            description="중점 확인 사항 목록"
        )

    class InvestmentBrief(BaseModel):
        """CIO의 투자 판단 근거."""
        rationale: str = Field(description="CIO의 판단 근거")
        target_universe: List[str] = Field(
            default_factory=list,
            description="분석 대상 종목/ETF 유니버스"
        )

    class OrchestratorOutput(BaseModel):
        """① Orchestrator LLM 구조화 출력."""
        current_iteration: int = Field(ge=0, description="현재 반복 횟수")
        action_type: Literal[
            "initial_delegation",
            "scale_down",
            "add_hedge",
            "pivot_strategy",
            "fallback_abort",
        ] = Field(description="Orchestrator 액션 유형")
        investment_brief: InvestmentBrief
        desk_tasks: Dict[str, DeskTask] = Field(
            description="macro / fundamental / sentiment / quant 데스크별 지시"
        )
else:
    OrchestratorOutput = dict  # type: ignore[assignment,misc]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 시스템 프롬프트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ORCHESTRATOR_SYSTEM_PROMPT = """\
당신은 헤지펀드의 총괄 PM이자 CIO입니다.
당신의 역할은 투자 목표를 세우고 4개의 전문 데스크(Macro, Funda, Senti, Quant)에 업무를 지시하며,
리스크 위원회의 피드백을 수용해 전략을 수정하는 것입니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A. 초기 지시 모드 (iteration_count == 0)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
사용자의 요청을 분석하여 아래 사항을 포함한 미니 IPS(Investment Policy Statement)를 수립하라:
- 투자 목적과 판단 근거 (rationale)
- 분석 대상 종목/ETF 유니버스 (target_universe)
- Portfolio Context가 제공되면 허용/금지 티커, 벤치마크, 리스크 예산, 리밸런싱 주기 같은 mandate를 우선 반영하라.
각 데스크가 수행할 분석의 파라미터를 명시한 작업 지시서(Task Payload)를 작성하라:
- horizon_days: 투자기간
- risk_budget: 리스크 예산 수준 (Conservative / Moderate / Aggressive) — Quant 데스크에 적용
- focus_areas: 중점 확인 사항

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
B. 리스크 피드백 대응 모드 (POST_RISK_REJECT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Risk Manager가 반려(Reject) 피드백을 보냈을 경우, 동일한 전략을 반복하지 말고
아래 의사결정 트리에 따라 조치하라:
- Scaling (비중 축소): 위반 강도가 낮고 투자 아이디어가 여전히 유효할 때. (예: CVaR 한도 경미 초과)
- Hedge (헷지 추가): 팩터/베타 노출이 과도하거나 상관관계가 높을 때. (예: 숏 헷지 종목 탐색 지시)
- Pivot (테마 변경): 매크로나 펀더멘털 상 구조적 악재가 있어 기존 논리가 무효화됐을 때. (예: 다른 섹터로 변경)

iteration 기반 에스컬레이션 규칙:
- iteration_count == 1  →  Scale 또는 Hedge를 지시하라.
- iteration_count == 2  →  Pivot을 지시하라.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
C. 무한 루프 방지 — Fallback 모드
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
iteration_count가 최대치(3) 이상에 도달하여 리스크 승인을 받지 못했다면 Fallback 모드로 전환하라:
- action_type을 "fallback_abort"로 설정
- 신규 매수를 포기하거나 최소 비중만 유지
- target_universe에 현금(CASH) 또는 방어 섹터 ETF(SHY, TLT, XLU 등)를 포함해 보수적 대안을 제시
- rationale에 반복 실패 사유를 명시하고 분석을 강제 종료

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
출력 형식
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
출력은 반드시 아래 JSON 스키마를 따르라:
{
  "current_iteration": <int>,
  "action_type": "initial_delegation | scale_down | add_hedge | pivot_strategy | fallback_abort",
  "investment_brief": {
    "rationale": "<CIO의 판단 근거>",
    "target_universe": ["<TICKER>", ...]
  },
  "desk_tasks": {
    "macro":       { "horizon_days": <int>, "focus_areas": ["..."] },
    "fundamental": { "horizon_days": <int>, "focus_areas": ["..."] },
    "sentiment":   { "horizon_days": <int>, "focus_areas": ["..."] },
    "quant":       { "horizon_days": <int>, "risk_budget": "<Conservative|Moderate|Aggressive>", "focus_areas": ["..."] }
  }
}"""


def _build_orchestrator_human_msg(
    user_request: str,
    iteration: int,
    risk_feedback: Optional[dict] = None,
    portfolio_context: Optional[dict] = None,
    book_context: Optional[dict] = None,
) -> str:
    """LLM에게 전달할 Human 메시지 조립."""
    parts = [
        f"[현재 시각] {datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"[iteration_count] {iteration}",
        f"[사용자 원본 요청] {user_request}",
    ]
    if risk_feedback:
        parts.append(
            f"[Risk Manager 피드백]\n```json\n"
            f"{json.dumps(risk_feedback, ensure_ascii=False, indent=2)}\n```"
        )
    if portfolio_context:
        parts.append(
            f"[Portfolio Context]\n```json\n"
            f"{json.dumps(portfolio_context, ensure_ascii=False, indent=2)}\n```"
        )
    if book_context:
        parts.append(
            f"[Book Context]\n```json\n"
            f"{json.dumps(book_context, ensure_ascii=False, indent=2)}\n```"
        )
    return "\n".join(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Intent 분류 (규칙 기반 fallback)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_INTENT_INTENTS = ["single_ticker_entry", "overheated_check", "compare_tickers",
                   "market_outlook", "event_risk"]


def _default_desk_tasks(
    horizon_days: int = 30,
    risk_budget: str = "Moderate",
    focus: str = "",
) -> dict:
    """Default desk_tasks for fallback / mock."""
    return {
        "macro":       {"horizon_days": min(30, horizon_days), "focus_areas": ["레짐 확인", "리스크온/오프"]},
        "fundamental": {"horizon_days": max(30, horizon_days),
                        "focus_areas": ["밸류에이션" if focus == "valuation" else "구조적 리스크", "FCF 품질"]},
        "sentiment":   {"horizon_days": min(14, horizon_days),
                        "focus_areas": ["이벤트 리스크" if focus == "event" else "포지셔닝", "뉴스 볼륨"]},
        "quant":       {"horizon_days": horizon_days, "risk_budget": risk_budget,
                        "focus_areas": ["CVaR 최적화", "Z-score"]},
    }


def _normalize_ticker_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = re.split(r"[\s,]+", value)
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        t = str(item or "").strip().upper()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _parse_iso_date(value: Any) -> Optional[datetime]:
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


def _days_to_event(as_of: str, event_date: Any) -> Optional[int]:
    base_dt = _parse_iso_date(as_of)
    event_dt = _parse_iso_date(event_date)
    if base_dt is None or event_dt is None:
        return None
    return int((event_dt.date() - base_dt.date()).days)


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_risk_budget(value: Any) -> Optional[str]:
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    if raw.startswith("cons"):
        return "Conservative"
    if raw.startswith("agg"):
        return "Aggressive"
    if raw.startswith("mod"):
        return "Moderate"
    return None


def _normalize_portfolio_context(portfolio_context: Optional[dict]) -> dict:
    if not isinstance(portfolio_context, dict):
        return {}
    benchmark = str(
        portfolio_context.get("benchmark")
        or portfolio_context.get("benchmark_ticker")
        or ""
    ).strip().upper()
    return {
        "allowed_tickers": _normalize_ticker_list(
            portfolio_context.get("allowed_tickers")
            or portfolio_context.get("allowed_universe")
        ),
        "blocked_tickers": _normalize_ticker_list(
            portfolio_context.get("blocked_tickers")
            or portfolio_context.get("forbidden_tickers")
        ),
        "required_tickers": _normalize_ticker_list(
            portfolio_context.get("required_tickers")
            or portfolio_context.get("must_include")
        ),
        "preferred_hedges": _normalize_ticker_list(
            portfolio_context.get("preferred_hedges")
            or portfolio_context.get("hedge_candidates")
        ),
        "benchmark": benchmark,
        "max_universe_size": _coerce_int(portfolio_context.get("max_universe_size")),
        "quant_risk_budget": _normalize_risk_budget(
            portfolio_context.get("quant_risk_budget")
            or portfolio_context.get("risk_budget_override")
        ),
        "review_frequency": str(
            portfolio_context.get("rebalance_frequency")
            or portfolio_context.get("review_frequency")
            or ""
        ).strip().lower(),
        "max_single_name_weight": _coerce_float(portfolio_context.get("max_single_name_weight")),
        "max_drawdown_pct": _coerce_float(portfolio_context.get("max_drawdown_pct")),
        "target_gross_exposure": _coerce_float(portfolio_context.get("target_gross_exposure")),
        "target_net_exposure": _coerce_float(portfolio_context.get("target_net_exposure")),
    }


def _append_focus_area(task: dict, item: str) -> dict:
    out = dict(task or {})
    focus_areas = list(out.get("focus_areas", []) or [])
    if item and item not in focus_areas:
        focus_areas.append(item)
    out["focus_areas"] = focus_areas
    return out


def _trim_universe(universe: list[str], max_size: Optional[int], required: list[str]) -> list[str]:
    if max_size is None or max_size <= 0 or len(universe) <= max_size:
        return universe
    required_set = set(required)
    ordered = list(universe[:1])
    ordered.extend(t for t in universe[1:] if t in required_set)
    ordered.extend(t for t in universe[1:] if t not in required_set)
    out: list[str] = []
    seen: set[str] = set()
    for ticker in ordered:
        if ticker in seen:
            continue
        seen.add(ticker)
        out.append(ticker)
        if len(out) >= max_size:
            break
    return out


def _build_monitoring_plan(mandate: dict) -> dict:
    review_triggers: list[str] = []
    benchmark = mandate.get("benchmark")
    if benchmark:
        review_triggers.append(f"벤치마크({benchmark}) 대비 상대 성과/베타 이탈 시 재점검")
    if mandate.get("target_gross_exposure") is not None:
        review_triggers.append(
            f"총 그로스 노출이 {float(mandate['target_gross_exposure']):.0%} 수준에 근접하면 재점검"
        )
    if mandate.get("target_net_exposure") is not None:
        review_triggers.append(
            f"순 노출이 {float(mandate['target_net_exposure']):.0%} 수준을 벗어나면 재점검"
        )
    if mandate.get("max_single_name_weight") is not None:
        review_triggers.append(
            f"단일 아이디어 비중이 {float(mandate['max_single_name_weight']):.0%} 한도에 접근하면 재점검"
        )
    if mandate.get("max_drawdown_pct") is not None:
        review_triggers.append(
            f"누적 손실이 {float(mandate['max_drawdown_pct']):.0%} 수준에 근접하면 리밸런싱 검토"
        )
    if mandate.get("preferred_hedges"):
        review_triggers.append(
            f"지정 헤지 후보({', '.join(mandate['preferred_hedges'][:3])})의 상관/유동성 훼손 시 재검토"
        )
    return {
        "review_frequency": mandate.get("review_frequency") or "event-driven",
        "review_triggers": review_triggers,
    }


def _normalize_weight_map(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for ticker, weight in raw.items():
        symbol = str(ticker or "").strip().upper()
        if not symbol:
            continue
        try:
            out[symbol] = float(weight)
        except (TypeError, ValueError):
            continue
    return out


def _ordered_union(*groups: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group or []:
            symbol = str(item or "").strip().upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            out.append(symbol)
    return out


def _build_book_context_summary(state: InvestmentState) -> dict:
    current_positions = _normalize_weight_map(
        state.get("positions_final") or state.get("positions_proposed") or {}
    )
    proposed_positions = _normalize_weight_map(state.get("positions_proposed") or {})
    prior_active = state.get("active_ideas", {}) if isinstance(state.get("active_ideas"), dict) else {}
    raw_backlog = state.get("monitoring_backlog", []) if isinstance(state.get("monitoring_backlog"), list) else []
    open_backlog = [
        item for item in raw_backlog
        if isinstance(item, dict) and str(item.get("status", "open")).strip().lower() not in {"closed", "done", "archived"}
    ]
    active_position_weights = {
        ticker: round(weight, 6)
        for ticker, weight in current_positions.items()
        if abs(weight) > 1e-6
    }
    proposed_nonzero = {
        ticker: round(weight, 6)
        for ticker, weight in proposed_positions.items()
        if abs(weight) > 1e-6
    }
    tracked_ideas = _ordered_union(active_position_weights.keys(), prior_active.keys(), proposed_nonzero.keys())
    summary = {
        "current_positions": active_position_weights,
        "proposed_positions": proposed_nonzero,
        "tracked_ideas": tracked_ideas[:8],
        "open_review_count": len(open_backlog),
    }
    event_calendar = state.get("event_calendar", []) if isinstance(state.get("event_calendar"), list) else []
    if event_calendar:
        summary["next_events_preview"] = [
            {
                "ticker": str(item.get("ticker", "")).strip().upper(),
                "type": str(item.get("type", "")).strip(),
                "status": str(item.get("status", "")).strip(),
                "days_to_event": item.get("days_to_event"),
            }
            for item in event_calendar[:5]
            if isinstance(item, dict)
        ]
    quality = state.get("decision_quality_scorecard", {}) if isinstance(state.get("decision_quality_scorecard"), dict) else {}
    if quality:
        summary["decision_quality_overall"] = quality.get("overall_score")
        summary["weak_desks"] = list(quality.get("weak_desks", []) or [])
    benchmark = str(
        ((state.get("portfolio_context", {}) or {}).get("benchmark"))
        or ((state.get("portfolio_context", {}) or {}).get("benchmark_ticker"))
        or ""
    ).strip().upper()
    if benchmark:
        summary["benchmark"] = benchmark
    if open_backlog:
        preview = []
        for item in open_backlog[:5]:
            preview.append({
                "ticker": str(item.get("ticker", "")).strip().upper(),
                "trigger": str(item.get("trigger", "")).strip(),
                "priority": int(item.get("priority", 3) or 3),
            })
        summary["review_backlog_preview"] = preview
    return summary


def _derive_idea_status(ticker: str, universe: list[str], current_weight: float, previous_status: str) -> str:
    if abs(current_weight) > 1e-6:
        return "active_position"
    if ticker in universe:
        return "candidate"
    if previous_status in {"active_position", "candidate", "monitor", "watchlist"}:
        return "monitor"
    return "inactive"


def _build_active_ideas_registry(state: InvestmentState, decision: dict) -> dict[str, dict]:
    brief = decision.get("investment_brief", {}) if isinstance(decision.get("investment_brief"), dict) else {}
    universe = _normalize_ticker_list(brief.get("target_universe", []))
    current_positions = _normalize_weight_map(
        state.get("positions_final") or state.get("positions_proposed") or {}
    )
    proposed_positions = _normalize_weight_map(state.get("positions_proposed") or {})
    prior_registry = state.get("active_ideas", {}) if isinstance(state.get("active_ideas"), dict) else {}
    monitoring_plan = decision.get("monitoring_plan", {}) if isinstance(decision.get("monitoring_plan"), dict) else {}
    mandate = decision.get("portfolio_mandate", {}) if isinstance(decision.get("portfolio_mandate"), dict) else {}
    action_type = str(decision.get("action_type", "")).strip() or "initial_delegation"
    intent = str(decision.get("intent", "")).strip() or str(state.get("intent", "")).strip()
    as_of = str(state.get("as_of", "")).strip()
    run_id = str(state.get("run_id", "")).strip()
    benchmark = str(mandate.get("benchmark", "")).strip().upper()
    review_frequency = str(monitoring_plan.get("review_frequency", "")).strip() or "event-driven"
    review_triggers = [
        str(item).strip()
        for item in (monitoring_plan.get("review_triggers", []) or [])
        if str(item).strip()
    ][:3]

    tracked = _ordered_union(universe, current_positions.keys(), prior_registry.keys())
    out: dict[str, dict] = {}
    for idx, ticker in enumerate(tracked):
        prev = prior_registry.get(ticker, {}) if isinstance(prior_registry.get(ticker), dict) else {}
        current_weight = float(current_positions.get(ticker, 0.0))
        proposed_weight = float(proposed_positions.get(ticker, current_weight))
        status = _derive_idea_status(ticker, universe, current_weight, str(prev.get("status", "")).strip())
        if status == "inactive":
            continue
        if universe and ticker == universe[0]:
            role = "main"
        elif ticker in universe[1:]:
            role = "hedge_candidate"
        else:
            role = str(prev.get("role", "")).strip() or "carry"
        out[ticker] = {
            "ticker": ticker,
            "status": status,
            "role": role,
            "priority_rank": idx + 1,
            "in_current_universe": ticker in universe,
            "current_weight": round(current_weight, 6),
            "proposed_weight": round(proposed_weight, 6),
            "last_action_type": action_type if ticker in universe else str(prev.get("last_action_type", "")).strip(),
            "last_intent": intent if ticker in universe else str(prev.get("last_intent", "")).strip() or intent,
            "benchmark": benchmark or str(prev.get("benchmark", "")).strip().upper(),
            "review_frequency": review_frequency if ticker in universe else str(prev.get("review_frequency", "")).strip() or review_frequency,
            "review_triggers": review_triggers if ticker in universe else list(prev.get("review_triggers", []) or review_triggers),
            "first_seen_at": str(prev.get("first_seen_at", "")).strip() or as_of,
            "updated_at": as_of,
            "source_run_id": run_id,
        }
    return out


def _build_portfolio_memory(
    state: InvestmentState,
    decision: dict,
    active_ideas: dict[str, dict],
    conviction_by_ticker: Optional[dict[str, dict]] = None,
    allocation_signals_by_ticker: Optional[dict[str, dict]] = None,
) -> dict[str, dict]:
    prior_memory = state.get("portfolio_memory", {}) if isinstance(state.get("portfolio_memory"), dict) else {}
    as_of = str(state.get("as_of", "")).strip()
    user_request = str(state.get("user_request", "")).strip()
    action_type = str(decision.get("action_type", "")).strip() or "initial_delegation"
    intent = str(decision.get("intent", "")).strip() or str(state.get("intent", "")).strip()
    out: dict[str, dict] = {
        str(ticker).strip().upper(): dict(value)
        for ticker, value in prior_memory.items()
        if str(ticker).strip() and isinstance(value, dict)
    }

    for ticker, idea in active_ideas.items():
        prev = out.get(ticker, {})
        conviction = (
            conviction_by_ticker.get(ticker, {})
            if isinstance(conviction_by_ticker, dict)
            else {}
        )
        allocator_signals = (
            allocation_signals_by_ticker.get(ticker, {})
            if isinstance(allocation_signals_by_ticker, dict)
            else {}
        )
        prior_count = prev.get("times_seen", 0)
        try:
            times_seen = int(prior_count) + 1
        except (TypeError, ValueError):
            times_seen = 1
        out[ticker] = {
            **prev,
            "ticker": ticker,
            "first_seen_at": str(prev.get("first_seen_at", "")).strip() or str(idea.get("first_seen_at", "")).strip() or as_of,
            "last_seen_at": as_of,
            "times_seen": times_seen,
            "last_user_request": user_request,
            "last_intent": str(idea.get("last_intent", "")).strip() or intent,
            "last_action_type": str(idea.get("last_action_type", "")).strip() or action_type,
            "current_weight": float(idea.get("current_weight", 0.0) or 0.0),
            "proposed_weight": float(idea.get("proposed_weight", 0.0) or 0.0),
            "thesis_status": str(idea.get("status", "")).strip() or "candidate",
            "role": str(idea.get("role", "")).strip() or str(prev.get("role", "")).strip(),
            "review_frequency": str(idea.get("review_frequency", "")).strip() or str(prev.get("review_frequency", "")).strip(),
            "review_triggers": list(idea.get("review_triggers", []) or prev.get("review_triggers", [])),
            "benchmark": str(idea.get("benchmark", "")).strip().upper() or str(prev.get("benchmark", "")).strip().upper(),
            "conviction": conviction if conviction else dict(prev.get("conviction", {}) or {}),
            "allocator_signals": allocator_signals if allocator_signals else dict(prev.get("allocator_signals", {}) or {}),
        }
    return out


def _build_monitoring_backlog(
    state: InvestmentState,
    decision: dict,
    active_ideas: dict[str, dict],
) -> list[dict]:
    prior_backlog = state.get("monitoring_backlog", []) if isinstance(state.get("monitoring_backlog"), list) else []
    monitoring_plan = decision.get("monitoring_plan", {}) if isinstance(decision.get("monitoring_plan"), dict) else {}
    review_frequency = str(monitoring_plan.get("review_frequency", "")).strip() or "event-driven"
    review_triggers = [
        str(item).strip()
        for item in (monitoring_plan.get("review_triggers", []) or [])
        if str(item).strip()
    ]
    as_of = str(state.get("as_of", "")).strip()
    out: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    def _append(item: dict) -> None:
        ticker = str(item.get("ticker", "__PORTFOLIO__")).strip().upper() or "__PORTFOLIO__"
        trigger = str(item.get("trigger", "")).strip()
        source = str(item.get("source", "monitoring_plan")).strip().lower()
        if not trigger:
            return
        key = (ticker, trigger, source)
        if key in seen:
            return
        seen.add(key)
        payload = dict(item)
        payload["ticker"] = ticker
        payload["trigger"] = trigger
        payload["source"] = source
        payload["status"] = str(payload.get("status", "open")).strip().lower() or "open"
        payload["updated_at"] = as_of
        out.append(payload)

    for item in prior_backlog:
        if not isinstance(item, dict):
            continue
        if str(item.get("status", "open")).strip().lower() in {"closed", "done", "archived"}:
            continue
        _append(item)

    for trigger in review_triggers:
        _append({
            "ticker": "__PORTFOLIO__",
            "scope": "portfolio",
            "priority": 1,
            "trigger": trigger,
            "source": "monitoring_plan",
            "review_frequency": review_frequency,
            "created_at": as_of,
        })

    for ticker, idea in active_ideas.items():
        status = str(idea.get("status", "")).strip()
        if status not in {"active_position", "candidate", "monitor"}:
            continue
        priority = 1 if status == "active_position" else (2 if status == "candidate" else 3)
        _append({
            "ticker": ticker,
            "scope": "ticker",
            "priority": priority,
            "trigger": f"{ticker} thesis/status review ({status})",
            "source": "active_ideas",
            "review_frequency": str(idea.get("review_frequency", "")).strip() or review_frequency,
            "created_at": str(idea.get("updated_at", "")).strip() or as_of,
            "role": str(idea.get("role", "")).strip(),
            "current_weight": float(idea.get("current_weight", 0.0) or 0.0),
        })
    return out


def _decision_direction(decision: Any) -> float:
    text = str(decision or "").strip().lower()
    if text in {"bullish", "long", "buy", "allow"}:
        return 1.0
    if text in {"bearish", "short", "sell", "avoid", "reject"}:
        return -1.0
    return 0.0


def _desk_conviction_component(output: Any, desk: str) -> dict:
    if not isinstance(output, dict):
        return {"score": 0.0, "direction": 0.0, "confidence": 0.0, "signal_strength": 0.0}
    decision = output.get("primary_decision", output.get("decision", "neutral"))
    direction = _decision_direction(decision)
    try:
        confidence = float(output.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    try:
        signal_strength = float(output.get("signal_strength", 0.0) or 0.0)
    except (TypeError, ValueError):
        signal_strength = 0.0
    if desk == "quant" and signal_strength <= 0.0:
        try:
            signal_strength = min(abs(float(output.get("final_allocation_pct", 0.0) or 0.0)) * 4.0, 1.0)
        except (TypeError, ValueError):
            signal_strength = 0.0
    score = direction * max(0.0, confidence) * max(0.0, signal_strength or 0.5)
    return {
        "score": round(score, 6),
        "direction": round(direction, 6),
        "confidence": round(confidence, 6),
        "signal_strength": round(signal_strength, 6),
    }


def _current_conviction_snapshot(state: InvestmentState, ticker: str, role: str) -> dict:
    symbol = str(ticker or "").strip().upper()
    target = str(state.get("target_ticker", "")).strip().upper()
    memory = state.get("portfolio_memory", {}) if isinstance(state.get("portfolio_memory"), dict) else {}
    prior = memory.get(symbol, {}) if isinstance(memory.get(symbol), dict) else {}
    prior_conviction = prior.get("conviction", {}) if isinstance(prior.get("conviction"), dict) else {}

    snapshot = {
        "source": "memory" if prior_conviction else "neutral",
        "macro": dict(prior_conviction.get("macro", {}) or {}),
        "fundamental": dict(prior_conviction.get("fundamental", {}) or {}),
        "sentiment": dict(prior_conviction.get("sentiment", {}) or {}),
        "quant": dict(prior_conviction.get("quant", {}) or {}),
        "hedge_lite": dict(prior_conviction.get("hedge_lite", {}) or {}),
        "composite_score": float(prior_conviction.get("composite_score", 0.0) or 0.0),
    }

    if symbol == target:
        macro = _desk_conviction_component(state.get("macro_analysis", {}), "macro")
        funda = _desk_conviction_component(state.get("fundamental_analysis", {}), "fundamental")
        senti = _desk_conviction_component(state.get("sentiment_analysis", {}), "sentiment")
        quant = _desk_conviction_component(state.get("technical_analysis", {}), "quant")
        scorecard = state.get("decision_quality_scorecard", {}) if isinstance(state.get("decision_quality_scorecard"), dict) else {}
        quality_by_desk = scorecard.get("desks", {}) if isinstance(scorecard.get("desks"), dict) else {}
        for desk_name, comp in {
            "macro": macro,
            "fundamental": funda,
            "sentiment": senti,
            "quant": quant,
        }.items():
            desk_quality = quality_by_desk.get(desk_name, {}) if isinstance(quality_by_desk.get(desk_name), dict) else {}
            quality_score = _coerce_float(desk_quality.get("quality_score"))
            if quality_score is None:
                continue
            multiplier = 0.5 + 0.5 * _clamp(float(quality_score), 0.0, 1.0)
            comp["score"] = round(float(comp.get("score", 0.0) or 0.0) * multiplier, 6)
            comp["quality_multiplier"] = round(multiplier, 6)
        components = {"macro": macro, "fundamental": funda, "sentiment": senti, "quant": quant}
        scores = [comp["score"] for comp in components.values()]
        snapshot.update(components)
        snapshot["source"] = "current_run"
        snapshot["composite_score"] = round(sum(scores) / len(scores), 6)

    hedge_lite = state.get("hedge_lite", {}) if isinstance(state.get("hedge_lite"), dict) else {}
    hedge_rows = hedge_lite.get("hedges", {}) if isinstance(hedge_lite.get("hedges"), dict) else {}
    if symbol in hedge_rows and isinstance(hedge_rows[symbol], dict):
        row = hedge_rows[symbol]
        raw_score = float(row.get("score", 0.0) or 0.0)
        snapshot["hedge_lite"] = {
            "score": round(raw_score, 6),
            "selected": bool(row.get("selected")),
            "status": str(row.get("status", "")).strip() or "unknown",
        }
        if role == "hedge_candidate":
            base = float(snapshot.get("composite_score", 0.0) or 0.0)
            blended = max(base, raw_score * 0.8)
            snapshot["composite_score"] = round(blended, 6)
            if snapshot.get("source") == "neutral":
                snapshot["source"] = "hedge_lite"

    return snapshot


def _extract_allocator_signals_from_fundamental(output: Any, as_of: str) -> dict:
    if not isinstance(output, dict):
        return {
            "source": "neutral",
            "status": "insufficient_data",
            "expected_return_pct": 0.0,
            "downside_pct": 0.0,
            "catalyst_proximity_score": 0.0,
            "catalyst_days": None,
            "catalyst_type": "",
            "catalyst_source": "",
        }

    valuation_anchor = output.get("valuation_anchor", {}) if isinstance(output.get("valuation_anchor"), dict) else {}
    model_pack = output.get("model_pack", {}) if isinstance(output.get("model_pack"), dict) else {}
    scenario_targets = model_pack.get("scenario_targets", {}) if isinstance(model_pack.get("scenario_targets"), dict) else {}
    base_target = scenario_targets.get("base", {}) if isinstance(scenario_targets.get("base"), dict) else {}
    bear_target = scenario_targets.get("bear", {}) if isinstance(scenario_targets.get("bear"), dict) else {}
    catalyst_calendar = output.get("catalyst_calendar", []) if isinstance(output.get("catalyst_calendar"), list) else []

    street_upside = _coerce_float(valuation_anchor.get("price_target_upside_pct"))
    model_upside = _coerce_float(base_target.get("upside_pct"))
    if street_upside is not None and model_upside is not None:
        expected_return_pct = round(model_upside * 0.6 + street_upside * 0.4, 4)
        source = "fundamental_current_run_blended"
    elif model_upside is not None:
        expected_return_pct = round(model_upside, 4)
        source = "fundamental_model_pack"
    elif street_upside is not None:
        expected_return_pct = round(street_upside, 4)
        source = "fundamental_valuation_anchor"
    else:
        expected_return_pct = 0.0
        source = "neutral"

    raw_downside = _coerce_float(bear_target.get("downside_pct"))
    downside_pct = round(abs(raw_downside) if raw_downside is not None and raw_downside < 0 else 0.0, 4)

    catalyst = None
    for item in catalyst_calendar:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "")).strip().lower()
        if status == "stale":
            continue
        source_class = str(item.get("source_classification", "")).strip().lower()
        days = _coerce_int(item.get("days_to_event"))
        if days is None:
            days = _days_to_event(as_of, item.get("date"))
        if days is not None and days < 0:
            continue
        rank = 0 if source_class == "confirmed" else 1
        candidate = (rank, days if days is not None else 10_000, item)
        if catalyst is None or candidate < catalyst:
            catalyst = candidate

    catalyst_proximity_score = 0.0
    catalyst_days = None
    catalyst_type = ""
    catalyst_source = ""
    if catalyst is not None:
        item = catalyst[2]
        catalyst_days = _coerce_int(item.get("days_to_event"))
        if catalyst_days is None:
            catalyst_days = _days_to_event(as_of, item.get("date"))
        catalyst_type = str(item.get("type", "")).strip()
        catalyst_source = str(item.get("source_classification", "")).strip() or "unknown"
        if catalyst_days is not None:
            if catalyst_days <= 7:
                catalyst_proximity_score = 1.0
            elif catalyst_days <= 14:
                catalyst_proximity_score = 0.8
            elif catalyst_days <= 30:
                catalyst_proximity_score = 0.55
            elif catalyst_days <= 60:
                catalyst_proximity_score = 0.25
            else:
                catalyst_proximity_score = 0.05

    status = "ok" if source != "neutral" or downside_pct > 0 or catalyst_proximity_score > 0 else "insufficient_data"
    return {
        "source": source,
        "status": status,
        "expected_return_pct": expected_return_pct,
        "downside_pct": downside_pct,
        "catalyst_proximity_score": round(catalyst_proximity_score, 4),
        "catalyst_days": catalyst_days,
        "catalyst_type": catalyst_type,
        "catalyst_source": catalyst_source,
    }


def _current_allocator_signal_snapshot(state: InvestmentState, ticker: str) -> dict:
    symbol = str(ticker or "").strip().upper()
    target = str(state.get("target_ticker", "")).strip().upper()
    memory = state.get("portfolio_memory", {}) if isinstance(state.get("portfolio_memory"), dict) else {}
    prior = memory.get(symbol, {}) if isinstance(memory.get(symbol), dict) else {}
    prior_signals = prior.get("allocator_signals", {}) if isinstance(prior.get("allocator_signals"), dict) else {}

    snapshot = {
        "source": str(prior_signals.get("source", "neutral")).strip() or "neutral",
        "status": str(prior_signals.get("status", "insufficient_data")).strip() or "insufficient_data",
        "expected_return_pct": float(prior_signals.get("expected_return_pct", 0.0) or 0.0),
        "downside_pct": float(prior_signals.get("downside_pct", 0.0) or 0.0),
        "catalyst_proximity_score": float(prior_signals.get("catalyst_proximity_score", 0.0) or 0.0),
        "catalyst_days": _coerce_int(prior_signals.get("catalyst_days")),
        "catalyst_type": str(prior_signals.get("catalyst_type", "")).strip(),
        "catalyst_source": str(prior_signals.get("catalyst_source", "")).strip(),
    }

    if symbol == target:
        current = _extract_allocator_signals_from_fundamental(state.get("fundamental_analysis", {}), str(state.get("as_of", "")).strip())
        if current.get("status") != "insufficient_data":
            return current
    return snapshot


def _normalize_positive_weights(raw: dict[str, float]) -> dict[str, float]:
    cleaned = {t: max(0.0, float(w)) for t, w in (raw or {}).items()}
    total = float(sum(cleaned.values()))
    if total <= 1e-12:
        return {}
    return {t: float(w) / total for t, w in cleaned.items()}


def _derive_allocator_gross_target(action_type: str, current_gross: float, universe_size: int, mandate: dict) -> float:
    mandate_cap = _coerce_float((mandate.get("constraints", {}) or {}).get("target_gross_exposure"))
    if current_gross <= 1e-12:
        base = 0.35 if universe_size <= 1 else 0.60
    else:
        base = min(current_gross, 1.0)
    if action_type == "scale_down":
        base *= 0.75
    elif action_type == "pivot_strategy":
        base *= 0.65
    elif action_type == "fallback_abort":
        base = min(base, 0.35)
    elif action_type == "add_hedge":
        base = min(max(base, 0.50), 0.90)
    else:
        floor = 0.35 if universe_size <= 1 else 0.60
        base = min(max(base, floor), 1.0)
    if mandate_cap is not None:
        base = min(base, float(mandate_cap))
    return round(max(0.10, min(base, 1.0)), 4)


def _cap_book_weights(targets: dict[str, float], gross_target: float, single_name_cap: float) -> dict[str, float]:
    capped = {t: max(0.0, float(w)) for t, w in (targets or {}).items()}
    if not capped or gross_target <= 1e-12:
        return {}
    single_name_cap = max(0.0, min(float(single_name_cap), gross_target))
    for _ in range(5):
        overflow = 0.0
        uncapped: list[str] = []
        for ticker, weight in capped.items():
            if weight > single_name_cap:
                overflow += weight - single_name_cap
                capped[ticker] = single_name_cap
            else:
                uncapped.append(ticker)
        if overflow <= 1e-12 or not uncapped:
            break
        raw_room = {t: max(0.0, single_name_cap - capped[t]) for t in uncapped}
        room_total = float(sum(raw_room.values()))
        if room_total <= 1e-12:
            break
        for ticker in uncapped:
            capped[ticker] += overflow * (raw_room[ticker] / room_total)
    total = float(sum(capped.values()))
    if total <= 1e-12:
        return {}
    scale = gross_target / total
    return {t: round(max(0.0, w * scale), 6) for t, w in capped.items()}


def _build_book_allocation_plan(
    state: InvestmentState,
    decision: dict,
    active_ideas: dict[str, dict],
) -> tuple[dict, list[dict]]:
    if not active_ideas:
        return {}, []

    brief = decision.get("investment_brief", {}) if isinstance(decision.get("investment_brief"), dict) else {}
    universe = _normalize_ticker_list(brief.get("target_universe", []))
    main = universe[0] if universe else ""
    action_type = str(decision.get("action_type", "")).strip() or "initial_delegation"
    mandate = decision.get("portfolio_mandate", {}) if isinstance(decision.get("portfolio_mandate"), dict) else {}
    current_positions = _normalize_weight_map(
        state.get("positions_final") or state.get("positions_proposed") or {}
    )
    current_gross = float(sum(abs(float(v)) for v in current_positions.values()))
    gross_target = _derive_allocator_gross_target(action_type, current_gross, len(universe), mandate)
    single_name_cap = _coerce_float((mandate.get("constraints", {}) or {}).get("max_single_name_weight"))
    if single_name_cap is None:
        single_name_cap = min(0.35, gross_target)
    single_name_cap = max(0.10, float(single_name_cap))
    scorecard = state.get("decision_quality_scorecard", {}) if isinstance(state.get("decision_quality_scorecard"), dict) else {}
    quality_overall = _coerce_float(scorecard.get("overall_score"))
    weak_desks = sorted({
        str(desk).strip().lower()
        for desk in (scorecard.get("weak_desks", []) or [])
        if str(desk).strip().lower()
    })
    quality_haircut = 1.0
    quality_adjustments: list[str] = []
    if quality_overall is not None:
        if quality_overall < 0.35:
            quality_haircut *= 0.65
            quality_adjustments.append("overall_score_critical")
        elif quality_overall < 0.50:
            quality_haircut *= 0.80
            quality_adjustments.append("overall_score_low")
        elif quality_overall < 0.65:
            quality_haircut *= 0.90
            quality_adjustments.append("overall_score_soft")
    if len(weak_desks) >= 3:
        quality_haircut *= 0.85
        quality_adjustments.append("multi_desk_weakness")
    elif len(weak_desks) == 2:
        quality_haircut *= 0.92
        quality_adjustments.append("two_weak_desks")
    quality_haircut = round(max(0.55, min(quality_haircut, 1.0)), 4)
    gross_target = round(max(0.10, min(gross_target * quality_haircut, 1.0)), 4)
    single_name_cap = round(min(gross_target, max(0.10, single_name_cap * min(1.0, quality_haircut + 0.05))), 4)

    raw_scores: dict[str, float] = {}
    rows: list[dict] = []
    new_universe = set(universe)
    conviction_by_ticker = {
        ticker: _current_conviction_snapshot(state, ticker, str(idea.get("role", "")).strip())
        for ticker, idea in active_ideas.items()
    }
    allocation_signals_by_ticker = {
        ticker: _current_allocator_signal_snapshot(state, ticker)
        for ticker in active_ideas.keys()
    }
    for ticker, idea in active_ideas.items():
        current_weight = max(0.0, float(idea.get("current_weight", 0.0) or 0.0))
        in_universe = bool(idea.get("in_current_universe"))
        role = str(idea.get("role", "")).strip() or "carry"
        status = str(idea.get("status", "")).strip() or "candidate"
        conviction = conviction_by_ticker.get(ticker, {})
        conviction_score = float(conviction.get("composite_score", 0.0) or 0.0)
        allocator_signals = allocation_signals_by_ticker.get(ticker, {})
        expected_return_pct = float(allocator_signals.get("expected_return_pct", 0.0) or 0.0)
        downside_pct = float(allocator_signals.get("downside_pct", 0.0) or 0.0)
        catalyst_proximity = float(allocator_signals.get("catalyst_proximity_score", 0.0) or 0.0)
        expected_return_score = _clamp(expected_return_pct / 20.0, -1.0, 1.0)
        downside_penalty = _clamp(downside_pct / 20.0, 0.0, 1.0)

        score = 0.0
        reasons: list[str] = []
        if current_weight > 0:
            score += 1.0 + min(current_weight, 0.5)
            reasons.append("existing_position")
        if ticker == main:
            score += 2.0
            reasons.append("main_candidate")
        elif role == "hedge_candidate":
            score += 1.0
            reasons.append("hedge_candidate")
        elif in_universe:
            score += 0.75
            reasons.append("in_universe")
        if status == "monitor":
            score -= 0.25
        if action_type == "scale_down" and current_weight > 0:
            score *= 0.85
            reasons.append("scale_down_bias")
        if action_type == "pivot_strategy" and ticker not in new_universe and current_weight > 0:
            score *= 0.35
            reasons.append("pivot_displacement")
        if action_type == "fallback_abort" and ticker not in new_universe and current_weight > 0:
            score *= 0.20
            reasons.append("fallback_displacement")
        if action_type == "add_hedge" and role == "hedge_candidate":
            score += 0.75
            reasons.append("hedge_budget_priority")
        if conviction_score > 0:
            score += min(conviction_score, 0.8) * (1.8 if ticker == main else 1.2)
            reasons.append("positive_conviction")
        elif conviction_score < 0:
            score -= min(abs(conviction_score), 0.8) * (1.6 if current_weight > 0 else 1.0)
            reasons.append("negative_conviction")
        hedge_component = conviction.get("hedge_lite", {}) if isinstance(conviction.get("hedge_lite"), dict) else {}
        if role == "hedge_candidate" and hedge_component.get("selected"):
            score += 0.35
            reasons.append("selected_hedge")
        if expected_return_score > 0:
            score += expected_return_score * (1.25 if ticker == main else 0.85)
            reasons.append("expected_return_support")
        elif expected_return_score < 0:
            score += expected_return_score * (1.1 if current_weight > 0 else 0.8)
            reasons.append("negative_expected_return")
        if downside_penalty > 0:
            score -= downside_penalty * (1.15 if current_weight > 0 else 0.9)
            reasons.append("downside_risk")
        if catalyst_proximity > 0:
            catalyst_direction = 0.0
            if conviction_score > 0.05 or expected_return_score > 0.15 or ticker == main:
                catalyst_direction = 1.0
            elif conviction_score < -0.05 or expected_return_score < -0.1:
                catalyst_direction = -1.0
            elif in_universe:
                catalyst_direction = 0.4
            score += catalyst_proximity * catalyst_direction * 0.65
            reasons.append("catalyst_proximity")

        raw_scores[ticker] = max(0.0, score)
        rows.append({
            "ticker": ticker,
            "status": status,
            "role": role,
            "in_current_universe": in_universe,
            "current_weight": round(current_weight, 6),
            "score": round(max(0.0, score), 6),
            "conviction_score": round(conviction_score, 6),
            "conviction_source": str(conviction.get("source", "neutral")),
            "expected_return_score": round(expected_return_score, 6),
            "downside_penalty": round(downside_penalty, 6),
            "catalyst_proximity_score": round(catalyst_proximity, 6),
            "drivers": reasons,
        })

    mix_weights = _normalize_positive_weights(raw_scores)
    target_book_weights = _cap_book_weights(
        {t: mix * gross_target for t, mix in mix_weights.items()},
        gross_target=gross_target,
        single_name_cap=single_name_cap,
    )
    target_mix_weights = {
        ticker: round(float(weight) / gross_target, 6)
        for ticker, weight in target_book_weights.items()
        if gross_target > 1e-12
    }

    enriched_rows: list[dict] = []
    for row in rows:
        ticker = row["ticker"]
        current_weight = float(row["current_weight"])
        target_book = float(target_book_weights.get(ticker, 0.0))
        target_mix = float(target_mix_weights.get(ticker, 0.0))
        delta = target_book - current_weight
        displaced_by = universe[:2] if current_weight > target_book + 0.03 and ticker not in new_universe else []
        if current_weight > 1e-12 and target_book < 0.01 and displaced_by:
            book_action = "replace"
        elif current_weight > 1e-12 and target_book < 0.01:
            book_action = "exit"
        elif current_weight <= 1e-12 and target_book >= 0.05:
            book_action = "add"
        elif current_weight <= 1e-12 and target_book < 0.03:
            book_action = "ignore"
        elif delta > 0.03:
            book_action = "scale_up"
        elif delta < -0.03:
            book_action = "scale_down"
        else:
            book_action = "hold"
        conviction = conviction_by_ticker.get(ticker, {})
        allocator_signals = allocation_signals_by_ticker.get(ticker, {})
        enriched_rows.append({
            **row,
            "target_mix_weight": round(target_mix, 6),
            "target_book_weight": round(target_book, 6),
            "delta_weight": round(delta, 6),
            "book_action": book_action,
            "displaced_by": displaced_by,
            "conviction_components": {
                "macro": float(((conviction.get("macro", {}) or {}).get("score", 0.0) or 0.0)),
                "fundamental": float(((conviction.get("fundamental", {}) or {}).get("score", 0.0) or 0.0)),
                "sentiment": float(((conviction.get("sentiment", {}) or {}).get("score", 0.0) or 0.0)),
                "quant": float(((conviction.get("quant", {}) or {}).get("score", 0.0) or 0.0)),
                "hedge_lite": float(((conviction.get("hedge_lite", {}) or {}).get("score", 0.0) or 0.0)),
            },
            "allocation_signal_components": {
                "expected_return_pct": round(float(allocator_signals.get("expected_return_pct", 0.0) or 0.0), 4),
                "downside_pct": round(float(allocator_signals.get("downside_pct", 0.0) or 0.0), 4),
                "catalyst_proximity_score": round(float(allocator_signals.get("catalyst_proximity_score", 0.0) or 0.0), 4),
                "catalyst_days": _coerce_int(allocator_signals.get("catalyst_days")),
                "catalyst_type": str(allocator_signals.get("catalyst_type", "")).strip(),
                "source": str(allocator_signals.get("source", "neutral")).strip() or "neutral",
            },
        })

    capital_competition = sorted(
        enriched_rows,
        key=lambda item: (
            -float(item.get("score", 0.0)),
            -float(item.get("target_book_weight", 0.0)),
            str(item.get("ticker", "")),
        ),
    )
    for idx, row in enumerate(capital_competition, start=1):
        row["rank"] = idx

    plan = {
        "status": "ok",
        "action_type": action_type,
        "current_gross": round(current_gross, 6),
        "gross_target": gross_target,
        "recommended_gross_delta": round(gross_target - current_gross, 6),
        "single_name_cap": round(single_name_cap, 6),
        "quality_haircut": quality_haircut,
        "quality_overall_score": round(quality_overall, 4) if quality_overall is not None else None,
        "quality_weak_desks": weak_desks,
        "quality_adjustments": quality_adjustments,
        "weights_relative": target_mix_weights,
        "target_book_weights": target_book_weights,
        "current_weights": {t: round(w, 6) for t, w in current_positions.items()},
        "current_universe": universe,
        "main_ticker": main,
        "rebalances": capital_competition,
        "allocator_source": "orchestrator_book_allocator_v1",
        "conviction_by_ticker": conviction_by_ticker,
        "allocation_signals_by_ticker": allocation_signals_by_ticker,
    }
    return plan, capital_competition


def _apply_portfolio_mandate(decision: dict, portfolio_context: Optional[dict]) -> dict:
    mandate = _normalize_portfolio_context(portfolio_context)
    if not mandate:
        return decision

    out = dict(decision or {})
    brief = dict(out.get("investment_brief", {}) or {})
    raw_tasks = out.get("desk_tasks", {}) or {}
    desk_tasks = {
        str(k): (dict(v) if isinstance(v, dict) else {})
        for k, v in raw_tasks.items()
    }
    universe = _normalize_ticker_list(brief.get("target_universe", []))
    original_universe = list(universe)
    changes: list[str] = []

    allowed = set(mandate.get("allowed_tickers", []))
    blocked = set(mandate.get("blocked_tickers", []))
    required = [
        t for t in mandate.get("required_tickers", [])
        if (not allowed or t in allowed) and t not in blocked
    ]
    preferred_hedges = [
        t for t in mandate.get("preferred_hedges", [])
        if (not allowed or t in allowed) and t not in blocked
    ]

    if allowed:
        filtered = [t for t in universe if t in allowed]
        if filtered != universe:
            changes.append("allowed_tickers")
        universe = filtered

    if blocked:
        filtered = [t for t in universe if t not in blocked]
        if filtered != universe:
            changes.append("blocked_tickers")
        universe = filtered

    if not universe:
        fallback = []
        if required:
            fallback.extend(required)
        elif allowed:
            fallback.extend(list(mandate.get("allowed_tickers", []))[:1])
        elif mandate.get("benchmark") and mandate["benchmark"] not in blocked:
            fallback.append(mandate["benchmark"])
        if fallback:
            universe = _normalize_ticker_list(fallback)
            changes.append("fallback_universe")

    for ticker in required:
        if ticker not in universe:
            universe.append(ticker)
            changes.append("required_tickers")

    for hedge in preferred_hedges:
        if hedge not in universe:
            universe.append(hedge)
            changes.append("preferred_hedges")

    trimmed = _trim_universe(universe, mandate.get("max_universe_size"), required)
    if trimmed != universe:
        changes.append("max_universe_size")
    universe = trimmed

    quant = dict(desk_tasks.get("quant", {}) or {})
    if mandate.get("quant_risk_budget"):
        if quant.get("risk_budget") != mandate["quant_risk_budget"]:
            changes.append("quant_risk_budget")
        quant["risk_budget"] = mandate["quant_risk_budget"]
    if mandate.get("benchmark"):
        quant = _append_focus_area(quant, f"벤치마크({mandate['benchmark']}) 대비 베타/상대 성과 점검")
        desk_tasks["macro"] = _append_focus_area(
            desk_tasks.get("macro", {}),
            f"벤치마크({mandate['benchmark']}) 대비 매크로 레짐 적합성 점검",
        )
    if mandate.get("preferred_hedges"):
        quant = _append_focus_area(
            quant,
            f"지정 헤지 후보({', '.join(mandate['preferred_hedges'][:3])}) 상관/유동성 점검",
        )
        desk_tasks["sentiment"] = _append_focus_area(
            desk_tasks.get("sentiment", {}),
            "리스크오프 흐름과 헤지 수요 변화 확인",
        )
    if mandate.get("max_single_name_weight") is not None:
        quant = _append_focus_area(
            quant,
            f"단일 아이디어 비중 한도({float(mandate['max_single_name_weight']):.0%}) 적합성 점검",
        )
        desk_tasks["fundamental"] = _append_focus_area(
            desk_tasks.get("fundamental", {}),
            "아이디어 집중도 정당화 가능 여부 점검",
        )
    if (
        mandate.get("target_gross_exposure") is not None
        or mandate.get("target_net_exposure") is not None
        or mandate.get("max_drawdown_pct") is not None
    ):
        quant = _append_focus_area(quant, "포트폴리오 mandate(그로스/넷/드로다운) 적합성 점검")
    desk_tasks["quant"] = quant

    brief["target_universe"] = universe
    rationale = str(brief.get("rationale", "")).strip()
    rationale_notes: list[str] = []
    if "allowed_tickers" in changes or "blocked_tickers" in changes or "fallback_universe" in changes:
        rationale_notes.append("허용/금지 유니버스 제약 반영")
    if "preferred_hedges" in changes:
        rationale_notes.append("지정 헤지 후보 반영")
    if "quant_risk_budget" in changes:
        rationale_notes.append(f"quant 리스크 예산을 {mandate['quant_risk_budget']}로 조정")
    if "max_universe_size" in changes:
        rationale_notes.append("유니버스 크기 한도 반영")
    if rationale_notes:
        suffix = " 포트폴리오 mandate 반영: " + "; ".join(rationale_notes) + "."
        if suffix not in rationale:
            brief["rationale"] = (rationale + suffix).strip()

    out["investment_brief"] = brief
    out["desk_tasks"] = desk_tasks
    out["portfolio_mandate"] = {
        "applied": bool(changes),
        "changes": changes,
        "original_universe": original_universe,
        "final_universe": universe,
        "benchmark": mandate.get("benchmark"),
        "review_frequency": mandate.get("review_frequency") or "event-driven",
        "constraints": {
            "allowed_tickers": mandate.get("allowed_tickers", []),
            "blocked_tickers": mandate.get("blocked_tickers", []),
            "required_tickers": required,
            "preferred_hedges": preferred_hedges,
            "max_universe_size": mandate.get("max_universe_size"),
            "quant_risk_budget": mandate.get("quant_risk_budget"),
            "max_single_name_weight": mandate.get("max_single_name_weight"),
            "max_drawdown_pct": mandate.get("max_drawdown_pct"),
            "target_gross_exposure": mandate.get("target_gross_exposure"),
            "target_net_exposure": mandate.get("target_net_exposure"),
        },
    }
    out["monitoring_plan"] = _build_monitoring_plan(mandate)
    return out


def classify_intent_rules(user_request: str) -> dict:
    """
    규칙 기반 intent 분류 — LLM 실패 시 fallback.
    5 intents: single_ticker_entry | overheated_check | compare_tickers | market_outlook | event_risk
    """
    text    = user_request.lower()
    tickers = [m for m in _TICKER_RE.findall(user_request)
               if m not in ("AI", "ETF", "PM", "CEO", "CIO", "IPO")]

    # compare_tickers: ≥2 tickers or VS keywords
    compare_kws = ["vs", "비교", "대비", "or ", "대 ", "비해"]
    if len(tickers) >= 2 or any(kw in text for kw in compare_kws):
        universe = tickers[:4] if len(tickers) >= 2 else ["SPY", "QQQ"]
        return {"intent": "compare_tickers", "universe": universe,
                "horizon_days": 30,
                "desk_tasks": _default_desk_tasks(30, "Moderate", "valuation")}

    # overheated_check
    heat_kws = ["과열", "너무 올랐", "오른", "고점", "거품", "overbought", "expensive", "stretched", "비싸"]
    if any(kw in text for kw in heat_kws):
        ticker = _extract_ticker(user_request)
        universe = [ticker, "SPY"] if ticker else ["SPY", "QQQ"]
        return {"intent": "overheated_check", "universe": universe,
                "horizon_days": 90,
                "desk_tasks": _default_desk_tasks(90, "Conservative", "valuation")}

    # event_risk
    event_kws = ["실적", "earnings", "이벤트", "발표", "fomc", "어닝", "앞두", "before"]
    if any(kw in text for kw in event_kws):
        ticker = _extract_ticker(user_request)
        universe = [ticker] if ticker else ["SPY", "QQQ"]
        return {"intent": "event_risk", "universe": universe,
                "horizon_days": 14,
                "desk_tasks": _default_desk_tasks(14, "Conservative", "event")}

    # market_outlook
    outlook_kws = ["시장", "market", "전망", "outlook", "섹터", "sector", "경기", "economy"]
    if any(kw in text for kw in outlook_kws) and not tickers:
        return {"intent": "market_outlook", "universe": ["SPY", "QQQ", "IWM"],
                "horizon_days": 30,
                "desk_tasks": _default_desk_tasks(30, "Moderate")}

    # default:
    # ticker가 있으면 단일 종목 진입, 없으면 시장 전망으로 처리해 AAPL 기본 고정을 방지
    ticker = _extract_ticker(user_request)
    if ticker:
        return {"intent": "single_ticker_entry", "universe": [ticker],
                "horizon_days": 30,
                "desk_tasks": _default_desk_tasks(30, "Moderate")}
    return {"intent": "market_outlook", "universe": ["SPY", "QQQ", "IWM"],
            "horizon_days": 30,
            "desk_tasks": _default_desk_tasks(30, "Moderate")}


def _plan_cache_key(
    user_request: str,
    iteration: int,
    risk_feedback: Optional[dict],
    portfolio_context: Optional[dict] = None,
    book_context: Optional[dict] = None,
) -> str:
    """(user_request + iteration + context hashes) → md5 key."""
    fb_str = json.dumps(risk_feedback or {}, sort_keys=True)
    ctx_str = json.dumps(portfolio_context or {}, sort_keys=True, ensure_ascii=False)
    book_str = json.dumps(book_context or {}, sort_keys=True, ensure_ascii=False)
    raw    = f"{user_request}::{iteration}::{fb_str}::{ctx_str}::{book_str}"
    return hashlib.md5(raw.encode()).hexdigest()


_PLAN_CACHE: dict[str, dict] = {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM 호출 (LLM-first, rules fallback)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _validate_llm_plan(result: dict, user_request: str) -> bool:
    """LLM plan 유효성 검사. 비현실적이면 False."""
    if not isinstance(result, dict):
        return False
    # Must have investment_brief and desk_tasks
    brief = result.get("investment_brief", {})
    if not brief or not brief.get("target_universe"):
        return False
    tasks = result.get("desk_tasks", {})
    if not tasks or not isinstance(tasks, dict):
        return False
    # universe must be non-empty list of strings
    universe = brief.get("target_universe", [])
    if not universe or not all(isinstance(t, str) for t in universe):
        return False
    # desk_tasks must have at least macro and quant
    return "macro" in tasks and "quant" in tasks


def _parse_json_object_maybe_fenced(text: str) -> dict:
    """
    LLM 출력이 ```json ... ``` 형태일 때도 dict로 복원한다.
    """
    s = (text or "").strip()
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


def _call_llm(
    user_request: str,
    iteration: int,
    risk_feedback: Optional[dict] = None,
    portfolio_context: Optional[dict] = None,
    book_context: Optional[dict] = None,
) -> dict:
    """
    LLM-first: LLM plan을 생성하고 Pydantic 검증.
    실패/불일치 시 classify_intent_rules 결과로 fallback.
    캐시 키: (user_request + iteration + risk_feedback_hash).
    """
    use_plan_cache = not force_real_llm_in_tests()
    _orch_trace(
        f"start iteration={iteration} use_plan_cache={use_plan_cache} "
        f"has_risk_feedback={bool(risk_feedback)}"
    )
    _orch_trace(f"user_request={user_request}")

    # ── Plan cache check ──────────────────────────────────────────────
    cache_key = _plan_cache_key(user_request, iteration, risk_feedback, portfolio_context, book_context)
    if use_plan_cache and cache_key in _PLAN_CACHE:
        _orch_trace(f"plan cache hit key={cache_key[:10]}")
        print("   [LLM] plan cache hit")
        return _PLAN_CACHE[cache_key]

    # ── Rules fallback always computed (for validation/fallback) ───────
    rules_plan = _mock_orchestrator_decision(user_request, iteration, risk_feedback)
    _orch_trace(
        "rules_plan intent="
        + str(rules_plan.get("intent", "n/a"))
        + " universe="
        + str((rules_plan.get("investment_brief", {}) or {}).get("target_universe", []))
    )

    if not HAS_LC:
        print("   [LLM] langchain 미설치 → rules fallback")
        if use_plan_cache:
            _PLAN_CACHE[cache_key] = rules_plan
        return rules_plan

    human_msg = _build_orchestrator_human_msg(
        user_request,
        iteration,
        risk_feedback,
        portfolio_context,
        book_context,
    )
    human_msg_preview = human_msg if len(human_msg) <= 800 else (human_msg[:800] + " ...<truncated>")
    _orch_trace("human_msg:\n" + human_msg_preview)

    # llm_router cache (response-level)
    llm, cached = get_llm_with_cache("orchestrator", human_msg)
    if cached is not None:
        _orch_trace("llm response cache hit")
        if use_plan_cache:
            _PLAN_CACHE[cache_key] = cached
        return cached
    if llm is None:
        _orch_trace("no llm available -> rules fallback")
        print("   [LLM] API 키 없음 → rules fallback")
        if use_plan_cache:
            _PLAN_CACHE[cache_key] = rules_plan
        return rules_plan

    # ── LLM call (raw JSON path; fenced JSON 허용) ─────────────────────
    try:
        msgs = [
            SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
            HumanMessage(content=human_msg),
        ]
        _orch_trace("invoke llm with 2 messages (system+human)")
        raw = llm.invoke(msgs)
        result = _parse_json_object_maybe_fenced(getattr(raw, "content", ""))
        _orch_trace(
            "llm parsed action_type="
            + str(result.get("action_type"))
            + " universe="
            + str((result.get("investment_brief", {}) or {}).get("target_universe", []))
        )

        # ── Validate ──────────────────────────────────────────────────
        if not _validate_llm_plan(result, user_request):
            _orch_trace("validate_llm_plan=false -> rules fallback")
            print("   [LLM] plan 검증 실패 → rules fallback")
            result = rules_plan
        else:
            # Intent enrichment from rules (rules adds intent field)
            intent_info = classify_intent_rules(user_request)
            result.setdefault("intent",        intent_info["intent"])
            result.setdefault("horizon_days",  intent_info["horizon_days"])
            print(f"   [LLM] ✅ plan 사용 (intent={result.get('intent')})")

        set_cache("orchestrator", human_msg, result)
        if use_plan_cache:
            _PLAN_CACHE[cache_key] = result
        _orch_trace("return llm plan")
        return result

    except Exception as exc:
        _orch_trace(f"exception -> rules fallback: {exc}")
        print(f"   [LLM] ⚠️ 호출 실패 → rules fallback: {exc}")
        if use_plan_cache:
            _PLAN_CACHE[cache_key] = rules_plan
        return rules_plan


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 규칙 기반 Mock
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_ticker(user_request: str) -> str:
    """사용자 요청에서 티커 심볼 추출 (간이 NER)."""
    # 한국어 종목명 → 티커
    for kr, ticker in _KR_TICKER_MAP.items():
        if kr in user_request:
            return ticker
    # 영문 티커 직접 매칭
    matches = _TICKER_RE.findall(user_request)
    for m in matches:
        if m not in ("AI", "ETF", "PM", "CEO", "CIO", "IPO"):
            return m
    return ""


def _mock_orchestrator_decision(
    user_request: str,
    iteration: int,
    risk_feedback: Optional[dict] = None,
) -> dict:
    """iteration 기반 CIO 의사결정 Mock."""

    intent_info = classify_intent_rules(user_request)
    intent = intent_info.get("intent", "single_ticker_entry")
    universe = list(intent_info.get("universe", []) or [])
    horizon_days = int(intent_info.get("horizon_days", 30))
    desk_tasks = dict(intent_info.get("desk_tasks", {}) or _default_desk_tasks(horizon_days))
    ticker = universe[0] if universe else _extract_ticker(user_request)

    # ── Fallback 모드 (iteration >= MAX_ITERATIONS) ───────────────────────
    if iteration >= MAX_ITERATIONS:
        fb_reasons = []
        if risk_feedback:
            fb_reasons = risk_feedback.get("orchestrator_feedback", {}).get("reasons", [])

        return {
            "current_iteration": iteration,
            "action_type": "fallback_abort",
            "investment_brief": {
                "rationale": (
                    f"{iteration}회 반복에도 리스크 위원회 승인 획득 실패. "
                    f"반려 사유: {', '.join(fb_reasons) if fb_reasons else '지속적 한도 초과'}. "
                    f"{ticker} 신규 매수를 포기하고 현금 관망 또는 방어 섹터 ETF 대안을 권고함."
                ),
                "target_universe": ["CASH", "SHY", "TLT", "XLU"],
            },
            "desk_tasks": {
                "macro": {
                    "horizon_days": 30,
                    "focus_areas": ["방어적 자산배분 점검", "글로벌 리세션 확률"],
                },
                "fundamental": {
                    "horizon_days": 90,
                    "focus_areas": ["방어 섹터 ETF 비용 비교"],
                },
                "sentiment": {
                    "horizon_days": 7,
                    "focus_areas": ["시장 공포 지수(VIX)", "투자심리 바닥 신호"],
                },
                "quant": {
                    "horizon_days": 10,
                    "risk_budget": "Conservative",
                    "focus_areas": ["최소 변동성 포트폴리오", "현금 비중 최적화"],
                },
            },
        }

    # ── 피드백 대응 모드 (iteration >= 1) ──────────────────────────────────
    if iteration >= 1 and risk_feedback:
        fb = risk_feedback.get("orchestrator_feedback", {})
        fb_reasons = fb.get("reasons", [])
        fb_detail = fb.get("detail", "")

        has_structural = any(
            r in fb_reasons
            for r in ("structural_risk", "going_concern", "accounting_fraud")
        )

        # iteration 2 → Pivot
        if iteration == 2 or has_structural:
            return {
                "current_iteration": iteration,
                "action_type": "pivot_strategy",
                "investment_brief": {
                    "rationale": (
                        f"리스크 위원회 반려 ({', '.join(fb_reasons)}). "
                        f"기존 {ticker} 전략의 구조적 문제를 인정하고 방어 섹터로 피벗. "
                        f"원본 요인: {fb_detail[:100]}"
                    ),
                    "target_universe": ["XLV", "XLU", "XLP"],
                },
                "desk_tasks": {
                    "macro": {
                        "horizon_days": 30,
                        "focus_areas": ["방어 섹터 매크로 우위", "금리 민감도"],
                    },
                    "fundamental": {
                        "horizon_days": 90,
                        "focus_areas": ["방어 섹터 FCF 안정성", "배당 수익률"],
                    },
                    "sentiment": {
                        "horizon_days": 7,
                        "focus_areas": ["섹터 로테이션 신호", "기관 자금 흐름"],
                    },
                    "quant": {
                        "horizon_days": 10,
                        "risk_budget": "Conservative",
                        "focus_areas": ["방어 섹터 상대 모멘텀", "하방 리스크 최소화"],
                    },
                },
            }

        # iteration 1 → Scale 또는 Hedge
        has_concentration = any(
            r in fb_reasons
            for r in ("concentration_hhi", "sector_overweight", "component_var_dominant")
        )
        if has_concentration:
            action = "add_hedge"
            rationale = (
                f"리스크 위원회가 집중도/베타 과다 노출 지적 ({', '.join(fb_reasons)}). "
                f"{ticker} 포지션 유지하되 숏 헷지 종목을 탐색·추가하여 Net Exposure를 축소."
            )
        else:
            action = "scale_down"
            rationale = (
                f"리스크 위원회가 한도 위반 지적 ({', '.join(fb_reasons)}). "
                f"{ticker} 투자 아이디어는 유효하나 비중을 축소하여 한도 내 진입."
            )

        return {
            "current_iteration": iteration,
            "action_type": action,
            "investment_brief": {
                "rationale": rationale,
                "target_universe": [ticker],
            },
            "desk_tasks": {
                "macro": {
                    "horizon_days": 30,
                    "focus_areas": ["한도 재확인: 금리 경로", "매크로 레짐 재점검"],
                },
                "fundamental": {
                    "horizon_days": 90,
                    "focus_areas": ["밸류에이션 하방 시나리오", "재무 건전성 재확인"],
                },
                "sentiment": {
                    "horizon_days": 5,
                    "focus_areas": ["단기 뉴스 촉매 유무", "옵션 스큐 변화"],
                },
                "quant": {
                    "horizon_days": 10,
                    "risk_budget": "Conservative",
                    "focus_areas": [
                        "축소 비중 CVaR 재계산" if action == "scale_down"
                        else "헷지 종목 상관계수 분석",
                        "최적 포지션 사이징",
                    ],
                },
            },
        }

    # ── 초기 지시 모드 (iteration == 0) ────────────────────────────────────
    if intent == "market_outlook":
        return {
            "current_iteration": iteration,
            "action_type": "initial_delegation",
            "investment_brief": {
                "rationale": (
                    f"사용자 요청 '{user_request}'은 단일 종목 매수 판단이 아니라 "
                    f"시장/섹터 전망 점검으로 분류되어, 미국 지수 중심 유니버스를 우선 분석함."
                ),
                "target_universe": universe or ["SPY", "QQQ", "IWM"],
            },
            "desk_tasks": desk_tasks,
        }

    return {
        "current_iteration": iteration,
        "action_type": "initial_delegation",
        "investment_brief": {
            "rationale": (
                f"사용자 요청 '{user_request}'을 분석하여 {ticker} 중심의 "
                f"롱 포지션 검토를 개시. 투자 기간 중기(1~6개월). "
                f"4개 데스크에 종합 분석을 지시."
            ),
            "target_universe": universe or [ticker],
        },
        "desk_tasks": desk_tasks,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LangGraph 노드 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def orchestrator_node(state: InvestmentState) -> dict:
    """
    ① Orchestrator LangGraph 노드.

    실행 흐름:
      1. iteration_count를 읽어 모드를 결정한다 (초기 지시 / 피드백 대응 / Fallback).
      2. risk_assessment가 있으면 피드백으로 해석한다.
      3. LLM(또는 Mock)에게 CIO 의사결정을 요청한다.
      4. target_ticker, analysis_tasks, iteration_count를 업데이트한다.

    Returns:
        {
            "target_ticker": str,
            "analysis_tasks": list,
            "iteration_count": int,
            "orchestrator_directives": dict,  # CIO 지시 내용 전체
        }
    """
    iteration = state.get("iteration_count", 0)
    user_request = state.get("user_request", "")

    print(f"\n{'=' * 60}")
    print(f"① ORCHESTRATOR (CIO/PM)  —  iteration #{iteration}")
    print(f"{'=' * 60}")
    print(f"   [입력] 사용자 요청: {user_request}")

    # 피드백 해석
    risk_feedback: Optional[dict] = None
    if iteration > 0:
        risk_assessment = state.get("risk_assessment", {})
        risk_decision = risk_assessment.get("risk_decision", risk_assessment)
        risk_feedback = risk_decision if risk_decision else None
        if risk_feedback:
            fb = risk_feedback.get("orchestrator_feedback", {})
            print(f"   [피드백 수신] required={fb.get('required')}")
            print(f"   [피드백 수신] reasons={fb.get('reasons', [])}")
            print(f"   [피드백 수신] detail={str(fb.get('detail', ''))[:120]}")

    # 모드 표시
    if iteration == 0:
        mode = "INITIAL_DELEGATION"
    elif iteration >= MAX_ITERATIONS:
        mode = "FALLBACK_ABORT"
    else:
        mode = "POST_RISK_REJECT"
    print(f"   [모드] {mode}  (iteration={iteration}, max={MAX_ITERATIONS})")

    # LLM/Mock 호출
    print(f"   [LLM] CIO 의사결정 요청 중...")
    portfolio_context = state.get("portfolio_context", {})
    book_context = _build_book_context_summary(state)
    decision = _call_llm(user_request, iteration, risk_feedback, portfolio_context, book_context)
    decision = _apply_portfolio_mandate(decision, portfolio_context)
    active_ideas = _build_active_ideas_registry(state, decision)
    book_allocation_plan, capital_competition = _build_book_allocation_plan(state, decision, active_ideas)
    conviction_by_ticker = (
        book_allocation_plan.get("conviction_by_ticker", {})
        if isinstance(book_allocation_plan, dict)
        else {}
    )
    allocation_signals_by_ticker = (
        book_allocation_plan.get("allocation_signals_by_ticker", {})
        if isinstance(book_allocation_plan, dict)
        else {}
    )
    portfolio_memory = _build_portfolio_memory(
        state,
        decision,
        active_ideas,
        conviction_by_ticker,
        allocation_signals_by_ticker,
    )
    monitoring_backlog = _build_monitoring_backlog(state, decision, active_ideas)
    current_book_exists = bool(_normalize_weight_map(state.get("positions_final") or state.get("positions_proposed") or {}))
    decision["book_context_summary"] = book_context
    decision["active_idea_count"] = len(active_ideas)
    decision["open_review_count"] = len(monitoring_backlog)
    if book_allocation_plan:
        decision["allocator_guidance"] = {
            "target_gross_exposure": book_allocation_plan.get("gross_target"),
            "single_name_cap": book_allocation_plan.get("single_name_cap"),
            "allocator_source": book_allocation_plan.get("allocator_source"),
            "quality_haircut": book_allocation_plan.get("quality_haircut"),
        }
        decision["capital_competition_preview"] = capital_competition[:5]

    action = decision.get("action_type", "initial_delegation")
    brief = decision.get("investment_brief", {})
    universe = brief.get("target_universe", [])
    ticker = universe[0] if universe else _extract_ticker(user_request)
    tasks = ["macro_analysis", "fundamental_analysis", "sentiment_analysis", "technical_analysis"]

    print(f"   [결정] action_type: {action}")
    print(f"   [결정] target_universe: {universe}")
    print(f"   [결정] rationale: {brief.get('rationale', '')[:120]}...")

    desk_tasks = decision.get("desk_tasks", {})
    for desk, task in desk_tasks.items():
        print(f"   [데스크:{desk}] horizon={task.get('horizon_days')}d, focus={task.get('focus_areas', [])}")
    mandate_meta = decision.get("portfolio_mandate", {})
    if isinstance(mandate_meta, dict) and mandate_meta.get("applied"):
        print(f"   [mandate] changes={mandate_meta.get('changes', [])}")
        print(f"   [mandate] final_universe={mandate_meta.get('final_universe', universe)}")
    if active_ideas:
        print(f"   [book] active_ideas={len(active_ideas)} open_reviews={len(monitoring_backlog)}")
    if book_allocation_plan:
        print(
            "   [allocator] gross_target="
            f"{book_allocation_plan.get('gross_target')} top_mix={book_allocation_plan.get('weights_relative', {})}"
        )

    result_positions: dict[str, float] = {}
    if current_book_exists and isinstance(book_allocation_plan, dict):
        result_positions = {
            str(t).strip().upper(): float(w)
            for t, w in (book_allocation_plan.get("weights_relative", {}) or {}).items()
        }

    if action == "fallback_abort":
        print(f"   ⚠️  FALLBACK: 신규 매수 포기. 현금 관망 또는 방어 대안 제시.")
    else:
        print(f"   → 4개 데스크에 분석 태스크를 위임합니다.")

    return {
        "target_ticker": ticker,
        "analysis_tasks": tasks,
        "iteration_count": iteration + 1,
        "orchestrator_directives": decision,
        "active_ideas": active_ideas,
        "portfolio_memory": portfolio_memory,
        "monitoring_backlog": monitoring_backlog,
        "book_allocation_plan": book_allocation_plan,
        "capital_competition": capital_competition,
        "positions_proposed": result_positions,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# __main__ — Mock 시뮬레이션
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    SEP = "=" * 65
    DASH = "─" * 65

    print(SEP)
    print("① ORCHESTRATOR (CIO/PM) — Mock 시뮬레이션")
    print(SEP)

    # ══════════════════════════════════════════════════════════════════════
    # 시나리오 1: 초기 위임 (iteration=0)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{DASH}")
    print("📋 시나리오 1: 초기 위임 — 사용자 최초 요청")
    print(DASH)

    state_initial: InvestmentState = {
        "user_request": "AI 주식 AAPL 지금 사도 돼?",
        "target_ticker": "",
        "analysis_tasks": [],
        "macro_analysis": {},
        "fundamental_analysis": {},
        "sentiment_analysis": {},
        "technical_analysis": {},
        "risk_assessment": {},
        "final_report": "",
        "iteration_count": 0,
    }

    result1 = orchestrator_node(state_initial)
    print(f"\n{DASH}")
    print("📄 시나리오 1 — OrchestratorOutput JSON")
    print(DASH)
    print(json.dumps(result1.get("orchestrator_directives", {}), indent=2, ensure_ascii=False))

    # ══════════════════════════════════════════════════════════════════════
    # 시나리오 2: Fallback (iteration=3, CVaR 지속 초과)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n\n{DASH}")
    print("📋 시나리오 2: Fallback — iteration=3, Risk Manager CVaR 지속 초과")
    print(DASH)

    mock_risk_feedback = {
        "per_ticker_decisions": {
            "AAPL": {
                "final_weight": 0.02,
                "decision": "reduce",
                "flags": ["cvar_limit_breach", "macro_headwind", "stress_test_violation"],
                "rationale_short": (
                    "포트폴리오 CVaR 및 스트레스 테스트 손실이 3회 연속 한도 초과. "
                    "비중을 최소화했으나 여전히 한도 위반."
                ),
            }
        },
        "portfolio_actions": {
            "hedge_recommendations": [
                {
                    "type": "index_hedge",
                    "direction": "short",
                    "notional_suggestion": 0.15,
                    "reason": "CVaR 한도 초과 방어",
                }
            ],
            "gross_net_adjustment": {
                "target_gross_exposure": 0.8,
                "target_net_exposure": 0.1,
                "reason": "스트레스 테스트 손실 한도 방어를 위한 Net Exposure 축소",
            },
        },
        "orchestrator_feedback": {
            "required": True,
            "reasons": ["stress_test_violation", "portfolio_risk_violation"],
            "detail": (
                "포트폴리오 CVaR 0.0230 > 한도 0.0150. "
                "스트레스 테스트 손실 12.3% > 한도 10.0%. "
                "3회 반복에도 리스크 한도 내 진입 불가."
            ),
        },
    }

    state_fallback: InvestmentState = {
        "user_request": "AI 주식 AAPL 지금 사도 돼?",
        "target_ticker": "AAPL",
        "analysis_tasks": ["macro_analysis", "fundamental_analysis", "sentiment_analysis", "technical_analysis"],
        "macro_analysis": {"regime": "recession", "gdp_growth": -0.5},
        "fundamental_analysis": {"sector": "Technology", "risk_flags": []},
        "sentiment_analysis": {"overall_sentiment": "negative", "sentiment_score": -0.35},
        "technical_analysis": {"decision": "LONG", "final_allocation_pct": 0.02},
        "risk_assessment": {"grade": "High", "risk_decision": mock_risk_feedback},
        "final_report": "",
        "iteration_count": 3,
    }

    result2 = orchestrator_node(state_fallback)
    print(f"\n{DASH}")
    print("📄 시나리오 2 — OrchestratorOutput JSON (Fallback)")
    print(DASH)
    print(json.dumps(result2.get("orchestrator_directives", {}), indent=2, ensure_ascii=False))

    # ══════════════════════════════════════════════════════════════════════
    # 검증 Summary
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n\n{SEP}")
    print("✅ 시뮬레이션 검증 요약")
    print(SEP)

    d1 = result1.get("orchestrator_directives", {})
    d2 = result2.get("orchestrator_directives", {})

    checks = [
        ("시나리오1: action_type == 'initial_delegation'",
         d1.get("action_type") == "initial_delegation"),
        ("시나리오1: 4개 데스크 task 모두 존재",
         all(k in d1.get("desk_tasks", {}) for k in ("macro", "fundamental", "sentiment", "quant"))),
        ("시나리오1: target_universe에 AAPL 포함",
         "AAPL" in d1.get("investment_brief", {}).get("target_universe", [])),
        ("시나리오2: action_type == 'fallback_abort'",
         d2.get("action_type") == "fallback_abort"),
        ("시나리오2: target_universe에 방어 자산 포함",
         any(t in d2.get("investment_brief", {}).get("target_universe", [])
             for t in ("CASH", "SHY", "TLT", "XLU"))),
        ("시나리오2: rationale에 매수 포기 명시",
         "포기" in d2.get("investment_brief", {}).get("rationale", "")),
    ]

    all_pass = True
    for label, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_pass = False
        print(f"   {status}  {label}")

    print(f"\n{'✅ 모든 검증 통과!' if all_pass else '❌ 일부 검증 실패 — 위 항목 확인 필요'}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Research Planner (V4) — research_router_node에서만 호출
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_QUERY_TEMPLATES = {
    "press_release_or_ir": "{ticker} investor relations earnings press release date",
    "macro_headline_context": "{ticker} macro economy headline driver {query}",
    "ownership_identity": "{ticker} insider trading institutional ownership SEC Form 4 13F",
    "valuation_context": "{ticker} PE ratio valuation historical comparison peers",
    "catalyst_event_detail": "{ticker} catalyst event upcoming {query}",
    "sec_filing": "{ticker} SEC filing 10-K 10-Q annual report",
}

_ALLOWLIST_BY_KIND = {
    "ownership_identity": ["sec.gov", "efts.sec.gov"],
    "sec_filing": ["sec.gov", "efts.sec.gov"],
    "press_release_or_ir": ["sec.gov", "prnewswire.com", "businesswire.com", "globenewswire.com"],
    "macro_headline_context": [
        "fred.stlouisfed.org", "nyfed.org", "bls.gov", "bea.gov", "eia.gov",
        "federalreserve.gov", "treasury.gov",
    ],
}


def _request_key(req: dict) -> tuple:
    return (
        req.get("desk", ""),
        req.get("kind", ""),
        req.get("ticker", ""),
        req.get("series_id", ""),
        req.get("query", ""),
    )


def _sanitize_research_request(req: dict) -> dict | None:
    kind = str(req.get("kind", "")).strip()
    if not kind:
        return None
    out = dict(req)
    out["kind"] = kind
    out["desk"] = str(out.get("desk", "orchestrator") or "orchestrator")
    out["ticker"] = str(out.get("ticker", "")).upper().strip()
    out["query"] = str(out.get("query", "")).strip()
    out["priority"] = max(1, min(5, int(out.get("priority", 3))))
    out["recency_days"] = max(1, min(365, int(out.get("recency_days", 30))))
    out["max_items"] = max(1, min(20, int(out.get("max_items", 5))))
    out["rationale"] = str(out.get("rationale", "")).strip()
    hint = out.get("_allowlist_hint")
    if hint and not isinstance(hint, list):
        out["_allowlist_hint"] = []
    return out


def _enforce_research_budget(
    requests: list[dict],
    *,
    max_web_queries_per_run: int,
    max_web_queries_per_ticker: int,
) -> list[dict]:
    capped = []
    ticker_count: dict[str, int] = {}
    for req in sorted(requests, key=lambda r: r.get("priority", 5)):
        if len(capped) >= max_web_queries_per_run:
            break
        t = (req.get("ticker", "") or "__GLOBAL__").upper()
        if ticker_count.get(t, 0) >= max_web_queries_per_ticker:
            continue
        capped.append(req)
        ticker_count[t] = ticker_count.get(t, 0) + 1
    return capped


def plan_additional_research(
    evidence_requests: list,
    desk_outputs: dict,
    user_request: str,
    *,
    policy_state: dict | None = None,
) -> list | None:
    """
    Orchestrator가 EvidenceRequest를 재정렬/보강.
    LLM 사용 가능하면 LLM-first, 실패시 규칙 fallback.
    iteration_count 변경 금지.
    Returns: enriched requests list, or None (= use original).
    """
    # ── 먼저 규칙 기반으로 query template 보강 ──────────────────────────
    enriched = []
    max_per_run = int((policy_state or {}).get("max_web_queries_per_run", 6))
    max_per_ticker = int((policy_state or {}).get("max_web_queries_per_ticker", 3))
    for req in evidence_requests:
        req = dict(req)  # copy
        kind = req.get("kind", "web_search")
        ticker = req.get("ticker", "")
        query = req.get("query", "")

        # query가 비어있으면 template으로 채움
        if not query and kind in _QUERY_TEMPLATES:
            req["query"] = _QUERY_TEMPLATES[kind].format(ticker=ticker, query="")

        # allowlist 힌트 추가 (resolver가 사용)
        if kind in _ALLOWLIST_BY_KIND:
            req["_allowlist_hint"] = _ALLOWLIST_BY_KIND[kind]

        sreq = _sanitize_research_request(req)
        if sreq:
            enriched.append(sreq)

    # dedupe + 우선순위 정렬
    deduped = []
    seen = set()
    for req in sorted(enriched, key=lambda r: r.get("priority", 5)):
        k = _request_key(req)
        if k in seen:
            continue
        seen.add(k)
        deduped.append(req)
    enriched = deduped

    # ── LLM enrichment (optional) ────────────────────────────────────
    try:
        if HAS_LC and HAS_PYDANTIC:
            from llm.router import get_llm_with_cache
            human_msg = (
                f"Given these evidence requests: {json.dumps(enriched[:5], ensure_ascii=False)}\n"
                f"User request: {user_request}\n"
                f"Add up to 2 additional high-priority requests if needed. "
                f"Return JSON array of EvidenceRequest objects. "
                f"Only use allowlisted domains."
            )
            llm, cached = get_llm_with_cache("orchestrator", human_msg)
            if cached:
                # Validate and merge
                if isinstance(cached, list):
                    for item in cached[:2]:
                        if isinstance(item, dict):
                            sitem = _sanitize_research_request(item)
                            if sitem:
                                enriched.append(sitem)
            elif llm:
                msgs = [
                    SystemMessage(content="You are a research planner. Return JSON array only."),
                    HumanMessage(content=human_msg),
                ]
                resp = llm.invoke(msgs)
                try:
                    extra = json.loads(resp.content)
                    if isinstance(extra, list):
                        for item in extra[:2]:
                            if isinstance(item, dict):
                                sitem = _sanitize_research_request(item)
                                if sitem:
                                    enriched.append(sitem)
                except (json.JSONDecodeError, TypeError):
                    pass  # LLM output 파싱 실패 → rules only
    except Exception:
        pass  # LLM 사용 불가 → rules fallback

    # re-dedupe + budget cap (fallback if over-budget/invalid domains)
    final_reqs = []
    seen = set()
    for req in sorted(enriched, key=lambda r: r.get("priority", 5)):
        k = _request_key(req)
        if k in seen:
            continue
        seen.add(k)
        if req.get("kind") in _ALLOWLIST_BY_KIND:
            req["_allowlist_hint"] = _ALLOWLIST_BY_KIND[req["kind"]]
        final_reqs.append(req)

    final_reqs = _enforce_research_budget(
        final_reqs,
        max_web_queries_per_run=max_per_run,
        max_web_queries_per_ticker=max_per_ticker,
    )
    return final_reqs if final_reqs else None
