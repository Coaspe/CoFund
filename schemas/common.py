"""
schemas/common.py — 공통 Pydantic 스키마 + Evidence 시스템
==========================================================
CHANGELOG:
  v2.0 (2026-02-22) — 전면 재설계. EvidenceItem(metric 기반), RiskFlag,
                       BaseAnalystOutput, 에이전트별 Output 서브클래스,
                       InvestmentState V2 (Annotated reducers).

Iron Rules Enforced:
  R0: No Trading
  R2: No Evidence, No Trade
  R5: Sentiment tilt cap [0.7, 1.3]
  R7: Auditability (run_id, generated_at, as_of)
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Evidence 시스템 (Iron Rule R2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EvidenceItem(BaseModel):
    """수치/웹 근거를 모두 포괄하는 통합 Evidence 스키마."""
    # ── Web research fields (primary for evidence_store) ──────────────────
    url: str = ""
    title: str = ""
    published_at: str = ""
    snippet: str = ""
    source: str = ""
    retrieved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    hash: str = ""
    kind: str = ""
    desk: str = ""
    ticker: str = ""
    trust_tier: float = Field(default=0.4, ge=0.0, le=1.0)
    resolver_path: str = ""
    # ── Legacy metric evidence fields (backward compatible) ───────────────
    source_type: Literal["api", "file", "database", "model"] = "model"
    source_name: str = Field(default="", description='출처 (예: "FRED:T10Y2Y", "yfinance", "mock")')
    as_of: str = Field(default="", description="ISO 8601 UTC timestamp")
    metric: str = Field(default="", description='측정 항목 (예: "gdp_growth", "z_score")')
    value: Any = Field(default=None, description="값 (숫자/문자열/부울)")
    quality_score: float = Field(
        ge=0.0, le=1.0,
        description="데이터 품질 (0=최저, 1=최고). mock=0.3, live_delayed=0.7, realtime=1.0",
    )
    note: Optional[str] = Field(default=None, description="짧은 주의사항")


class EvidenceRequest(BaseModel):
    """웹 리서치 요청 단위."""
    desk: Literal["macro", "fundamental", "sentiment", "orchestrator"]
    kind: str
    ticker: Optional[str] = None
    series_id: Optional[str] = None
    query: Optional[str] = None
    priority: int = Field(default=3, ge=1, le=5)
    recency_days: int = Field(default=30, ge=1, le=365)
    max_items: int = Field(default=5, ge=1, le=20)
    rationale: str = ""

    @field_validator("query")
    @classmethod
    def _strip_query(cls, v: str | None) -> str | None:
        if v is None:
            return None
        q = v.strip()
        return q or None

    @field_validator("rationale")
    @classmethod
    def _strip_rationale(cls, v: str) -> str:
        return (v or "").strip()

    @field_validator("series_id")
    @classmethod
    def _strip_series_id(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip()
        return s or None

    @field_validator("ticker")
    @classmethod
    def _strip_ticker(cls, v: str | None) -> str | None:
        if v is None:
            return None
        t = v.strip().upper()
        return t or None

    @model_validator(mode="after")
    def _validate_target_fields(self):
        if not (self.ticker or self.series_id or self.query):
            raise ValueError("EvidenceRequest requires one of ticker/series_id/query")
        return self


class RiskFlag(BaseModel):
    """리스크 플래그 — Risk Manager Gate3+ 에서 소비."""
    code: str = Field(description='예: "default_risk", "going_concern"')
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    description: str = ""
    linked_metrics: List[str] = Field(default_factory=list)


class DataQuality(BaseModel):
    """에이전트 출력의 데이터 품질 메타데이터."""
    missing_pct: float = Field(default=0.0, ge=0.0, le=1.0, description="결측 필드 비율")
    freshness_days: float = Field(default=999.0, ge=0.0, description="신선도(경과 일수)")
    warnings: List[str] = Field(default_factory=list, description="경고 목록")
    delay_hrs: float = Field(default=0.0, ge=0.0, description="데이터 지연 시간(시간)")
    anomaly_flags: List[str] = Field(default_factory=list, description="이상 플래그 (예: lookahead_violation)")
    source_timestamps: Dict[str, str] = Field(default_factory=dict, description="소스별 타임스탬프")
    is_mock: bool = False


class Assumption(BaseModel):
    """에이전트 결정의 핵심 가정."""
    key: str
    text: str
    value: Any = None


class ReviewTrigger(BaseModel):
    """결정 재검토 조건."""
    condition: str = Field(description="재검토가 필요한 조건")
    action: str = Field(description="권장 대응 액션")


def make_evidence(
    metric: str,
    value: Any,
    source_name: str = "mock",
    source_type: str = "model",
    quality: float = 0.3,
    note: str | None = None,
    as_of: str | None = None,
) -> dict:
    """EvidenceItem을 dict로 빠르게 생성."""
    ts = as_of or datetime.now(timezone.utc).isoformat()
    hraw = f"{source_name}|{metric}|{ts}|{value}"
    return {
        "url": "",
        "title": metric,
        "published_at": ts,
        "snippet": "",
        "source": source_name,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "hash": hashlib.sha256(hraw.encode("utf-8")).hexdigest()[:16],
        "kind": "metric",
        "desk": "",
        "ticker": "",
        "trust_tier": quality,
        "resolver_path": "legacy_metric",
        "source_type": source_type,
        "source_name": source_name,
        "as_of": ts,
        "metric": metric,
        "value": value,
        "quality_score": quality,
        "note": note,
    }


def make_risk_flag(
    code: str,
    severity: str = "medium",
    description: str = "",
    linked_metrics: list | None = None,
) -> dict:
    """RiskFlag을 dict로 빠르게 생성."""
    return {
        "code": code,
        "severity": severity,
        "description": description,
        "linked_metrics": linked_metrics or [],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Safe helpers (Bug fix: 0.0 weight / signed weight / disagreement)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def first_not_none(d: dict, keys: list[str], default: Any = 0.0) -> Any:
    """
    Return the first value from d[keys[0]], d[keys[1]], ... that is not None.

    CRITICAL: 0.0, 0, False are VALID values and are returned as-is.
    Only None triggers fallback to the next key or the default.
    """
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return default


def compute_signed_weight(decision: str, allocation: float | None) -> float:
    """
    Convert quant decision + unsigned allocation to signed weight.

    SHORT/SELL → negative, LONG/BUY → positive, HOLD/CLEAR/NEUTRAL → 0.0
    """
    if allocation is None:
        return 0.0
    decision_upper = (decision or "").upper().strip()
    if decision_upper in ("SHORT", "SELL"):
        return -abs(allocation)
    if decision_upper in ("LONG", "BUY"):
        return abs(allocation)
    # HOLD, CLEAR, NEUTRAL, or unknown
    return 0.0


def compute_disagreement_score(desk_outputs: dict[str, dict]) -> float:
    """
    Compute disagreement score across desk analysts.

    Maps primary_decision to [-1, 0, +1] and weights by confidence.
    Returns 0.0 (perfect agreement) to 1.0 (max disagreement).
    """
    _direction_map = {
        "bullish": 1.0, "bearish": -1.0,
        "neutral": 0.0, "hold": 0.0, "no_trade": 0.0, "avoid": -0.5,
    }
    signals = []
    for desk_name, output in desk_outputs.items():
        if not output or output.get("status") == "skipped":
            continue
        pd_val = output.get("primary_decision", "neutral")
        conf = output.get("confidence", 0.5)
        direction = _direction_map.get(pd_val, 0.0)
        signals.append(direction * conf)

    if len(signals) < 2:
        return 0.0

    mean_signal = sum(signals) / len(signals)
    variance = sum((s - mean_signal) ** 2 for s in signals) / len(signals)
    # Normalize: max variance ≈ 1.0 (when signals span -1 to +1)
    return round(min(variance ** 0.5, 1.0), 3)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BaseAnalystOutput (공통 베이스)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BaseAnalystOutput(BaseModel):
    """모든 데스크 에이전트 출력의 공통 필드."""
    agent_type: str
    run_id: str = ""
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    as_of: str = ""
    ticker: str = ""
    horizon_days: int = 30
    primary_decision: Literal[
        "bullish", "neutral", "bearish", "avoid", "hold", "no_trade"
    ] = "neutral"
    recommendation: Literal[
        "allow", "allow_with_limits", "reject", "no_trade"
    ] = "allow_with_limits"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    signal_strength: float = Field(default=0.5, ge=0.0, le=1.0, description="신호 강도 (0=약, 1=강)")
    risk_flags: List[RiskFlag] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    data_quality: DataQuality = Field(default_factory=DataQuality)
    assumptions: List[Assumption] = Field(default_factory=list)
    review_triggers: List[ReviewTrigger] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    data_ok: bool = True
    summary: str = ""
    status: Literal["ok", "skipped", "error", "insufficient"] = "ok"

    @field_validator("confidence")
    @classmethod
    def lower_if_bad_data(cls, v: float, info) -> float:
        data = info.data if hasattr(info, "data") else {}
        if not data.get("data_ok", True) and v > 0.5:
            return min(v, 0.4)
        return v


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 에이전트별 출력 모델
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MacroOutput(BaseAnalystOutput):
    """② Macro Analyst 출력."""
    agent_type: str = "macro"
    macro_regime: str = "normal"
    overlay_guidance: dict = Field(default_factory=dict)
    tail_risk_warning: bool = False
    recommended_constraints: dict = Field(default_factory=dict)
    # raw engine results
    indicators: dict = Field(default_factory=dict)


class FundamentalOutput(BaseAnalystOutput):
    """③ Fundamental Analyst 출력."""
    agent_type: str = "fundamental"
    valuation_block: dict = Field(default_factory=dict)
    quality_block: dict = Field(default_factory=dict)
    structural_risk_flag: bool = False
    hard_red_flags: List[RiskFlag] = Field(default_factory=list)
    soft_flags: List[RiskFlag] = Field(default_factory=list)
    notes_for_risk_manager: str = ""


class SentimentOutput(BaseAnalystOutput):
    """④ Sentiment Analyst 출력."""
    agent_type: str = "sentiment"
    sentiment_regime: str = "neutral"
    positioning_crowding: str = "balanced"
    volatility_regime: str = "normal"
    entry_timing_signal: str = "neutral"
    catalysts: List[str] = Field(default_factory=list)
    tilt_factor: float = Field(default=1.0, ge=0.7, le=1.3)
    tactical_notes: str = ""


class QuantAnalystOutput(BaseAnalystOutput):
    """⑤ Quant Analyst 출력 — 기존 flat fields 하위호환."""
    agent_type: str = "quant"
    decision: str = "HOLD"
    final_allocation_pct: float = 0.0
    z_score: Optional[float] = None
    regime_2_high_vol: Optional[float] = None
    asset_cvar_99_daily: Optional[float] = None
    quant_payload: Optional[dict] = None
    llm_decision: Optional[dict] = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# InvestmentState V2 — LangGraph 공유 상태
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _merge_dicts(a: dict, b: dict) -> dict:
    merged = dict(a) if a else {}
    if b:
        merged.update(b)
    return merged


def _merge_lists(a: list, b: list) -> list:
    return (a or []) + (b or [])


def _normalize_query_text(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _request_key(req: dict) -> tuple:
    return (
        str((req or {}).get("desk", "")).strip().lower(),
        str((req or {}).get("kind", "")).strip().lower(),
        str((req or {}).get("ticker", "")).strip().upper(),
        str((req or {}).get("series_id", "")).strip(),
        _normalize_query_text((req or {}).get("query", "")),
    )


def _merge_evidence_requests(a: list, b: list) -> list:
    merged = []
    seen = set()
    for req in (a or []) + (b or []):
        key = _request_key(req if isinstance(req, dict) else {})
        if key in seen:
            continue
        seen.add(key)
        merged.append(req)
    return merged


class InvestmentState(TypedDict, total=False):
    """LangGraph 공유 상태 (V3). 멀티 종목 + 포지션 추적 + 감사 지원."""
    # Run context
    run_id: str
    as_of: str
    mode: str
    intent: str  # canonical workflow intent: market_outlook | single_name | hedge_design | relative_value ...
    scenario_tags: list  # planner nuance tags: event_risk | overheated_check | ...
    output_language: str  # ko | en
    analysis_execution_mode: str  # single_main | B_main_plus_hedge_lite
    workflow_kind: str  # general | position_review
    run_context: dict  # {run_id, as_of, mode, seed, git_commit, config_hash}

    # Universe & positions
    universe: List[str]          # 분석 대상 티커 목록
    asset_type_by_ticker: dict   # {ticker -> ETF|EQUITY|INDEX|...}
    hedge_lite: dict             # B 모드 hedge lite 분석 산출물
    positions_proposed: dict     # Allocator 출력: {ticker -> weight}
    positions_final: dict        # Risk 최종 확정: {ticker -> weight}
    active_ideas: dict           # Orchestrator snapshot: {ticker -> idea status/meta}
    portfolio_memory: dict       # Persistent thesis/position memory across runs
    monitoring_backlog: list     # Open review/monitoring tasks derived by orchestrator
    book_allocation_plan: dict   # Book-level pre-allocation plan from orchestrator
    capital_competition: list    # Ranked capital competition rows across current/new ideas
    portfolio_construction_analysis: dict  # Construction quant output
    event_calendar: list         # First-class normalized event/trigger calendar across desks
    monitoring_actions: dict     # Monitoring/escalation router output
    decision_quality_scorecard: dict  # Desk/source quality snapshot for rerun prioritization
    question_understanding: dict  # Frontdoor question classification/extraction output
    portfolio_intake: dict       # Structured portfolio intake extracted from natural language
    normalized_portfolio_snapshot: dict  # Deterministic holdings normalization snapshot

    # Legacy
    user_request: str
    target_ticker: str
    analysis_tasks: list

    portfolio_context: dict

    orchestrator_directives: dict
    macro_analysis: dict
    fundamental_analysis: dict
    sentiment_analysis: dict
    technical_analysis: dict
    risk_assessment: dict

    final_report: str

    iteration_count: int
    completed_tasks: Annotated[dict, _merge_dicts]
    errors: Annotated[list, _merge_lists]
    trace: Annotated[list, _merge_lists]
    task_backlog: Annotated[list, _merge_lists]
    react_log: Annotated[list, _merge_lists]
    evidence_requests: Annotated[list, _merge_evidence_requests]
    evidence_store: Annotated[dict, _merge_dicts]
    evidence_score: int
    research_round: int
    max_research_rounds: int
    last_research_delta: int
    research_stop_reason: str
    user_action_required: bool
    user_action_items: Annotated[list, _merge_lists]
    _run_research: bool
    _research_plan: list
    _swarm_candidates: list
    _swarm_plan: list
    _rerun_plan: dict
    _executed_requests: list
    _evidence_delta_kinds: dict
    _frontdoor_prepared: bool

    # Audit
    audit: dict  # {paths, gate_trace, validations}


def generate_run_id() -> str:
    return str(uuid.uuid4())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_initial_state(
    user_request: str,
    mode: str = "mock",
    portfolio_context: dict | None = None,
    universe: list | None = None,
    seed: int | None = None,
) -> InvestmentState:
    """초기 InvestmentState 팩토리."""
    run_id = generate_run_id()
    as_of = now_iso()
    return InvestmentState(
        run_id=run_id,
        as_of=as_of,
        mode=mode,
        intent="single_name",
        scenario_tags=[],
        output_language="ko",
        analysis_execution_mode="single_main",
        workflow_kind="general",
        run_context={
            "run_id": run_id,
            "as_of": as_of,
            "mode": mode,
            "seed": seed,
        },
        universe=universe or [],
        asset_type_by_ticker={},
        hedge_lite={},
        positions_proposed={},
        positions_final={},
        active_ideas={},
        portfolio_memory={},
        monitoring_backlog=[],
        book_allocation_plan={},
        capital_competition=[],
        portfolio_construction_analysis={},
        event_calendar=[],
        monitoring_actions={},
        decision_quality_scorecard={},
        question_understanding={},
        portfolio_intake={},
        normalized_portfolio_snapshot={},
        user_request=user_request,
        target_ticker="",
        analysis_tasks=[],
        portfolio_context=portfolio_context or {},
        orchestrator_directives={},
        macro_analysis={},
        fundamental_analysis={},
        sentiment_analysis={},
        technical_analysis={},
        risk_assessment={},
        final_report="",
        iteration_count=0,
        completed_tasks={},
        errors=[],
        trace=[],
        task_backlog=[],
        react_log=[],
        evidence_requests=[],
        evidence_store={},
        evidence_score=0,
        research_round=0,
        max_research_rounds=2,
        last_research_delta=0,
        research_stop_reason="",
        user_action_required=False,
        user_action_items=[],
        _run_research=False,
        _research_plan=[],
        _swarm_candidates=[],
        _swarm_plan=[],
        _rerun_plan={},
        _executed_requests=[],
        _evidence_delta_kinds={},
        _frontdoor_prepared=False,
        audit={
            "paths": {},
            "gate_trace": [],
            "validations": [],
            "research": {"web_queries_total": 0, "web_queries_by_ticker": {}},
        },
    )
