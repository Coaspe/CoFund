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

import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field, field_validator


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Evidence 시스템 (Iron Rule R2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EvidenceItem(BaseModel):
    """모든 수치적 주장의 추적 가능한 근거."""
    source_type: Literal["api", "file", "database", "model"] = "model"
    source_name: str = Field(description='출처 (예: "FRED:T10Y2Y", "yfinance", "mock")')
    as_of: str = Field(description="ISO 8601 UTC timestamp")
    metric: str = Field(description='측정 항목 (예: "gdp_growth", "z_score")')
    value: Any = Field(description="값 (숫자/문자열/부울)")
    quality_score: float = Field(
        ge=0.0, le=1.0,
        description="데이터 품질 (0=최저, 1=최고). mock=0.3, live_delayed=0.7, realtime=1.0",
    )
    note: Optional[str] = Field(default=None, description="짧은 주의사항")


class RiskFlag(BaseModel):
    """리스크 플래그 — Risk Manager Gate3+ 에서 소비."""
    code: str = Field(description='예: "default_risk", "going_concern"')
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    description: str = ""
    linked_metrics: List[str] = Field(default_factory=list)


class DataQuality(BaseModel):
    """에이전트 출력의 데이터 품질 메타데이터."""
    missing_pct: float = Field(default=0.0, ge=0.0, le=1.0, description="결측 필드 비율")
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
    return {
        "source_type": source_type,
        "source_name": source_name,
        "as_of": as_of or datetime.now(timezone.utc).isoformat(),
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


class InvestmentState(TypedDict, total=False):
    """LangGraph 공유 상태 (V3). 멀티 종목 + 포지션 추적 + 감사 지원."""
    # Run context
    run_id: str
    as_of: str
    mode: str
    run_context: dict  # {run_id, as_of, mode, seed, git_commit, config_hash}

    # Universe & positions
    universe: List[str]          # 분석 대상 티커 목록
    positions_proposed: dict     # Allocator 출력: {ticker -> weight}
    positions_final: dict        # Risk 최종 확정: {ticker -> weight}

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
        run_context={
            "run_id": run_id,
            "as_of": as_of,
            "mode": mode,
            "seed": seed,
        },
        universe=universe or [],
        positions_proposed={},
        positions_final={},
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
        audit={"paths": {}, "gate_trace": [], "validations": []},
    )
