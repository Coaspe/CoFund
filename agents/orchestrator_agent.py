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
    return "\n".join(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM 호출 (실제 / Mock)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _call_llm(
    user_request: str,
    iteration: int,
    risk_feedback: Optional[dict] = None,
) -> dict:
    """LLM 호출 → OrchestratorOutput. API 키 없으면 규칙 기반 Mock."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key or not HAS_LC:
        print("   [LLM] API 키 없음 → 규칙 기반 Mock 결정")
        return _mock_orchestrator_decision(user_request, iteration, risk_feedback)

    try:
        from langchain_openai import ChatOpenAI  # type: ignore
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=key)
        human_msg = _build_orchestrator_human_msg(user_request, iteration, risk_feedback)

        if HAS_PYDANTIC:
            structured = llm.with_structured_output(OrchestratorOutput)
            msgs = [
                SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
                HumanMessage(content=human_msg),
            ]
            resp = structured.invoke(msgs)
            return resp.model_dump()
        else:
            msgs = [
                SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
                HumanMessage(content=human_msg),
            ]
            raw = llm.invoke(msgs)
            return json.loads(raw.content)
    except Exception as exc:
        print(f"   [LLM] ⚠️ 호출 실패 → Mock: {exc}")
        return _mock_orchestrator_decision(user_request, iteration, risk_feedback)


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
    return "AAPL"  # 기본값


def _mock_orchestrator_decision(
    user_request: str,
    iteration: int,
    risk_feedback: Optional[dict] = None,
) -> dict:
    """iteration 기반 CIO 의사결정 Mock."""

    ticker = _extract_ticker(user_request)

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
    return {
        "current_iteration": iteration,
        "action_type": "initial_delegation",
        "investment_brief": {
            "rationale": (
                f"사용자 요청 '{user_request}'을 분석하여 {ticker} 중심의 "
                f"롱 포지션 검토를 개시. 투자 기간 중기(1~6개월). "
                f"4개 데스크에 종합 분석을 지시."
            ),
            "target_universe": [ticker],
        },
        "desk_tasks": {
            "macro": {
                "horizon_days": 30,
                "focus_areas": [
                    "금리 인하 확률",
                    "테크 섹터 베타 환경",
                    "GDP/CPI 추세",
                ],
            },
            "fundamental": {
                "horizon_days": 90,
                "focus_areas": [
                    "부채비율 및 유동성",
                    "현금흐름 안정성",
                    "밸류에이션 적정성 (PER, PBR)",
                ],
            },
            "sentiment": {
                "horizon_days": 5,
                "focus_areas": [
                    "옵션 스큐(Skew)",
                    "실적 발표 관련 뉴스 감성",
                    "내부자 매매 동향",
                ],
            },
            "quant": {
                "horizon_days": 10,
                "risk_budget": "Moderate",
                "focus_areas": [
                    "평균 회귀 Z-score",
                    "CVaR 한도 내 최적 비중",
                    "모멘텀 점수",
                ],
            },
        },
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
    decision = _call_llm(user_request, iteration, risk_feedback)

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

    if action == "fallback_abort":
        print(f"   ⚠️  FALLBACK: 신규 매수 포기. 현금 관망 또는 방어 대안 제시.")
    else:
        print(f"   → 4개 데스크에 분석 태스크를 위임합니다.")

    return {
        "target_ticker": ticker,
        "analysis_tasks": tasks,
        "iteration_count": iteration + 1,
        "orchestrator_directives": decision,
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
