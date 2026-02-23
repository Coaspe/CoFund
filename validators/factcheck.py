"""
validators/factcheck.py — LLM Output Fact-Check Validators
===========================================================
R5: LLM 출력의 사실성/정합성 자동 검증.

Amendment 3 (preferred): Risk narrative no-digits/no-%/no-tickers rule.
"""
from __future__ import annotations

import re
from typing import Optional


class FactCheckError(ValueError):
    """LLM 출력 사실성 검증 실패."""
    pass


# ── Orchestrator 검증 ─────────────────────────────────────────────────────────

def validate_orchestrator_output(state: dict, orch_json: dict) -> None:
    """
    Orchestrator가 allowed_universe 외 티커를 생성하면 FactCheckError.
    universe가 비어 있으면 검증 생략 (초기 run).
    """
    universe: list = state.get("universe", [])
    if not universe:
        return

    target_universe = orch_json.get("investment_brief", {}).get("target_universe", [])
    extra = [t for t in target_universe if t not in universe]
    if extra:
        raise FactCheckError(
            f"Orchestrator가 허용되지 않은 티커를 생성했습니다: {extra}. "
            f"허용 목록: {universe}"
        )


# ── Risk Narrative 검증 (Amendment 3: no-digits rule) ────────────────────────

_DIGIT_RE = re.compile(r"\d")
_PERCENT_RE = re.compile(r"%")
_TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")  # 1~5 대문자 단어


def validate_risk_narrative(state: dict, narrative_text: str) -> None:
    """
    Risk narrative에 숫자/퍼센트/티커 심볼이 포함되면 FactCheckError.

    Rule (Amendment 3 Preferred):
      - 숫자(0-9) 금지
      - '%' 기호 금지
      - 대문자 티커 패턴 (universe에 있는 것) 금지
    """
    if not narrative_text:
        return

    if _DIGIT_RE.search(narrative_text):
        raise FactCheckError(
            "Risk narrative에 숫자가 포함되어 있습니다. "
            "수치는 Python 결정에서만 나와야 합니다."
        )

    if _PERCENT_RE.search(narrative_text):
        raise FactCheckError(
            "Risk narrative에 '%'이 포함되어 있습니다."
        )

    universe: list = state.get("universe", []) or [state.get("target_ticker", "")]
    for ticker in universe:
        if ticker and ticker in narrative_text:
            raise FactCheckError(
                f"Risk narrative에 티커 '{ticker}'이 포함되어 있습니다."
            )


# ── Report Markdown 검증 ──────────────────────────────────────────────────────

_WEIGHT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_TICKER_IN_MD_RE = re.compile(r"\b([A-Z]{1,6}(?:\.[A-Z]{1,3})?)\b")


def validate_report_markdown(
    state: dict,
    report_md: str,
    llm=None,
    report_system_prompt: str = "",
    report_human_msg_fn=None,
) -> str:
    """
    보고서에 등장하는 비중 수치가 positions_final과 ±1% 이내인지 검증.

    실패 시:
      1. LLM이 있으면 correction 프롬프트로 1회 재시도
      2. 그래도 실패하면 Python 템플릿 fallback 반환

    Returns: 검증 통과한 report 문자열
    """
    positions_final: dict[str, float] = state.get("positions_final", {})
    if not positions_final:
        return report_md  # 포지션 없으면 검증 생략

    violations = _check_report_violations(report_md, positions_final)
    if not violations:
        return report_md

    # 1차 실패: LLM 재시도
    if llm is not None:
        correction_prompt = (
            f"아래 보고서의 비중 수치에 오류가 있습니다: {violations}\n"
            f"올바른 positions_final: {positions_final}\n\n"
            "비중 수치를 올바른 값으로 교정한 완성된 마크다운 보고서를 다시 작성하세요.\n\n"
            f"원본 보고서:\n{report_md}"
        )
        try:
            from langchain_core.messages import HumanMessage
            resp = llm.invoke([HumanMessage(content=correction_prompt)])
            corrected = resp.content
            violations2 = _check_report_violations(corrected, positions_final)
            if not violations2:
                return corrected
        except Exception:
            pass

    # 2차 실패: Python 템플릿 fallback
    return _template_report(state)


def _check_report_violations(report_md: str, positions_final: dict[str, float]) -> list[str]:
    """보고서에서 비중 수치를 추출하여 positions_final과 비교."""
    violations = []
    # 퍼센트 형태 검색: e.g. "8.5%" 혹은 "AAPL 8.5%"
    for m in _WEIGHT_RE.finditer(report_md):
        pct_value = float(m.group(1)) / 100.0
        # 전후 문맥에서 티커 찾기
        context = report_md[max(0, m.start()-30):m.end()+5]
        for ticker in positions_final:
            if ticker in context:
                expected = positions_final[ticker]
                if abs(pct_value - abs(expected)) > 0.01:
                    violations.append(
                        f"{ticker}: 보고서={pct_value:.2%}, 실제={expected:.2%}"
                    )
    return violations


def _template_report(state: dict) -> str:
    """Python 순수 템플릿 기반 보고서 (최후 fallback)."""
    positions_final: dict[str, float] = state.get("positions_final", {})
    risk_decision = state.get("risk_manager_decision", {})
    feedback = risk_decision.get("orchestrator_feedback", {})
    as_of = state.get("as_of", "N/A")

    lines = [
        f"# IC 투자 메모 (템플릿 생성)",
        f"",
        f"**기준일**: {as_of}",
        f"",
        f"## Executive Summary",
        f"",
        f"리스크 엔진 결정에 따른 최종 포지션:",
        f"",
        f"| Ticker | 최종 비중 | 결정 |",
        f"|--------|-----------|------|",
    ]
    per_ticker = risk_decision.get("per_ticker_decisions", {})
    for ticker, weight in positions_final.items():
        decision = per_ticker.get(ticker, {}).get("decision", "N/A")
        lines.append(f"| {ticker} | {weight:.2%} | {decision} |")

    if feedback.get("required"):
        lines += [
            f"",
            f"## Orchestrator 피드백 필요",
            f"",
            f"{feedback.get('detail', '')}",
        ]

    lines += [
        f"",
        f"---",
        f"*본 보고서는 LLM 검증 실패로 인해 Python 템플릿으로 자동 생성되었습니다.*",
    ]
    return "\n".join(lines)
