"""
portfolio/allocator.py — Deterministic Multi-Ticker Portfolio Allocator
=======================================================================
Amendment 4: Allocator 출력(positions_proposed)은 Risk 입력의 유일한 source.
Risk 엔진이 positions_final을 생성하고, Report는 positions_final만 사용.

규칙 (결정론적 Python, LLM 없음):
  1. Fundamental structural_risk_flag=True → weight = 0
  2. Quant final_allocation_pct가 base weight
  3. Sentiment tilt_factor를 tactical overlay로 반영 (hardcap 0.7~1.3)
  4. Macro overlay로 beta 조정 (hardcap ±0.3)
  5. 최종 합산 후 전체 합이 1.0 초과 시 정규화
"""
from __future__ import annotations

from typing import Any


# Sentiment tilt hardcap
TILT_MIN = 0.7
TILT_MAX = 1.3
# Macro overlay hardcap
MACRO_OVERLAY_MAX = 0.30


def allocate(
    universe: list[str],
    desk_outputs: dict[str, dict],
    risk_budget: float = 1.0,
) -> dict[str, float]:
    """
    멀티 종목 포지션 제안 생성 (결정론적).

    Args:
        universe: 분석 대상 티커 목록
        desk_outputs: {
            ticker: {
                "quant": {...},
                "fundamental": {...},
                "sentiment": {...},
                "macro": {...},
            }
        }
        risk_budget: 총 포트폴리오 리스크 예산 (기본 1.0 = 100%)

    Returns:
        positions_proposed: {ticker -> weight}
    """
    positions: dict[str, float] = {}

    # 매크로가 공통인 경우 (모든 티커 공유)
    common_macro = desk_outputs.get("_common", {}).get("macro", {})

    for ticker in universe:
        ticker_data = desk_outputs.get(ticker, {})

        quant = ticker_data.get("quant", {})
        fundamental = ticker_data.get("fundamental", {})
        sentiment = ticker_data.get("sentiment", {})
        macro = ticker_data.get("macro", common_macro)

        # Rule 1: Fundamental structural_risk_flag → weight = 0
        if fundamental.get("structural_risk_flag", False):
            positions[ticker] = 0.0
            continue

        # Rule 2: Quant allocation_pct가 base weight
        quant_decision = quant.get("decision", "HOLD").upper()
        quant_alloc = float(quant.get("final_allocation_pct", 0.0))
        if quant_decision in ("HOLD", "CLEAR", ""):
            base_weight = 0.0
        elif quant_decision == "SHORT":
            base_weight = -abs(quant_alloc)
        else:  # LONG, BUY
            base_weight = abs(quant_alloc)

        if abs(base_weight) < 1e-9:
            positions[ticker] = 0.0
            continue

        # Rule 3: Sentiment tilt overlay (hardcap 0.7~1.3)
        tilt = float(sentiment.get("tilt_factor", 1.0))
        tilt = max(TILT_MIN, min(TILT_MAX, tilt))
        adjusted = base_weight * tilt

        # Rule 4: Macro beta overlay (±30% 한도)
        macro_regime = macro.get("macro_regime", macro.get("regime", "normal"))
        macro_decision = macro.get("primary_decision", "neutral")
        macro_overlay = _macro_overlay(macro_regime, macro_decision)
        # overlay는 절대 비중에 더함 (부호 유지)
        sign = 1.0 if adjusted >= 0 else -1.0
        adjusted = sign * max(0.0, abs(adjusted) + macro_overlay)

        positions[ticker] = round(adjusted, 6)

    # Rule 5: 전체 합이 risk_budget 초과 시 정규화
    total = sum(abs(w) for w in positions.values())
    if total > risk_budget + 1e-9:
        scale = risk_budget / total
        positions = {t: round(w * scale, 6) for t, w in positions.items()}

    return positions


def _macro_overlay(regime: str, decision: str) -> float:
    """거시 레짐 기반 overlay. 절대값 기준 ±MACRO_OVERLAY_MAX 이내."""
    regime = (regime or "").lower()
    decision = (decision or "").lower()

    if regime in ("goldilocks", "expansion") or decision == "bullish":
        return min(0.02, MACRO_OVERLAY_MAX)   # 소폭 증가
    if regime in ("recession", "crisis", "contraction") or decision == "bearish":
        return max(-0.02, -MACRO_OVERLAY_MAX)  # 소폭 감소
    return 0.0
