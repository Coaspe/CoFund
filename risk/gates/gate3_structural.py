"""
risk/gates/gate3_structural.py — Gate 3: Structural / Fundamental Risk
"""
from __future__ import annotations

HARD_REJECT_FLAGS = {"default_risk", "accounting_fraud", "regulatory_action", "going_concern"}


class Gate3Structural:
    """Gate 3: 개별 종목 구조적 리스크 (Fundamental 플래그 기반)."""

    @staticmethod
    def apply(
        payload: dict,
        positions_in: dict[str, float],
        lim: dict,
    ) -> tuple[dict[str, float], list[str], dict, dict]:
        analyst_reports = payload.get("analyst_reports", {})
        fundamental = analyst_reports.get("fundamental", {})

        # 지원: 단일 또는 per-ticker fundamental 데이터
        per_ticker_fundamental = {
            t: data.get("fundamental", {})
            for t, data in payload.get("per_ticker_data", {}).items()
        }
        # flat payload fallback
        if not per_ticker_fundamental and fundamental:
            for t in positions_in:
                per_ticker_fundamental[t] = fundamental

        flags: list[str] = []
        feedback_detail_parts: list[str] = []
        positions_out = dict(positions_in)
        rejected: list[str] = []

        for ticker, funda in per_ticker_fundamental.items():
            if ticker not in positions_out:
                continue
            # risk_flags 리스트 또는 hard_red_flags 확인
            ticker_flags = set()
            for rf in funda.get("risk_flags", []) + funda.get("hard_red_flags", []):
                code = rf.get("code", "") if isinstance(rf, dict) else str(rf)
                ticker_flags.add(code)

            # structural_risk_flag 부울 필드도 체크
            if funda.get("structural_risk_flag", False):
                ticker_flags.add("structural_risk_flag")

            hit = ticker_flags & HARD_REJECT_FLAGS
            if hit:
                positions_out[ticker] = 0.0
                rejected.append(ticker)
                flags.append(f"structural_reject:{ticker}")
                feedback_detail_parts.append(
                    f"{ticker} 구조적 리스크 ({', '.join(hit)}) → 비중 0 (reject_local)."
                )

        trace = {
            "gate": 3,
            "name": "structural_risk",
            "flags": flags,
            "rejected_tickers": rejected,
            "feedback_required": False,  # reject_local → 피드백 불필요
            "positions_after": dict(positions_out),
        }
        return positions_out, flags, {"required": False, "reasons": [], "detail": " ".join(feedback_detail_parts)}, trace
