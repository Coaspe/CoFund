"""
risk/gates/gate1_hard_limits.py — Gate 1: Hard Risk Limits
"""
from __future__ import annotations
from typing import Any


class Gate1HardLimits:
    """Gate 1: 포트폴리오 절대 리스크 한도 (CVaR, Leverage)."""

    @staticmethod
    def apply(
        payload: dict,
        positions_in: dict[str, float],
        lim: dict,
    ) -> tuple[dict[str, float], list[str], dict, dict]:
        """
        Returns:
            positions_out, flags_added, feedback_delta, trace_entry
        """
        summary = payload.get("portfolio_risk_summary", payload.get("portfolio_summary", {}))
        tickers = list(positions_in.keys())

        cvar = summary.get("portfolio_cvar_1d", 0.0) or 0.0
        leverage = summary.get("leverage_ratio", 0.0) or 0.0

        flags: list[str] = []
        feedback_required = False
        feedback_reasons: list[str] = []
        feedback_detail_parts: list[str] = []
        positions_out = dict(positions_in)

        max_cvar = lim.get("max_portfolio_cvar_1d", 0.05)
        max_leverage = lim.get("max_leverage", 2.5)

        if cvar > max_cvar:
            flags.append("portfolio_risk_violation")
            feedback_required = True
            feedback_reasons.append("portfolio_risk_violation")
            feedback_detail_parts.append(
                f"포트폴리오 CVaR {cvar:.4f} > 한도 {max_cvar}."
            )
            # 장기 비중 축소
            for t in tickers:
                w = positions_out[t]
                if w > 0:
                    positions_out[t] = round(w * (max_cvar / cvar), 6)

        if leverage > max_leverage:
            flags.append("leverage_violation")
            feedback_required = True
            feedback_reasons.append("leverage_violation")
            feedback_detail_parts.append(
                f"레버리지 {leverage:.2f} > 한도 {max_leverage}."
            )
            scale = max_leverage / leverage
            positions_out = {t: round(w * scale, 6) for t, w in positions_out.items()}

        trace = {
            "gate": 1,
            "name": "hard_limits",
            "flags": flags,
            "feedback_required": feedback_required,
            "cvar_checked": cvar,
            "leverage_checked": leverage,
            "positions_after": dict(positions_out),
        }
        return positions_out, flags, {"required": feedback_required, "reasons": feedback_reasons, "detail": " ".join(feedback_detail_parts)}, trace
