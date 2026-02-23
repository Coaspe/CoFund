"""
risk/gates/gate2_concentration.py — Gate 2: Concentration / HHI
"""
from __future__ import annotations
import math


class Gate2Concentration:
    """Gate 2: 섹터/인자 집중도 및 HHI 검사."""

    @staticmethod
    def apply(
        payload: dict,
        positions_in: dict[str, float],
        lim: dict,
    ) -> tuple[dict[str, float], list[str], dict, dict]:
        summary = payload.get("portfolio_risk_summary", payload.get("portfolio_summary", {}))

        hhi = summary.get("herfindahl_index", 0.0) or 0.0
        sector_exposure = summary.get("sector_exposure", {})
        component_var = summary.get("component_var_by_ticker", {})
        portfolio_cvar = summary.get("portfolio_cvar_1d", 1e-9) or 1e-9

        max_hhi = lim.get("max_hhi", 0.35)
        max_sector = lim.get("max_sector_weight", 0.40)
        max_component_var_share = 0.40  # 단일 종목이 포트 VaR의 40% 초과 시 reduce

        flags: list[str] = []
        feedback_required = False
        feedback_reasons: list[str] = []
        feedback_detail_parts: list[str] = []
        positions_out = dict(positions_in)

        if hhi > max_hhi:
            flags.append("concentration_hhi")
            feedback_required = True
            feedback_reasons.append("concentration_hhi")
            feedback_detail_parts.append(f"HHI {hhi:.3f} > 한도 {max_hhi}.")

        for sector, exposure in sector_exposure.items():
            if abs(exposure) > max_sector:
                flags.append(f"sector_concentration:{sector}")
                feedback_required = True
                feedback_reasons.append("sector_concentration")
                feedback_detail_parts.append(f"섹터 {sector} 노출 {exposure:.2f} > 한도 {max_sector}.")

        total_comp = sum(abs(v) for v in component_var.values()) or 1.0
        for t, cv in component_var.items():
            if t in positions_out:
                share = abs(cv) / total_comp
                if share > max_component_var_share:
                    flags.append(f"component_var_dominant:{t}")
                    old_w = positions_out[t]
                    positions_out[t] = round(old_w * 0.7, 6)
                    feedback_detail_parts.append(
                        f"{t} Component VaR {share:.0%} > {max_component_var_share:.0%}: 비중 {old_w:.4f}→{positions_out[t]:.4f}."
                    )

        trace = {
            "gate": 2,
            "name": "concentration",
            "flags": flags,
            "hhi_checked": hhi,
            "feedback_required": feedback_required,
            "positions_after": dict(positions_out),
        }
        return positions_out, flags, {"required": feedback_required, "reasons": feedback_reasons, "detail": " ".join(feedback_detail_parts)}, trace
