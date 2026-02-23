"""
risk/gates/gate4_regime_fit.py — Gate 4: Regime / Strategy Fit
"""
from __future__ import annotations

RISK_OFF_REGIMES = {"recession", "crisis", "contraction", "stagflation"}
LONG_THRESHOLD = 0.10  # 10% 이상 롱을 "적극 매수"로 판단


class Gate4RegimeFit:
    """Gate 4: 거시 레짐과 포지션 방향 정합성 검사."""

    @staticmethod
    def apply(
        payload: dict,
        positions_in: dict[str, float],
        lim: dict,
    ) -> tuple[dict[str, float], list[str], dict, dict]:
        analyst_reports = payload.get("analyst_reports", {})
        macro = analyst_reports.get("macro", {})
        regime = macro.get("regime", macro.get("macro_regime", "normal"))
        canonical_regime = (regime or "normal").lower()

        flags: list[str] = []
        feedback_required = False
        feedback_reasons: list[str] = []
        feedback_detail_parts: list[str] = []
        positions_out = dict(positions_in)

        if canonical_regime in RISK_OFF_REGIMES:
            for t, w in positions_out.items():
                if w > LONG_THRESHOLD:
                    flags.append(f"regime_mismatch:{t}")
                    feedback_required = True
                    feedback_reasons.append("regime_alignment_violation")
                    # 방어적 리밸런싱: 비중 절반으로 축소
                    positions_out[t] = round(w * 0.5, 6)
                    feedback_detail_parts.append(
                        f"{t} 레짐 {canonical_regime} 불일치 → 비중 {w:.4f}→{positions_out[t]:.4f}."
                    )

        trace = {
            "gate": 4,
            "name": "regime_fit",
            "flags": flags,
            "canonical_regime": canonical_regime,
            "feedback_required": feedback_required,
            "positions_after": dict(positions_out),
        }
        return positions_out, flags, {"required": feedback_required, "reasons": feedback_reasons, "detail": " ".join(feedback_detail_parts)}, trace
