"""
risk/gates/gate5_model_anomaly.py — Gate 5: Model / Data Anomaly
"""
from __future__ import annotations


class Gate5ModelAnomaly:
    """Gate 5: Quant 모델 이상 탐지 (비중 이상치, 데이터 품질)."""

    @staticmethod
    def apply(
        payload: dict,
        positions_in: dict[str, float],
        lim: dict,
    ) -> tuple[dict[str, float], list[str], dict, dict]:
        max_quant_weight = lim.get("max_quant_weight_anomaly", 0.30)
        fallback_weight = lim.get("conservative_fallback_weight", 0.05)

        flags: list[str] = []
        feedback_required = False
        feedback_reasons: list[str] = []
        feedback_detail_parts: list[str] = []
        positions_out = dict(positions_in)

        for t, w in positions_out.items():
            if abs(w) > max_quant_weight:
                flags.append(f"quant_weight_anomaly:{t}")
                feedback_required = True
                feedback_reasons.append("model_anomaly")
                sign = 1.0 if w >= 0 else -1.0
                positions_out[t] = round(sign * fallback_weight, 6)
                feedback_detail_parts.append(
                    f"{t} 비중 |{w:.4f}| > 한도 {max_quant_weight} → fallback {fallback_weight}."
                )

        # data_quality 이상 체크
        dq = payload.get("data_quality", {})
        if dq.get("is_mock", False):
            flags.append("mock_data_warning")

        trace = {
            "gate": 5,
            "name": "model_anomaly",
            "flags": flags,
            "feedback_required": feedback_required,
            "positions_after": dict(positions_out),
        }
        return positions_out, flags, {"required": feedback_required, "reasons": feedback_reasons, "detail": " ".join(feedback_detail_parts)}, trace
