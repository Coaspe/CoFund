from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.autonomy_overlay import ensure_schema
from agents.fundamental_agent import _apply_overlay_patch as apply_fundamental_patch
from agents.macro_agent import _apply_overlay_patch as apply_macro_patch
from agents.sentiment_agent import _apply_overlay_patch as apply_sentiment_patch


def test_macro_overlay_schema_accepts_bucket_patch():
    patch = ensure_schema(
        {
            "primary_decision": "neutral",
            "recommendation": "allow_with_limits",
            "confidence": 0.41,
        },
        desk="macro",
        ticker="NVDA",
    )

    assert patch == {
        "primary_decision": "neutral",
        "recommendation": "allow_with_limits",
        "confidence": 0.41,
    }


def test_fundamental_overlay_schema_accepts_avoid_bucket():
    patch = ensure_schema(
        {
            "primary_decision": "avoid",
            "recommendation": "reject",
            "confidence": 0.33,
        },
        desk="fundamental",
        ticker="TSLA",
    )

    assert patch["primary_decision"] == "avoid"
    assert patch["recommendation"] == "reject"
    assert patch["confidence"] == 0.33


def test_sentiment_overlay_schema_blocks_invalid_buckets():
    patch = ensure_schema(
        {
            "primary_decision": "avoid",
            "recommendation": "reject",
            "confidence": 1.25,
        },
        desk="sentiment",
        ticker="AAPL",
    )

    assert "primary_decision" not in patch
    assert "recommendation" not in patch
    assert patch["confidence"] == 1.0


@pytest.mark.parametrize(
    ("apply_patch", "patched_values"),
    [
        (
            apply_macro_patch,
            {
                "primary_decision": "neutral",
                "recommendation": "allow_with_limits",
                "confidence": 0.0,
            },
        ),
        (
            apply_fundamental_patch,
            {
                "primary_decision": "bearish",
                "recommendation": "allow_with_limits",
                "confidence": 0.37,
            },
        ),
        (
            apply_sentiment_patch,
            {
                "primary_decision": "bullish",
                "recommendation": "allow",
                "confidence": 0.44,
            },
        ),
    ],
)
def test_desk_overlay_patch_applies_bucket_fields(apply_patch, patched_values):
    output = {
        "primary_decision": "bullish",
        "recommendation": "allow",
        "confidence": 0.65,
        "evidence_requests": [],
    }

    apply_patch(output, patched_values)

    assert output["primary_decision"] == patched_values["primary_decision"]
    assert output["recommendation"] == patched_values["recommendation"]
    assert output["confidence"] == patched_values["confidence"]
