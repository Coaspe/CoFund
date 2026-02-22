"""
schemas/taxonomy.py — Canonical regime taxonomy
=================================================
Single source of truth for macro regime classification.
Macro agent emits raw regimes; this module maps them to canonical values
that risk_agent Gate4 uses for decisions.
"""

from __future__ import annotations

from typing import Set

# Canonical regime values (the only values Gate4 should inspect)
CANONICAL_REGIMES: Set[str] = frozenset({
    "expansion", "goldilocks", "late_cycle",
    "recession", "stagflation", "crisis", "normal",
})

# Risk-off regimes: Gate4 triggers defensive rebalancing for active LONG
RISK_OFF_REGIMES: Set[str] = frozenset({"recession", "crisis", "stagflation"})

# Mapping from raw macro_engine output → canonical
_RAW_TO_CANONICAL = {
    "expansion": "expansion",
    "goldilocks": "goldilocks",
    "late_cycle": "late_cycle",
    "contraction": "recession",       # ← THE KEY FIX: macro says "contraction", risk expects "recession"
    "recession": "recession",
    "stagflation": "stagflation",
    "crisis": "crisis",
    "normal": "normal",
    "early_recovery": "expansion",
    "reflation": "expansion",
}


def map_macro_regime_to_canonical(raw_regime: str) -> str:
    """Map raw macro_engine regime to canonical taxonomy."""
    return _RAW_TO_CANONICAL.get(raw_regime.lower().strip(), "normal")
