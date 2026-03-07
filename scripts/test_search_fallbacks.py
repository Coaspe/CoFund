#!/usr/bin/env python3
"""
scripts/test_search_fallbacks.py
================================
Live smoke test for Tavily/Exa fallback providers and desk evidence consumption.

Usage:
  python scripts/test_search_fallbacks.py
  python scripts/test_search_fallbacks.py --ticker NVDA
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.fundamental_agent import fundamental_analyst_run
from agents.macro_agent import macro_analyst_run
from data_providers.exa_search_provider import ExaSearchProvider
from data_providers.tavily_search_provider import TavilySearchProvider


def _print_provider_summary(name: str, items: list[dict]) -> None:
    print(f"\n[{name}] items={len(items)}")
    if not items:
        return
    top = items[0]
    print(f"  title={top.get('title', '')[:90]}")
    print(f"  url={top.get('url', '')}")
    print(f"  resolver_path={top.get('resolver_path', '')}")


def _macro_state(ticker: str, items: list[dict]) -> dict:
    return {
        "evidence_store": {
            str(item.get("hash", idx)): dict(item, ticker=ticker, desk="macro")
            for idx, item in enumerate(items, start=1)
        }
    }


def _funda_state(ticker: str, items: list[dict]) -> dict:
    return {
        "evidence_store": {
            str(item.get("hash", idx)): dict(item, ticker=ticker, desk="fundamental")
            for idx, item in enumerate(items, start=1)
        }
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="NVDA")
    args = parser.parse_args()

    ticker = str(args.ticker or "NVDA").strip().upper()
    tavily_key = bool(os.environ.get("TAVILY_API_KEY"))
    exa_key = bool(os.environ.get("EXA_API_KEY"))

    print(f"TAVILY_API_KEY={'SET' if tavily_key else 'MISSING'}")
    print(f"EXA_API_KEY={'SET' if exa_key else 'MISSING'}")
    if not tavily_key and not exa_key:
        print("No search API keys found. Populate .env and rerun.")
        return 1

    tavily_items: list[dict] = []
    exa_items: list[dict] = []

    if tavily_key:
        tavily = TavilySearchProvider(mode="live")
        tavily_items = tavily.collect_evidence(
            kind="macro_headline_context",
            ticker=ticker,
            query=f"{ticker} fed rates AI demand macro context",
            recency_days=14,
            max_items=3,
            desk="macro",
            resolver_path="tavily_fallback_macro",
        )
        _print_provider_summary("Tavily", tavily_items)

    if exa_key:
        exa = ExaSearchProvider(mode="live")
        exa_items = exa.collect_evidence(
            kind="valuation_context",
            ticker=ticker,
            query=f"{ticker} valuation premium semiconductor peers AI demand",
            recency_days=30,
            max_items=3,
            desk="fundamental",
            resolver_path="exa_fallback_default",
        )
        _print_provider_summary("Exa", exa_items)

    if tavily_items:
        macro = macro_analyst_run(
            ticker,
            {"yield_curve_spread": -0.1, "hy_oas": 410, "gdp_growth": 1.8, "pmi": 51.2},
            state=_macro_state(ticker, tavily_items),
        )
        print("\n[Macro consumption]")
        print(f"  evidence_digest={len(macro.get('evidence_digest', []))}")
        print(f"  key_driver_has_update={any('Evidence update:' in x for x in macro.get('key_drivers', []))}")

    if exa_items:
        funda = fundamental_analyst_run(
            ticker,
            {"pe_ratio": 32.0, "revenue_growth": 20.0, "roe": 18.0, "debt_to_equity": 0.3},
            state=_funda_state(ticker, exa_items),
        )
        print("\n[Fundamental consumption]")
        print(f"  evidence_digest={len(funda.get('evidence_digest', []))}")
        print(f"  key_driver_has_update={any('Evidence update:' in x for x in funda.get('key_drivers', []))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
