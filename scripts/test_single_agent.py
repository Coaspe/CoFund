#!/usr/bin/env python3
"""
scripts/test_single_agent.py — 개별 에이전트 LLM 단독 테스트
==============================================================
사용법:
  ./.venv/bin/python scripts/test_single_agent.py --agent orchestrator
  ./.venv/bin/python scripts/test_single_agent.py --agent macro
  ./.venv/bin/python scripts/test_single_agent.py --agent risk_manager
  ./.venv/bin/python scripts/test_single_agent.py --agent report_writer
  ./.venv/bin/python scripts/test_single_agent.py --all
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings
warnings.filterwarnings("ignore")

# ── 공통 fixture ──────────────────────────────────────────────────────────────

TICKER = "005930.KS"

def _quant_output():
    return {
        "decision": "LONG",
        "final_allocation_pct": 0.08,
        "cot_reasoning": "Z-score < -2 → LONG 신호. CVaR 통과.",
    }

def _macro_output():
    return {
        "primary_decision": "bullish",
        "recommendation": "allow",
        "confidence": 0.65,
        "macro_regime": "expansion",
        "tail_risk_warning": False,
        "summary": "Goldilocks: 완화적 금리 + 고성장 환경.",
    }

def _risk_payload():
    return {
        "target_ticker": TICKER,
        "portfolio_summary": {
            "portfolio_cvar_1d": 0.01,
            "leverage_ratio": 1.0,
            "concentration_top1": 0.3,
            "gross_exposure": 1.0,
        },
        "analyst_weights": {TICKER: 0.08},
        "per_ticker_data": {
            TICKER: {
                "quant": _quant_output(),
                "macro": _macro_output(),
            }
        },
        "limits": {},
    }


# ── 에이전트별 테스트 함수 ─────────────────────────────────────────────────────

def test_orchestrator():
    print("=" * 60)
    print("🤖 Orchestrator Agent (LLM Router)")
    print("=" * 60)
    from agents.orchestrator_agent import _call_llm
    result = _call_llm("삼성전자 매수해도 될까?", iteration=1)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def test_macro():
    print("=" * 60)
    print("📊 Macro Analyst Agent (Python 엔진 → Mock LLM)")
    print("=" * 60)
    from agents.macro_agent import macro_analyst_run
    result = macro_analyst_run(
        ticker=TICKER,
        macro_indicators={
            "yield_curve_spread": 0.50,
            "hy_oas": 3.2,
            "inflation_expectation": 2.1,
            "cpi_yoy": 2.5,
            "pmi": 52.0,
            "fed_funds_rate": 4.25,
            "gdp_growth": 2.8,
        },
        run_id="test",
    )
    print(json.dumps({k: result[k] for k in ["primary_decision", "recommendation", "confidence", "macro_regime", "summary"]}, indent=2, ensure_ascii=False))
    print(f"  → evidence 개수: {len(result.get('evidence', []))}")
    print("  ⓘ Macro/Fundamental/Sentiment 에이전트는 Python 엔진이 결정, LLM 불필요.")


def test_fundamental():
    print("=" * 60)
    print("� Fundamental Analyst Agent (Python 엔진 → Mock LLM)")
    print("=" * 60)
    from agents.fundamental_agent import fundamental_analyst_run
    result = fundamental_analyst_run(
        ticker=TICKER,
        financials={
            "total_assets": 100000,
            "total_liabilities": 35000,
            "revenue_growth": 8.0,
            "debt_to_equity": 0.35,
            "operating_margin": 20.0,
            "roe": 18.0,
            "pe_ratio": 18.0,
            "free_cash_flow": 2000,
            "interest_expense": 200,
        },
        run_id="test",
    )
    print(json.dumps({k: result[k] for k in ["primary_decision", "recommendation", "confidence", "summary"]}, indent=2, ensure_ascii=False))


def test_sentiment():
    print("=" * 60)
    print("💬 Sentiment Analyst Agent (Python 엔진 → Mock LLM)")
    print("=" * 60)
    from agents.sentiment_agent import sentiment_analyst_run
    result = sentiment_analyst_run(
        ticker=TICKER,
        sentiment_indicators={
            "put_call_ratio": 0.9,
            "vix_level": 18,
            "news_sentiment_score": 0.6,
            "news_articles": [
                {"title": "삼성전자 HBM 공급계약 체결", "source": "reuters", "published_at": "2026-01-01T00:00:00Z"},
                {"title": "반도체 수출 증가세 지속", "source": "bloomberg", "published_at": "2026-01-01T00:00:00Z"},
            ],
        },
        run_id="test",
    )
    print(json.dumps({k: result[k] for k in ["primary_decision", "recommendation", "confidence", "summary"]}, indent=2, ensure_ascii=False))


def test_risk_manager():
    print("=" * 60)
    print("🔒 Risk Manager Agent (Python 5-Gate + Groq narrative)")
    print("=" * 60)
    from agents.risk_agent import _call_llm
    payload = _risk_payload()
    result = _call_llm(payload)
    print("  결정:", json.dumps(result.get("per_ticker_decisions", {}), indent=2, ensure_ascii=False))
    print("  피드백:", json.dumps(result.get("orchestrator_feedback", {}), indent=2, ensure_ascii=False))


def test_quant():
    print("=" * 60)
    print("📐 Quant Agent (Python만 — LLM disabled)")
    print("=" * 60)
    from agents.quant_agent import _call_llm
    payload = {
        "alpha_signals": {
            "statistical_arbitrage": {
                "adf_test": {"p_value": 0.02, "is_stationary": True},
                "execution": {"current_z_score": -2.5},
            },
            "factor_exposures": {"newey_west_t_stat": -2.1, "p_value": 0.03},
        },
        "market_regime_context": {"state_probabilities": {"regime_2_high_vol": 0.2}},
        "portfolio_risk_parameters": {
            "full_kelly_fraction": 0.12,
            "fractional_multiplier": 0.5,
            "asset_cvar_99_daily": 0.03,
            "max_portfolio_cvar_limit": 0.05,
        },
    }
    result = _call_llm(payload)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("  ⓘ decision/final_allocation_pct는 항상 Python 규칙 기반 (Invariant A).")


def test_report_writer():
    print("=" * 60)
    print("📝 Report Writer Agent (Gemini — 마크다운 보고서)")
    print("=" * 60)
    from agents.report_agent import _call_llm
    from schemas.common import create_initial_state
    state = create_initial_state(TICKER)
    state.update({
        "risk_manager_decision": {
            "orchestrator_feedback": {"required": False, "reasons": [], "detail": "전 Gate 통과."},
            "per_ticker_decisions": {TICKER: {"decision": "approve", "final_weight": 0.08, "flags": []}},
            "portfolio_actions": {},
        },
        "technical_analysis": _quant_output(),
        "macro_analysis": _macro_output(),
        "fundamental_analysis": {"primary_decision": "bullish", "summary": "건전한 대차대조표."},
        "sentiment_analysis": {"primary_decision": "bullish", "summary": "긍정적 뉴스 흐름."},
    })
    report = _call_llm(state)
    # 너무 길면 앞 500자만 출력
    print(report[:500] + "...\n  [이하 생략]" if len(report) > 500 else report)


# ── 진입점 ────────────────────────────────────────────────────────────────────

AGENTS = {
    "orchestrator": test_orchestrator,
    "macro": test_macro,
    "fundamental": test_fundamental,
    "sentiment": test_sentiment,
    "quant": test_quant,
    "risk_manager": test_risk_manager,
    "report_writer": test_report_writer,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=list(AGENTS.keys()), help="테스트할 에이전트")
    parser.add_argument("--all", action="store_true", help="전체 에이전트 순서대로 테스트")
    args = parser.parse_args()

    if args.all:
        for name, fn in AGENTS.items():
            fn()
            print()
    elif args.agent:
        AGENTS[args.agent]()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
