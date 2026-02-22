"""
investment_team.py — 7-Agent AI Investment Team (V3 — Final Integration)
========================================================================
CHANGELOG:
  v3.0 (2026-02-22) — Final Integration.
    - New directory structure: engines/, agents/, data_providers/, schemas/
    - 4 desk agents (Macro, Fundamental, Sentiment, Quant) with engine isolation
    - barrier_node for fan-in (risk_manager 1회 실행 보장)
    - Telemetry hooks on all nodes
    - Conditional routing: orchestrator → requested desks only
    - Risk Manager feedback loop with MAX_ITERATIONS cap

실행:
    python investment_team.py
    python investment_team.py --mode mock --seed 42

Iron Rules Enforced:
  R0: No Trading (no broker code)
  R1: Python-LLM Separation (engines compute, LLM interprets)
  R2: No Evidence, No Trade (all outputs have evidence[])
  R3: Quant Engine Isolation (no data fetching in engines/)
  R4: Risk 5-Gate Order Fixed (Gate1→2→3→4→5)
  R5: Sentiment Tilt Cap [0.7, 1.3]
  R6: Disagreement = Model Risk
  R7: Auditability (run_id, events.jsonl, final_state.json)
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Literal

from langgraph.graph import END, START, StateGraph

from schemas.common import InvestmentState, create_initial_state, make_evidence
import telemetry

# ── Data Providers ────────────────────────────────────────────────────────
from data_providers.data_hub import DataHub

# ── Agents ────────────────────────────────────────────────────────────────
from agents.macro_agent import macro_analyst_run
from agents.fundamental_agent import fundamental_analyst_run
from agents.sentiment_agent import sentiment_analyst_run

# ── Engines (for quant) ──────────────────────────────────────────────────
from engines.quant_engine import generate_quant_payload, mock_quant_decision

# ── Existing agent imports (graceful) ─────────────────────────────────────
try:
    from agents.orchestrator_agent import orchestrator_node as _orch_node_impl
    HAS_ORCHESTRATOR = True
except ImportError:
    HAS_ORCHESTRATOR = False

try:
    from agents.risk_agent import risk_manager_node as _risk_node_impl
except ImportError:
    _risk_node_impl = None  # type: ignore

try:
    from agents.report_agent import report_writer_node as _report_node_impl
except ImportError:
    _report_node_impl = None  # type: ignore


MAX_ITERATIONS = 3
ALL_DESKS = ["macro", "fundamental", "sentiment", "quant"]

# ── Task key normalization map ────────────────────────────────────────────
_TASK_TO_DESK = {
    "macro_analysis": "macro", "macro": "macro",
    "fundamental_analysis": "fundamental", "fundamental": "fundamental",
    "sentiment_analysis": "sentiment", "sentiment": "sentiment",
    "technical_analysis": "quant", "quant": "quant",
}

_DESK_TO_STATE_KEY = {
    "macro": "macro_analysis",
    "fundamental": "fundamental_analysis",
    "sentiment": "sentiment_analysis",
    "quant": "technical_analysis",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Telemetry wrapper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _log(state: dict, node: str, phase: str, data: dict | None = None):
    rid = state.get("run_id")
    if not rid:
        return
    telemetry.log_event(
        rid, node_name=node, iteration=state.get("iteration_count", 0),
        phase=phase,
        inputs_summary={"ticker": state.get("target_ticker")} if phase == "enter" else None,
        outputs_summary=data,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ① Orchestrator Node
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def orchestrator_node(state: InvestmentState) -> dict:
    """① Orchestrator — CIO/PM."""
    _log(state, "orchestrator", "enter")

    if HAS_ORCHESTRATOR:
        result = _orch_node_impl(state)
    else:
        iteration = state.get("iteration_count", 0)
        ticker = "AAPL"
        result = {
            "target_ticker": ticker,
            "analysis_tasks": ALL_DESKS[:],
            "iteration_count": iteration + 1,
            "orchestrator_directives": {
                "action_type": "initial_delegation" if iteration == 0 else "fallback_abort",
            },
        }

    # Normalize analysis_tasks
    raw = result.get("analysis_tasks", [])
    normalized = list(dict.fromkeys(
        _TASK_TO_DESK.get(t, t) for t in raw if _TASK_TO_DESK.get(t, t) in ALL_DESKS
    ))
    result["analysis_tasks"] = normalized if normalized else ALL_DESKS[:]

    # Init completed_tasks for this iteration
    ct = {d: (d not in result["analysis_tasks"]) for d in ALL_DESKS}
    result["completed_tasks"] = ct

    trace_entry = {
        "node": "orchestrator",
        "iteration": result.get("iteration_count", 0),
        "action_type": result.get("orchestrator_directives", {}).get("action_type"),
        "analysis_tasks": result["analysis_tasks"],
    }
    result["trace"] = [trace_entry]

    _log(state, "orchestrator", "exit", {"action": trace_entry.get("action_type")})
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ②~⑤ Desk Analyst Nodes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _make_seed(state: dict) -> int:
    """Stable seed: CLI seed if present, else sha256 of run_id (cross-process reproducible)."""
    explicit_seed = state.get("seed")
    if explicit_seed is not None:
        return int(explicit_seed) % (2**31)
    import hashlib
    run_id = state.get("run_id", "42")
    h = hashlib.sha256(run_id.encode()).digest()
    return int.from_bytes(h[:4], "big") % (2**31)


def macro_analyst_node(state: InvestmentState) -> dict:
    """② Macro Analyst."""
    if state.get("completed_tasks", {}).get("macro", False):
        return {}
    _log(state, "macro", "enter")

    mode = state.get("mode", "mock")
    seed = _make_seed(state)
    as_of = state.get("as_of", "")
    ticker = state.get("target_ticker", "AAPL")

    hub = DataHub(run_id=state.get("run_id", ""), as_of=as_of, mode=mode)
    indicators, _, meta = hub.get_macro_indicators(ticker, seed=seed)
    output = macro_analyst_run(
        ticker, indicators,
        run_id=state.get("run_id", ""), as_of=as_of,
        source_name="mock" if mode == "mock" else "FRED",
    )

    print(f"\n② MACRO ANALYST  (iter #{state.get('iteration_count', 1)})")
    print(f"   Regime: {output['macro_regime']}, GDP: {indicators.get('gdp_growth')}")

    _log(state, "macro", "exit", {"regime": output["macro_regime"]})
    return {"macro_analysis": output, "completed_tasks": {"macro": True}}


def fundamental_analyst_node(state: InvestmentState) -> dict:
    """③ Fundamental Analyst."""
    if state.get("completed_tasks", {}).get("fundamental", False):
        return {}
    _log(state, "fundamental", "enter")

    mode = state.get("mode", "mock")
    seed = _make_seed(state) + 1
    as_of = state.get("as_of", "")
    ticker = state.get("target_ticker", "AAPL")

    hub = DataHub(run_id=state.get("run_id", ""), as_of=as_of, mode=mode)
    financials, _, _ = hub.get_fundamentals(ticker, seed=seed)
    sec_data, _, _ = hub.get_sec_flags(ticker)
    output = fundamental_analyst_run(
        ticker, financials, sec_data=sec_data,
        run_id=state.get("run_id", ""), as_of=as_of,
        source_name="mock" if mode == "mock" else "FMP/SEC",
    )

    print(f"\n③ FUNDAMENTAL ANALYST  (iter #{state.get('iteration_count', 1)})")
    print(f"   Structural Risk: {output['structural_risk_flag']}, Decision: {output['primary_decision']}")

    _log(state, "fundamental", "exit", {"structural_risk": output["structural_risk_flag"]})
    return {"fundamental_analysis": output, "completed_tasks": {"fundamental": True}}


def sentiment_analyst_node(state: InvestmentState) -> dict:
    """④ Sentiment Analyst."""
    if state.get("completed_tasks", {}).get("sentiment", False):
        return {}
    _log(state, "sentiment", "enter")

    mode = state.get("mode", "mock")
    seed = _make_seed(state) + 2
    as_of = state.get("as_of", "")
    ticker = state.get("target_ticker", "AAPL")

    hub = DataHub(run_id=state.get("run_id", ""), as_of=as_of, mode=mode)
    indicators, _, _ = hub.get_news_sentiment(ticker, seed=seed)
    output = sentiment_analyst_run(
        ticker, indicators,
        run_id=state.get("run_id", ""), as_of=as_of,
        source_name="mock" if mode == "mock" else "NewsAPI",
    )

    print(f"\n④ SENTIMENT ANALYST  (iter #{state.get('iteration_count', 1)})")
    print(f"   Regime: {output['sentiment_regime']}, Tilt: {output['tilt_factor']}")

    _log(state, "sentiment", "exit", {"tilt": output["tilt_factor"]})
    return {"sentiment_analysis": output, "completed_tasks": {"sentiment": True}}


def quant_analyst_node(state: InvestmentState) -> dict:
    """⑤ Quant Analyst (wrapper: data_provider → quant_engine → mock decision)."""
    if state.get("completed_tasks", {}).get("quant", False):
        return {}
    _log(state, "quant", "enter")

    mode = state.get("mode", "mock")
    seed = _make_seed(state) + 3
    as_of = state.get("as_of", "")
    ticker = state.get("target_ticker", "AAPL")

    # Data Provider → 가격 배열 (via DataHub)
    hub = DataHub(run_id=state.get("run_id", ""), as_of=as_of, mode=mode)
    prices, p_ev, _ = hub.get_price_series(ticker, seed=seed)
    pair_prices, _, _ = hub.get_price_series("MSFT", seed=seed + 100)
    market_prices, _, _ = hub.get_market_series("SPY", seed=seed + 200)

    # Engine → 순수 연산
    print(f"\n⑤ QUANT ANALYST  (iter #{state.get('iteration_count', 1)})")
    print(f"   Computing quant payload for {ticker}...")
    payload = generate_quant_payload(ticker, prices, pair_prices, market_prices)

    # Mock LLM Decision
    decision = mock_quant_decision(payload)
    z = ((payload.get("alpha_signals", {}).get("statistical_arbitrage", {})
          .get("execution", {})).get("current_z_score"))
    cvar = (payload.get("portfolio_risk_parameters", {}).get("asset_cvar_99_daily"))

    print(f"   Decision: {decision['decision']} | Alloc: {decision['final_allocation_pct']}")

    # Evidence
    evidence = list(p_ev)
    if z is not None:
        evidence.append(make_evidence(metric="z_score", value=z, source_name="quant_engine", source_type="model", quality=0.9, as_of=as_of))
    if cvar is not None:
        evidence.append(make_evidence(metric="asset_cvar_99_daily", value=cvar, source_name="quant_engine", source_type="model", quality=0.9, as_of=as_of))

    output = {
        "agent_type": "quant",
        "run_id": state.get("run_id", ""),
        "generated_at": as_of,
        "as_of": as_of,
        "ticker": ticker,
        "decision": decision["decision"],
        "final_allocation_pct": decision["final_allocation_pct"],
        "z_score": z,
        "asset_cvar_99_daily": cvar,
        "quant_payload": payload,
        "llm_decision": decision,
        "evidence": evidence,
        "summary": f"Quant: {decision['decision']} alloc={decision['final_allocation_pct']}, Z={z}",
        "status": "ok",
        "data_ok": payload.get("_data_ok", True),
        "primary_decision": {
            "LONG": "bullish", "SHORT": "bearish",
            "HOLD": "hold", "CLEAR": "neutral",
        }.get(decision["decision"], "hold"),
        "recommendation": "allow" if decision["decision"] in ("LONG", "SHORT") else "allow_with_limits",
        "confidence": 0.6 if payload.get("_data_ok") else 0.35,
        "risk_flags": [],
        "limitations": ["Mock data" if mode == "mock" else ""],
    }

    _log(state, "quant", "exit", {"decision": decision["decision"]})
    return {"technical_analysis": output, "completed_tasks": {"quant": True}}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Barrier + Risk Manager + Report Writer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def barrier_node(state: InvestmentState) -> dict:
    """Fan-in barrier. Marks unfinished desks as skipped with iteration_generated."""
    ct = dict(state.get("completed_tasks", {}))
    iteration = state.get("iteration_count", 0)
    updates: dict = {}
    for desk in ALL_DESKS:
        key = _DESK_TO_STATE_KEY[desk]
        desk_out = state.get(key)
        if not ct.get(desk, False):
            # Desk didn't run this iteration
            ct[desk] = True
            if not desk_out:
                updates[key] = {
                    "summary": f"{desk} skipped",
                    "evidence": [],
                    "status": "skipped",
                    "iteration_generated": iteration,
                }
            else:
                # Previous result exists but was not re-run this iteration
                merged = dict(desk_out)
                merged["status"] = "skipped"
                merged["stale_previous"] = True
                merged["iteration_generated"] = merged.get("iteration_generated", iteration - 1)
                updates[key] = merged
        else:
            # Desk ran: ensure iteration_generated is set
            if desk_out and desk_out.get("iteration_generated") is None:
                merged = dict(desk_out)
                merged["iteration_generated"] = iteration
                updates[key] = merged
    updates["completed_tasks"] = ct
    return updates


def risk_manager_node(state: InvestmentState) -> dict:
    """⑥ Risk Manager (5-Gate, barrier 후, iteration 중복 방지)."""
    _log(state, "risk_manager", "enter")
    iteration = state.get("iteration_count", 0)

    existing = state.get("risk_assessment", {})
    if existing.get("_iteration_evaluated") == iteration:
        _log(state, "risk_manager", "exit", {"skipped": True})
        return {}

    if _risk_node_impl is not None:
        result = _risk_node_impl(state)
    else:
        result = {
            "risk_assessment": {
                "grade": "Low",
                "risk_decision": {
                    "per_ticker_decisions": {},
                    "portfolio_actions": {},
                    "orchestrator_feedback": {"required": False, "reasons": [], "detail": "Mock pass."},
                },
                "summary": "Mock risk — all clear.",
                "evidence": [],
                "status": "ok",
            }
        }

    ra = result.get("risk_assessment", {})
    ra["_iteration_evaluated"] = iteration
    ra.setdefault("summary", "Risk assessment completed.")
    ra.setdefault("evidence", [])
    ra.setdefault("status", "ok")
    result["risk_assessment"] = ra

    _log(state, "risk_manager", "exit", {"grade": ra.get("grade")})
    return result


def report_writer_node(state: InvestmentState) -> dict:
    """⑦ Report Writer — IC Memo + Red Team."""
    _log(state, "report_writer", "enter")

    if _report_node_impl is not None:
        result = _report_node_impl(state)
    else:
        ticker = state.get("target_ticker", "N/A")
        result = {"final_report": f"# Mock IC Memo — {ticker}\n\n(Mock mode)"}

    _log(state, "report_writer", "exit", {"report_len": len(result.get("final_report", ""))})
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Router
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def risk_router(state: InvestmentState) -> Literal["orchestrator", "report_writer"]:
    risk = state.get("risk_assessment", {})
    grade = risk.get("grade", "Low")
    iteration = state.get("iteration_count", 0)
    rd = risk.get("risk_decision", risk)
    fb = rd.get("orchestrator_feedback", {})

    if grade == "High" and fb.get("required", False) and iteration < MAX_ITERATIONS:
        return "orchestrator"
    return "report_writer"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Graph Assembly
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_investment_graph() -> StateGraph:
    """
    V3 Graph:
      START → orchestrator → [4 desks parallel] → barrier → risk_manager → router
        → report_writer → END
        → orchestrator (feedback loop)
    """
    g = StateGraph(InvestmentState)

    g.add_node("orchestrator", orchestrator_node)
    g.add_node("macro_analyst", macro_analyst_node)
    g.add_node("fundamental_analyst", fundamental_analyst_node)
    g.add_node("sentiment_analyst", sentiment_analyst_node)
    g.add_node("quant_analyst", quant_analyst_node)
    g.add_node("barrier", barrier_node)
    g.add_node("risk_manager", risk_manager_node)
    g.add_node("report_writer", report_writer_node)

    g.add_edge(START, "orchestrator")

    # Fan-out: orchestrator → 4 desks
    for desk in ["macro_analyst", "fundamental_analyst", "sentiment_analyst", "quant_analyst"]:
        g.add_edge("orchestrator", desk)

    # Fan-in: 4 desks → barrier
    for desk in ["macro_analyst", "fundamental_analyst", "sentiment_analyst", "quant_analyst"]:
        g.add_edge(desk, "barrier")

    g.add_edge("barrier", "risk_manager")

    g.add_conditional_edges("risk_manager", risk_router, {
        "orchestrator": "orchestrator",
        "report_writer": "report_writer",
    })

    g.add_edge("report_writer", END)

    return g


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main(mode: str = "mock", seed: int | None = 42) -> dict:
    print("🚀 7-Agent AI Investment Team V3 (Final Integration)")
    print("=" * 60)
    print(f"   Mode: {mode} | Seed: {seed}")

    state = create_initial_state(
        user_request="애플(AAPL) 주식을 지금 매수해도 괜찮을까요? 6개월 투자 관점에서 분석해 주세요.",
        mode=mode,
    )
    run_id = state["run_id"]
    print(f"   Run ID: {run_id}")

    run_dir = telemetry.init_run(run_id, mode)
    print(f"   Run Dir: {run_dir}")

    graph = build_investment_graph()
    app = graph.compile()
    final_state = app.invoke(state)

    telemetry.save_final_state(run_id, final_state)

    print("\n" + "=" * 60)
    print("✅ Pipeline Complete")
    print("=" * 60)

    # Summary
    macro = final_state.get("macro_analysis", {})
    funda = final_state.get("fundamental_analysis", {})
    senti = final_state.get("sentiment_analysis", {})
    quant = final_state.get("technical_analysis", {})
    risk = final_state.get("risk_assessment", {})

    print(f"   Run ID:       {run_id}")
    print(f"   Iterations:   {final_state.get('iteration_count', 0)}")
    print(f"   Macro Regime: {macro.get('macro_regime', 'N/A')}")
    print(f"   Structural:   {funda.get('structural_risk_flag', 'N/A')}")
    print(f"   Sentiment:    {senti.get('sentiment_regime', 'N/A')} (tilt={senti.get('tilt_factor', 'N/A')})")
    print(f"   Quant:        {quant.get('decision', 'N/A')} (alloc={quant.get('final_allocation_pct', 'N/A')})")
    print(f"   Risk Grade:   {risk.get('grade', 'N/A')}")
    print(f"   Events:       runs/{run_id}/events.jsonl")
    print(f"   State:        runs/{run_id}/final_state.json")

    report = final_state.get("final_report", "")
    if report:
        print(f"\n{'─' * 60}")
        print("📄 Final Report (first 30 lines)")
        print(f"{'─' * 60}")
        for line in report.split("\n")[:30]:
            print(line)

    print(f"\n⚠️  Confirm: No Broker API / No order placement code added. (R0)")
    return final_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="7-Agent AI Investment Team V3")
    parser.add_argument("--mode", default="mock", choices=["mock", "live"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(mode=args.mode, seed=args.seed)
