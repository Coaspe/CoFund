import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visualization.agent_empire import _build_dashboard_model, write_run_dashboard


def test_agent_empire_dashboard_builds_from_run_artifacts(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    run_id = "demo-run"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)

    final_state = {
        "run_id": run_id,
        "mode": "mock",
        "user_request": "NVDA 실적 전후 에이전트 상호작용을 보고 싶다.",
        "target_ticker": "NVDA",
        "analysis_tasks": ["macro", "fundamental", "sentiment", "quant"],
        "risk_assessment": {"grade": "Low"},
        "output_language": "ko",
    }
    (run_dir / "final_state.json").write_text(
        json.dumps(final_state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "meta.json").write_text(
        json.dumps({"run_id": run_id, "mode": "mock"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    events = [
        {"ts": "2026-03-08T09:59:59+00:00", "run_id": run_id, "node_name": "question_understanding", "iteration": 0, "phase": "enter", "inputs_summary": {"ticker": "NVDA"}, "outputs_summary": None, "errors": []},
        {"ts": "2026-03-08T10:00:00+00:00", "run_id": run_id, "node_name": "question_understanding", "iteration": 0, "phase": "exit", "inputs_summary": None, "outputs_summary": {"question_type": "single_name_analysis"}, "errors": []},
        {"ts": "2026-03-08T10:00:00+00:00", "run_id": run_id, "node_name": "orchestrator", "iteration": 0, "phase": "enter", "inputs_summary": {"ticker": "NVDA"}, "outputs_summary": None, "errors": []},
        {"ts": "2026-03-08T10:00:01+00:00", "run_id": run_id, "node_name": "orchestrator", "iteration": 0, "phase": "exit", "inputs_summary": None, "outputs_summary": {"action": "initial_delegation"}, "errors": []},
        {"ts": "2026-03-08T10:00:02+00:00", "run_id": run_id, "node_name": "macro", "iteration": 1, "phase": "enter", "inputs_summary": {"ticker": "NVDA"}, "outputs_summary": None, "errors": []},
        {"ts": "2026-03-08T10:00:03+00:00", "run_id": run_id, "node_name": "macro", "iteration": 1, "phase": "exit", "inputs_summary": None, "outputs_summary": {"regime": "expansion"}, "errors": []},
        {"ts": "2026-03-08T10:00:02+00:00", "run_id": run_id, "node_name": "fundamental", "iteration": 1, "phase": "enter", "inputs_summary": {"ticker": "NVDA"}, "outputs_summary": None, "errors": []},
        {"ts": "2026-03-08T10:00:04+00:00", "run_id": run_id, "node_name": "fundamental", "iteration": 1, "phase": "exit", "inputs_summary": None, "outputs_summary": {"decision": "bullish"}, "errors": []},
        {"ts": "2026-03-08T10:00:02+00:00", "run_id": run_id, "node_name": "sentiment", "iteration": 1, "phase": "enter", "inputs_summary": {"ticker": "NVDA"}, "outputs_summary": None, "errors": []},
        {"ts": "2026-03-08T10:00:05+00:00", "run_id": run_id, "node_name": "sentiment", "iteration": 1, "phase": "exit", "inputs_summary": None, "outputs_summary": {"tilt": 1.1}, "errors": []},
        {"ts": "2026-03-08T10:00:02+00:00", "run_id": run_id, "node_name": "quant", "iteration": 1, "phase": "enter", "inputs_summary": {"ticker": "NVDA"}, "outputs_summary": None, "errors": []},
        {"ts": "2026-03-08T10:00:03+00:00", "run_id": run_id, "node_name": "quant", "iteration": 1, "phase": "exit", "inputs_summary": None, "outputs_summary": {"decision": "HOLD"}, "errors": []},
        {"ts": "2026-03-08T10:00:06+00:00", "run_id": run_id, "node_name": "research_router", "agent_id": "research_manager", "owner_agent_id": "research_manager", "iteration": 1, "phase": "enter", "inputs_summary": {"ticker": "NVDA"}, "outputs_summary": None, "errors": []},
        {"ts": "2026-03-08T10:00:09+00:00", "run_id": run_id, "node_name": "research_router", "agent_id": "research_manager", "owner_agent_id": "research_manager", "iteration": 1, "phase": "exit", "inputs_summary": None, "outputs_summary": {"run": True, "reason": "research_need_score", "issues": 2, "actions": 1, "evidence_score": 55, "swarm_plan": {"selected_count": 2}, "handoff": {"required": False, "items": []}, "research_round": {"round": 0, "queries_executed": 0, "last_research_delta": 0, "evidence_score": 55, "stop_reason": ""}}, "errors": []},
        {"ts": "2026-03-08T10:00:10+00:00", "run_id": run_id, "node_name": "research_executor", "agent_id": "research_manager", "owner_agent_id": "research_manager", "iteration": 1, "phase": "enter", "inputs_summary": {"ticker": "NVDA"}, "outputs_summary": None, "errors": []},
        {"ts": "2026-03-08T10:00:13+00:00", "run_id": run_id, "node_name": "research_executor", "agent_id": "research_manager", "owner_agent_id": "research_manager", "iteration": 1, "phase": "exit", "inputs_summary": None, "outputs_summary": {"queries_executed": 2, "delta": 14, "score": 60, "selected_desks": ["macro", "sentiment"], "rerun": {"selected_desks": ["macro", "sentiment"], "executed_requests": 2, "executed_kinds": ["macro_headline_context"]}, "research_round": {"round": 1, "queries_executed": 2, "last_research_delta": 14, "evidence_score": 60, "stop_reason": ""}}, "errors": []},
        {"ts": "2026-03-08T10:00:14+00:00", "run_id": run_id, "node_name": "macro", "iteration": 1, "phase": "enter", "inputs_summary": {"ticker": "NVDA"}, "outputs_summary": None, "errors": []},
        {"ts": "2026-03-08T10:00:15+00:00", "run_id": run_id, "node_name": "macro", "iteration": 1, "phase": "exit", "inputs_summary": None, "outputs_summary": {"regime": "expansion"}, "errors": []},
        {"ts": "2026-03-08T10:00:14+00:00", "run_id": run_id, "node_name": "sentiment", "iteration": 1, "phase": "enter", "inputs_summary": {"ticker": "NVDA"}, "outputs_summary": None, "errors": []},
        {"ts": "2026-03-08T10:00:16+00:00", "run_id": run_id, "node_name": "sentiment", "iteration": 1, "phase": "exit", "inputs_summary": None, "outputs_summary": {"tilt": 1.05}, "errors": []},
        {"ts": "2026-03-08T10:00:17+00:00", "run_id": run_id, "node_name": "research_router", "agent_id": "research_manager", "owner_agent_id": "research_manager", "iteration": 1, "phase": "enter", "inputs_summary": {"ticker": "NVDA"}, "outputs_summary": None, "errors": []},
        {"ts": "2026-03-08T10:00:18+00:00", "run_id": run_id, "node_name": "research_router", "agent_id": "research_manager", "owner_agent_id": "research_manager", "iteration": 1, "phase": "exit", "inputs_summary": None, "outputs_summary": {"run": False, "reason": "max_research_rounds", "issues": 0, "actions": 0, "evidence_score": 60, "handoff": {"required": False, "items": []}, "research_round": {"round": 1, "queries_executed": 0, "last_research_delta": 14, "evidence_score": 60, "stop_reason": "max_research_rounds"}}, "errors": []},
        {"ts": "2026-03-08T10:00:18.500000+00:00", "run_id": run_id, "node_name": "hedge_lite_builder", "iteration": 1, "phase": "enter", "inputs_summary": {"ticker": "NVDA"}, "outputs_summary": None, "errors": []},
        {"ts": "2026-03-08T10:00:18.700000+00:00", "run_id": run_id, "node_name": "hedge_lite_builder", "iteration": 1, "phase": "exit", "inputs_summary": None, "outputs_summary": {"status": "ok"}, "errors": []},
        {"ts": "2026-03-08T10:00:18.800000+00:00", "run_id": run_id, "node_name": "portfolio_construction_quant", "iteration": 1, "phase": "enter", "inputs_summary": {"ticker": "NVDA"}, "outputs_summary": None, "errors": []},
        {"ts": "2026-03-08T10:00:18.900000+00:00", "run_id": run_id, "node_name": "portfolio_construction_quant", "iteration": 1, "phase": "exit", "inputs_summary": None, "outputs_summary": {"turnover": 0.12}, "errors": []},
        {"ts": "2026-03-08T10:00:19+00:00", "run_id": run_id, "node_name": "monitoring_router", "agent_id": "research_manager", "owner_agent_id": "research_manager", "iteration": 1, "phase": "enter", "inputs_summary": {"ticker": "NVDA"}, "outputs_summary": None, "errors": []},
        {"ts": "2026-03-08T10:00:19.100000+00:00", "run_id": run_id, "node_name": "monitoring_router", "agent_id": "research_manager", "owner_agent_id": "research_manager", "iteration": 1, "phase": "exit", "inputs_summary": None, "outputs_summary": {"selected_desks": ["macro"]}, "errors": []},
        {"ts": "2026-03-08T10:00:19+00:00", "run_id": run_id, "node_name": "risk_manager", "iteration": 1, "phase": "enter", "inputs_summary": {"ticker": "NVDA"}, "outputs_summary": None, "errors": []},
        {"ts": "2026-03-08T10:00:20+00:00", "run_id": run_id, "node_name": "risk_manager", "iteration": 1, "phase": "exit", "inputs_summary": None, "outputs_summary": {"grade": "Low"}, "errors": []},
        {"ts": "2026-03-08T10:00:21+00:00", "run_id": run_id, "node_name": "report_writer", "iteration": 1, "phase": "enter", "inputs_summary": {"ticker": "NVDA"}, "outputs_summary": None, "errors": []},
        {"ts": "2026-03-08T10:00:22+00:00", "run_id": run_id, "node_name": "report_writer", "iteration": 1, "phase": "exit", "inputs_summary": None, "outputs_summary": {"report_len": 1234}, "errors": []},
    ]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )

    model = _build_dashboard_model(run_id, runs_dir=runs_dir)
    nodes = {node["id"]: node for node in model["nodes"]}
    map_nodes = {node["id"] for node in model["map_nodes"]}
    standby_nodes = {node["id"] for node in model["standby_nodes"]}
    edges = {(edge["source"], edge["target"]): edge["count"] for edge in model["edges"]}

    assert nodes["macro"]["enter_count"] == 2
    assert nodes["macro"]["exit_count"] == 2
    assert nodes["question_understanding"]["enter_count"] == 1
    assert nodes["portfolio_construction_quant"]["enter_count"] == 1
    assert nodes["research_manager"]["enter_count"] == 4
    assert nodes["research_manager"]["exit_count"] == 4
    assert "monitoring_router" not in nodes
    assert "research_router" not in nodes
    assert "research_executor" not in nodes
    assert nodes["macro"]["avatar"] == "M"
    assert nodes["macro"]["room_label"] == "Macro Bay"
    assert nodes["question_understanding"]["avatar"] == "QI"
    assert nodes["macro"]["render_mode"] == "card"
    assert "macro" in map_nodes
    assert "human_handoff" not in nodes
    assert edges[("orchestrator", "macro")] == 1
    assert edges[("question_understanding", "orchestrator")] == 1
    assert edges[("macro", "hedge_lite_builder")] == 2
    assert edges[("hedge_lite_builder", "portfolio_construction_quant")] == 1
    assert edges[("portfolio_construction_quant", "research_manager")] == 1
    assert edges[("research_manager", "macro")] == 1
    assert edges[("research_manager", "risk_manager")] == 1
    assert edges[("risk_manager", "report_writer")] == 1
    assert model["aux_nodes"] == []
    research_timeline = [item for item in model["timeline"] if item["node"] == "research_manager"]
    assert research_timeline
    assert any(item["node_name"] == "monitoring_router" for item in research_timeline)
    assert any(item["node_name"] == "research_router" for item in research_timeline)
    assert any(item["node_name"] == "research_executor" for item in research_timeline)
    assert all(item["agent_id"] == "research_manager" for item in research_timeline)
    assert all(item["owner_agent_id"] == "research_manager" for item in research_timeline)

    output_path = write_run_dashboard(run_id, runs_dir=runs_dir)
    html = output_path.read_text(encoding="utf-8")

    assert output_path.exists()
    assert "Agent Empire Replay" in html
    assert run_id in html
    assert "character-layer" in html
    assert "room-zone" in html
    assert "Standby Units" in html
    assert "Live Work" in html
    assert "No exit summary recorded." not in html
    assert "Reset Layout" in html
    assert "localStorage" in html
    assert "pointerdown" in html
    assert "current work" in html
    assert "execution node" in html


def test_agent_empire_dashboard_groups_graph_node_events_by_agent_id(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    run_id = "agent-id-run"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)

    final_state = {
        "run_id": run_id,
        "mode": "mock",
        "user_request": "NVDA 리서치 재실행 흐름을 보고 싶다.",
        "target_ticker": "NVDA",
        "analysis_tasks": ["macro"],
        "risk_assessment": {"grade": "Low"},
        "output_language": "ko",
    }
    (run_dir / "final_state.json").write_text(
        json.dumps(final_state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "meta.json").write_text(
        json.dumps({"run_id": run_id, "mode": "mock"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    events = [
        {"ts": "2026-03-08T10:00:00+00:00", "run_id": run_id, "node_name": "question_understanding", "iteration": 0, "phase": "enter", "inputs_summary": {"ticker": "NVDA"}, "outputs_summary": None, "errors": []},
        {"ts": "2026-03-08T10:00:01+00:00", "run_id": run_id, "node_name": "question_understanding", "iteration": 0, "phase": "exit", "inputs_summary": None, "outputs_summary": {"question_type": "single_name_analysis"}, "errors": []},
        {"ts": "2026-03-08T10:00:02+00:00", "run_id": run_id, "node_name": "orchestrator", "agent_id": "orchestrator", "iteration": 0, "phase": "enter", "inputs_summary": {"ticker": "NVDA"}, "outputs_summary": None, "errors": []},
        {"ts": "2026-03-08T10:00:03+00:00", "run_id": run_id, "node_name": "orchestrator", "agent_id": "orchestrator", "iteration": 0, "phase": "exit", "inputs_summary": None, "outputs_summary": {"action": "initial_delegation"}, "errors": []},
        {"ts": "2026-03-08T10:00:04+00:00", "run_id": run_id, "node_name": "macro_analyst", "agent_id": "macro", "iteration": 1, "phase": "enter", "inputs_summary": {"ticker": "NVDA"}, "outputs_summary": None, "errors": []},
        {"ts": "2026-03-08T10:00:05+00:00", "run_id": run_id, "node_name": "macro_analyst", "agent_id": "macro", "iteration": 1, "phase": "exit", "inputs_summary": None, "outputs_summary": {"regime": "expansion"}, "errors": []},
        {"ts": "2026-03-08T10:00:06+00:00", "run_id": run_id, "node_name": "research_executor", "agent_id": "research_manager", "owner_agent_id": "research_manager", "iteration": 1, "phase": "enter", "inputs_summary": {"ticker": "NVDA"}, "outputs_summary": None, "errors": []},
        {"ts": "2026-03-08T10:00:07+00:00", "run_id": run_id, "node_name": "rerun_selector", "iteration": 1, "phase": "exit", "inputs_summary": None, "outputs_summary": {"selected_desks": ["macro"], "executed_requests": 1}, "errors": []},
        {"ts": "2026-03-08T10:00:08+00:00", "run_id": run_id, "node_name": "research_executor", "agent_id": "research_manager", "owner_agent_id": "research_manager", "iteration": 1, "phase": "exit", "inputs_summary": None, "outputs_summary": {"queries_executed": 1}, "errors": []},
        {"ts": "2026-03-08T10:00:09+00:00", "run_id": run_id, "node_name": "macro_analyst_research", "agent_id": "macro", "iteration": 1, "phase": "enter", "inputs_summary": {"ticker": "NVDA"}, "outputs_summary": None, "errors": []},
        {"ts": "2026-03-08T10:00:10+00:00", "run_id": run_id, "node_name": "macro_analyst_research", "agent_id": "macro", "iteration": 1, "phase": "exit", "inputs_summary": None, "outputs_summary": {"regime": "expansion"}, "errors": []},
        {"ts": "2026-03-08T10:00:11+00:00", "run_id": run_id, "node_name": "risk_manager", "agent_id": "risk_manager", "iteration": 1, "phase": "enter", "inputs_summary": {"ticker": "NVDA"}, "outputs_summary": None, "errors": []},
        {"ts": "2026-03-08T10:00:12+00:00", "run_id": run_id, "node_name": "risk_manager", "agent_id": "risk_manager", "iteration": 1, "phase": "exit", "inputs_summary": None, "outputs_summary": {"grade": "Low"}, "errors": []},
        {"ts": "2026-03-08T10:00:13+00:00", "run_id": run_id, "node_name": "report_writer", "agent_id": "report_writer", "iteration": 1, "phase": "enter", "inputs_summary": {"ticker": "NVDA"}, "outputs_summary": None, "errors": []},
        {"ts": "2026-03-08T10:00:14+00:00", "run_id": run_id, "node_name": "report_writer", "agent_id": "report_writer", "iteration": 1, "phase": "exit", "inputs_summary": None, "outputs_summary": {"report_len": 321}, "errors": []},
    ]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )

    model = _build_dashboard_model(run_id, runs_dir=runs_dir)
    nodes = {node["id"]: node for node in model["nodes"]}
    aux_names = {node["name"] for node in model["aux_nodes"]}
    edges = {(edge["source"], edge["target"]): edge["count"] for edge in model["edges"]}

    assert nodes["macro"]["enter_count"] == 2
    assert nodes["macro"]["exit_count"] == 2
    assert "macro_analyst" not in aux_names
    assert "macro_analyst_research" not in aux_names
    assert edges[("orchestrator", "macro")] == 1
    assert edges[("research_manager", "macro")] == 1

    macro_timeline = [item for item in model["timeline"] if item["node"] == "macro"]
    research_exec_timeline = [item for item in model["timeline"] if item["node"] == "research_manager"]
    assert any(item["node_name"] == "macro_analyst" for item in macro_timeline)
    assert any(item["node_name"] == "macro_analyst_research" for item in macro_timeline)
    assert all(item["agent_id"] == "macro" for item in macro_timeline)
    assert research_exec_timeline
    assert any(item["node_name"] == "research_executor" for item in research_exec_timeline)
    assert all(item["agent_id"] == "research_manager" for item in research_exec_timeline)
    assert all(item["owner_agent_id"] == "research_manager" for item in research_exec_timeline)
