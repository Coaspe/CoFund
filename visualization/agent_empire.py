"""
visualization/agent_empire.py
=============================
Render a self-contained HTML replay dashboard for a run's agent activity.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from runtime_identity import dashboard_node_id_for_event, event_agent_id, event_node_name, event_owner_agent_id


RUNS_DIR = Path("runs")

CHARACTER_META: dict[str, dict[str, Any]] = {
    "question_understanding": {
        "avatar": "QI",
        "accent": "#8fd4ff",
        "room_label": "Frontdoor Intake",
        "home_dx": 0.0,
        "home_dy": 15.0,
    },
    "orchestrator": {
        "avatar": "CIO",
        "accent": "#ff8a3d",
        "room_label": "Command Deck",
        "home_dx": 0.0,
        "home_dy": 15.0,
    },
    "research_manager": {
        "avatar": "RG",
        "accent": "#f4c152",
        "room_label": "Research Ops",
        "home_dx": 3.0,
        "home_dy": 12.0,
    },
    "macro": {
        "avatar": "M",
        "accent": "#2fd5c4",
        "room_label": "Macro Bay",
        "home_dx": -5.0,
        "home_dy": 13.0,
    },
    "fundamental": {
        "avatar": "F",
        "accent": "#59e08f",
        "room_label": "Funda Lab",
        "home_dx": -1.0,
        "home_dy": 13.0,
    },
    "sentiment": {
        "avatar": "S",
        "accent": "#65b8ff",
        "room_label": "Narrative Desk",
        "home_dx": 1.0,
        "home_dy": 13.0,
    },
    "quant": {
        "avatar": "Q",
        "accent": "#ffcf5a",
        "room_label": "Quant Bench",
        "home_dx": 5.0,
        "home_dy": 13.0,
    },
    "hedge_lite_builder": {
        "avatar": "HL",
        "accent": "#e7db7a",
        "room_label": "Hedge Lite",
        "home_dx": -4.0,
        "home_dy": 11.5,
    },
    "portfolio_construction_quant": {
        "avatar": "PC",
        "accent": "#ffd966",
        "room_label": "Construction Desk",
        "home_dx": 0.0,
        "home_dy": 11.5,
    },
    "risk_manager": {
        "avatar": "RM",
        "accent": "#ff6e78",
        "room_label": "Risk Gate",
        "home_dx": -2.0,
        "home_dy": 9.5,
    },
    "report_writer": {
        "avatar": "RW",
        "accent": "#d7b3ff",
        "room_label": "Memo Studio",
        "home_dx": 2.0,
        "home_dy": 9.5,
    },
}

PRIMARY_NODE_META: dict[str, dict[str, Any]] = {
    "question_understanding": {
        "label": "Question Intake",
        "subtitle": "Frontdoor Parser",
        "group": "control",
        "x": 50,
        "y": 7,
    },
    "orchestrator": {
        "label": "Orchestrator",
        "subtitle": "CIO / Mission Control",
        "group": "control",
        "x": 50,
        "y": 16,
    },
    "research_manager": {
        "label": "Research Manager",
        "subtitle": "Monitoring / Evidence Ops",
        "group": "research",
        "x": 68,
        "y": 58,
    },
    "macro": {
        "label": "Macro",
        "subtitle": "Regime Desk",
        "group": "desk",
        "x": 16,
        "y": 30,
    },
    "fundamental": {
        "label": "Fundamental",
        "subtitle": "Company Desk",
        "group": "desk",
        "x": 36,
        "y": 32,
    },
    "sentiment": {
        "label": "Sentiment",
        "subtitle": "Narrative Desk",
        "group": "desk",
        "x": 64,
        "y": 32,
    },
    "quant": {
        "label": "Quant",
        "subtitle": "Sizing Desk",
        "group": "desk",
        "x": 84,
        "y": 30,
    },
    "hedge_lite_builder": {
        "label": "Hedge Lite",
        "subtitle": "Cross-Asset Screen",
        "group": "research",
        "x": 28,
        "y": 50,
    },
    "portfolio_construction_quant": {
        "label": "Portfolio Construction",
        "subtitle": "Assembly Quant",
        "group": "desk",
        "x": 50,
        "y": 46,
    },
    "risk_manager": {
        "label": "Risk Manager",
        "subtitle": "5-Gate Barrier",
        "group": "control",
        "x": 38,
        "y": 86,
    },
    "report_writer": {
        "label": "Report Writer",
        "subtitle": "Memo Composer",
        "group": "control",
        "x": 64,
        "y": 86,
    },
}

GROUP_LABELS = {
    "control": "Control",
    "desk": "Desk",
    "research": "Research",
    "aux": "Auxiliary",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        events.append(json.loads(raw))
    events.sort(key=lambda item: str(item.get("ts", "")))
    return events


def _parse_ts(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _short_text(value: Any, max_len: int = 88) -> str:
    text = str(value or "").strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 1:
        return f"{value * 1000:.0f} ms"
    if value < 60:
        return f"{value:.1f}s"
    minutes = int(value // 60)
    seconds = int(value % 60)
    return f"{minutes}m {seconds:02d}s"


def _event_outputs(event: dict[str, Any]) -> dict[str, Any]:
    outputs = event.get("outputs_summary", {})
    return outputs if isinstance(outputs, dict) else {}


def _research_round_payload(event: dict[str, Any]) -> dict[str, Any]:
    outputs = _event_outputs(event)
    if event_node_name(event) == "research_round":
        return outputs
    nested = outputs.get("research_round", {})
    return nested if isinstance(nested, dict) else {}


def _rerun_selected_desks(event: dict[str, Any]) -> list[str]:
    outputs = _event_outputs(event)
    if event_node_name(event) == "rerun_selector":
        raw = outputs.get("selected_desks", [])
    elif event_node_name(event) == "research_executor":
        rerun = outputs.get("rerun", {})
        if isinstance(rerun, dict) and isinstance(rerun.get("selected_desks"), list):
            raw = rerun.get("selected_desks", [])
        else:
            raw = outputs.get("selected_desks", [])
    else:
        raw = []
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        key = str(item or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _count_raw_node_events(
    events: list[dict[str, Any]],
    *,
    node_name: str,
    phase: str,
) -> int:
    target_node = str(node_name or "").strip()
    target_phase = str(phase or "").strip().lower()
    count = 0
    for event in events:
        if event_node_name(event) != target_node:
            continue
        if str(event.get("phase", "")).strip().lower() != target_phase:
            continue
        count += 1
    return count


def _summary_text(payload: Any) -> str:
    if not isinstance(payload, dict) or not payload:
        return ""
    ordered_keys = [
        "action",
        "decision",
        "grade",
        "regime",
        "tilt",
        "run",
        "reason",
        "report_len",
        "queries_executed",
        "selected_desks",
        "issues",
        "actions",
        "llm_enrichment_status",
    ]
    parts: list[str] = []
    used: set[str] = set()
    for key in ordered_keys:
        if key not in payload:
            continue
        value = payload.get(key)
        if value in (None, "", [], {}):
            continue
        used.add(key)
        if isinstance(value, list):
            rendered = ",".join(str(item) for item in value[:4])
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
        if len(parts) >= 3:
            break
    if not parts:
        for key, value in payload.items():
            if key in used or value in (None, "", [], {}):
                continue
            if isinstance(value, dict):
                continue
            if isinstance(value, list):
                rendered = ",".join(str(item) for item in value[:4])
            else:
                rendered = str(value)
            parts.append(f"{key}={rendered}")
            if len(parts) >= 3:
                break
    return _short_text(" | ".join(parts), max_len=120)


def _load_run_artifacts(run_id: str, runs_dir: Path) -> dict[str, Any]:
    run_dir = runs_dir / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory not found: {run_dir}")
    events = _read_events(run_dir / "events.jsonl")
    if not events:
        raise FileNotFoundError(f"events.jsonl not found or empty: {run_dir / 'events.jsonl'}")
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "meta": _read_json(run_dir / "meta.json"),
        "final_state": _read_json(run_dir / "final_state.json"),
        "events": events,
    }


def list_runs(runs_dir: Path = RUNS_DIR, *, limit: int = 40) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not runs_dir.exists():
        return runs

    now_ts = datetime.now().timestamp()
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        events_path = run_dir / "events.jsonl"
        if not events_path.exists():
            continue
        final_state_path = run_dir / "final_state.json"
        meta = _read_json(run_dir / "meta.json")
        final_state = _read_json(final_state_path)
        lines = events_path.read_text(encoding="utf-8").splitlines()
        events = _read_events(events_path)
        updated_at = datetime.fromtimestamp(events_path.stat().st_mtime).isoformat()
        started_at = str(meta.get("created_at") or (events[0].get("ts") if events else "") or updated_at)
        age_seconds = max(0.0, now_ts - events_path.stat().st_mtime)
        is_complete = final_state_path.exists() and bool(final_state)
        is_running = (not is_complete) and age_seconds <= 12.0
        status = "running" if is_running else ("complete" if is_complete else "incomplete")
        request_text = (
            final_state.get("user_request")
            or meta.get("user_request")
            or ""
        )
        runs.append(
            {
                "run_id": run_dir.name,
                "mode": str(final_state.get("mode") or meta.get("mode") or "unknown"),
                "started_at": started_at,
                "updated_at": updated_at,
                "event_count": len([line for line in lines if line.strip()]),
                "target_ticker": str(final_state.get("target_ticker") or final_state.get("ticker") or ""),
                "request": _short_text(request_text, max_len=120),
                "risk_grade": str(
                    ((final_state.get("risk_assessment") or {}) if isinstance(final_state.get("risk_assessment"), dict) else {}).get("grade")
                    or ""
                ),
                "has_final_state": bool(final_state),
                "has_dashboard": (run_dir / "agent_empire.html").exists(),
                "is_running": is_running,
                "is_complete": is_complete,
                "status": status,
                "age_seconds": round(age_seconds, 3),
            }
        )

    runs.sort(
        key=lambda item: (
            str(item.get("started_at", "")),
            str(item.get("updated_at", "")),
        ),
        reverse=True,
    )
    return runs[:limit]


def _infer_event_flow(event: dict[str, Any]) -> dict[str, str] | None:
    raw_node = event_node_name(event)
    node = dashboard_node_id_for_event(event)
    phase = str(event.get("phase", "")).strip().lower()
    if phase == "enter":
        if raw_node == "question_understanding":
            return None
        if raw_node == "orchestrator":
            source = "risk_manager" if (_coerce_int(event.get("iteration")) or 0) > 0 else "question_understanding"
            return {"source": source, "target": node}
        if raw_node in {
            "macro",
            "macro_analyst",
            "macro_analyst_research",
            "fundamental",
            "fundamental_analyst",
            "fundamental_analyst_research",
            "sentiment",
            "sentiment_analyst",
            "sentiment_analyst_research",
            "quant",
            "quant_analyst",
            "quant_analyst_research",
        }:
            source = "research_manager" if raw_node.endswith("_research") else "orchestrator"
            return {"source": source, "target": node}
        if raw_node == "hedge_lite_builder":
            source = "research_manager" if (_coerce_int(event.get("iteration")) or 0) > 1 else "quant"
            return {"source": source, "target": node}
        if raw_node == "portfolio_construction_quant":
            return {"source": "hedge_lite_builder", "target": node}
        if raw_node == "monitoring_router":
            return {"source": "portfolio_construction_quant", "target": node}
        if raw_node in {"research_router", "research_executor", "research_barrier"}:
            return None
        if raw_node == "risk_manager":
            return {"source": "research_manager", "target": node}
        if raw_node == "report_writer":
            return {"source": "risk_manager", "target": node}
    return None


def _build_dashboard_model(run_id: str, runs_dir: Path = RUNS_DIR) -> dict[str, Any]:
    artifacts = _load_run_artifacts(run_id, runs_dir)
    events = artifacts["events"]
    final_state = artifacts["final_state"]
    meta = artifacts["meta"]
    run_dir = artifacts["run_dir"]

    node_stats: dict[str, dict[str, Any]] = {}
    durations: dict[tuple[str, int], list[datetime]] = defaultdict(list)
    aux_counter: Counter[str] = Counter()
    unique_iterations: set[int] = set()

    start_ts = _parse_ts(events[0].get("ts"))
    end_ts = _parse_ts(events[-1].get("ts"))

    for event in events:
        node = dashboard_node_id_for_event(event)
        iteration = _coerce_int(event.get("iteration")) or 0
        unique_iterations.add(iteration)
        stats = node_stats.setdefault(
            node,
            {
                "node_id": node,
                "enter_count": 0,
                "exit_count": 0,
                "error_count": 0,
                "iteration_count": 0,
                "last_phase": "idle",
                "last_ts": "",
                "last_summary": "",
                "avg_duration_seconds": None,
                "total_duration_seconds": 0.0,
                "duration_samples": 0,
                "activity_score": 0,
            },
        )
        if iteration not in stats.setdefault("_iterations", set()):
            stats["_iterations"].add(iteration)
        phase = str(event.get("phase", "")).strip().lower()
        ts = _parse_ts(event.get("ts"))
        if phase == "enter":
            stats["enter_count"] += 1
            if ts is not None:
                durations[(node, iteration)].append(ts)
        elif phase == "exit":
            stats["exit_count"] += 1
            if ts is not None and durations[(node, iteration)]:
                start = durations[(node, iteration)].pop(0)
                duration = max((ts - start).total_seconds(), 0.0)
                stats["total_duration_seconds"] += duration
                stats["duration_samples"] += 1
            summary = _summary_text(event.get("outputs_summary"))
            if summary:
                stats["last_summary"] = summary
        errors = event.get("errors", [])
        if isinstance(errors, list):
            stats["error_count"] += len(errors)
        stats["last_phase"] = phase or "idle"
        stats["last_ts"] = str(event.get("ts", ""))
        stats["activity_score"] = (
            stats["enter_count"] + stats["exit_count"] + stats["error_count"] * 2
        )
        if node not in PRIMARY_NODE_META:
            aux_counter[node] += 1

    for stats in node_stats.values():
        stats["iteration_count"] = len(stats.pop("_iterations", set()))
        if stats["duration_samples"] > 0:
            stats["avg_duration_seconds"] = round(
                stats["total_duration_seconds"] / stats["duration_samples"], 3
            )
        stats["total_duration_seconds"] = round(stats["total_duration_seconds"], 3)

    analysis_tasks = list(final_state.get("analysis_tasks", []) or [])
    interaction_counts: Counter[tuple[str, str]] = Counter()

    intake_hits = int(node_stats.get("question_understanding", {}).get("exit_count", 0))
    if intake_hits > 0:
        interaction_counts[("question_understanding", "orchestrator")] += intake_hits

    for desk in analysis_tasks:
        if desk in PRIMARY_NODE_META and node_stats.get(desk, {}).get("enter_count", 0) > 0:
            interaction_counts[("orchestrator", desk)] += 1

    hedge_hits = int(node_stats.get("hedge_lite_builder", {}).get("enter_count", 0))
    if hedge_hits > 0:
        for desk in ("macro", "fundamental", "sentiment", "quant"):
            desk_exits = int(node_stats.get(desk, {}).get("exit_count", 0))
            if desk_exits > 0:
                interaction_counts[(desk, "hedge_lite_builder")] += desk_exits

    construction_hits = int(node_stats.get("portfolio_construction_quant", {}).get("enter_count", 0))
    if construction_hits > 0:
        interaction_counts[("hedge_lite_builder", "portfolio_construction_quant")] += construction_hits

    monitoring_hits = _count_raw_node_events(events, node_name="monitoring_router", phase="enter")
    if monitoring_hits > 0:
        interaction_counts[("portfolio_construction_quant", "research_manager")] += monitoring_hits

    risk_from_research_hits = 0
    for event in events:
        if event_node_name(event) != "research_router":
            continue
        if str(event.get("phase", "")).strip().lower() != "exit":
            continue
        outputs = _event_outputs(event)
        if outputs.get("run") is False:
            risk_from_research_hits += 1

    risk_hits = int(node_stats.get("risk_manager", {}).get("enter_count", 0))
    if risk_from_research_hits > 0:
        interaction_counts[("research_manager", "risk_manager")] += risk_from_research_hits
    if risk_hits > risk_from_research_hits:
        interaction_counts[("barrier", "risk_manager")] += (risk_hits - risk_from_research_hits)

    report_hits = int(node_stats.get("report_writer", {}).get("enter_count", 0))
    if report_hits > 0:
        interaction_counts[("risk_manager", "report_writer")] += report_hits

    for event in events:
        if (
            event_node_name(event) == "orchestrator"
            and str(event.get("phase", "")).strip().lower() == "enter"
            and (_coerce_int(event.get("iteration")) or 0) > 0
        ):
            interaction_counts[("risk_manager", "orchestrator")] += 1

    for event in events:
        for desk in _rerun_selected_desks(event):
            if str(desk).strip() in PRIMARY_NODE_META:
                interaction_counts[("research_manager", str(desk).strip())] += 1

    primary_nodes: list[dict[str, Any]] = []
    for node_id, meta_row in PRIMARY_NODE_META.items():
        stats = node_stats.get(node_id, {})
        char_row = CHARACTER_META.get(node_id, {})
        primary_nodes.append(
            {
                "id": node_id,
                "label": meta_row["label"],
                "subtitle": meta_row["subtitle"],
                "group": meta_row["group"],
                "group_label": GROUP_LABELS.get(meta_row["group"], meta_row["group"].title()),
                "x": meta_row["x"],
                "y": meta_row["y"],
                "visited": bool(stats),
                "enter_count": int(stats.get("enter_count", 0)),
                "exit_count": int(stats.get("exit_count", 0)),
                "error_count": int(stats.get("error_count", 0)),
                "iteration_count": int(stats.get("iteration_count", 0)),
                "last_phase": str(stats.get("last_phase", "idle")),
                "last_summary": str(stats.get("last_summary", "")),
                "avg_duration_label": _format_seconds(_coerce_float(stats.get("avg_duration_seconds"))),
                "activity_score": int(stats.get("activity_score", 0)),
                "avatar": str(char_row.get("avatar", meta_row["label"][:2].upper())),
                "accent": str(char_row.get("accent", "#2fd5c4")),
                "room_label": str(char_row.get("room_label", meta_row["label"])),
                "home_dx": float(char_row.get("home_dx", 0.0)),
                "home_dy": float(char_row.get("home_dy", 10.0)),
            }
        )

    engaged_node_ids: set[str] = {
        node["id"]
        for node in primary_nodes
        if node["enter_count"] > 0 or node["exit_count"] > 0 or node["error_count"] > 0
    }

    edges: list[dict[str, Any]] = []
    for (source, target), count in sorted(
        interaction_counts.items(),
        key=lambda item: (PRIMARY_NODE_META.get(item[0][0], {}).get("y", 999), item[0][0], item[0][1]),
    ):
        if source not in PRIMARY_NODE_META or target not in PRIMARY_NODE_META:
            continue
        edges.append(
            {
                "id": f"{source}->{target}",
                "source": source,
                "target": target,
                "count": count,
                "label": f"{count}x" if count > 0 else "",
            }
        )
        if count > 0:
            engaged_node_ids.add(source)
            engaged_node_ids.add(target)

    for node in primary_nodes:
        is_engaged = node["id"] in engaged_node_ids
        node["is_engaged"] = is_engaged
        node["render_mode"] = "card" if is_engaged else "standby"

    map_nodes = [node for node in primary_nodes if node["render_mode"] == "card"]
    standby_nodes = [node for node in primary_nodes if node["render_mode"] == "standby"]

    research_rounds = 0
    evidence_score = None
    for event in events:
        payload = _research_round_payload(event)
        if not payload:
            continue
        research_rounds = max(research_rounds, _coerce_int(payload.get("research_round") or payload.get("round")) or 0)
        score = _coerce_int(payload.get("evidence_score"))
        if score is not None:
            evidence_score = score

    timeline: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        ts = _parse_ts(event.get("ts"))
        offset_seconds = 0.0
        if ts is not None and start_ts is not None:
            offset_seconds = max((ts - start_ts).total_seconds(), 0.0)
        raw_node = event_node_name(event) or "unknown"
        node = dashboard_node_id_for_event(event)
        phase = str(event.get("phase", "")).strip().lower() or "event"
        payload = event.get("outputs_summary") if phase == "exit" else event.get("inputs_summary")
        flow = _infer_event_flow(event)
        timeline.append(
            {
                "index": index,
                "node": node,
                "node_name": raw_node,
                "agent_id": event_agent_id(event),
                "owner_agent_id": event_owner_agent_id(event),
                "label": PRIMARY_NODE_META.get(node, {}).get("label", raw_node.replace("_", " ").title()),
                "phase": phase,
                "iteration": _coerce_int(event.get("iteration")) or 0,
                "ts": str(event.get("ts", "")),
                "offset_seconds": round(offset_seconds, 3),
                "detail": _summary_text(payload) or "no summary",
                "errors": list(event.get("errors", []) or []),
                "flow": flow,
            }
        )

    request_text = (
        final_state.get("user_request")
        or meta.get("user_request")
        or "n/a"
    )
    target_ticker = (
        final_state.get("target_ticker")
        or final_state.get("ticker")
        or "n/a"
    )
    risk_grade = (
        ((final_state.get("risk_assessment") or {}) if isinstance(final_state.get("risk_assessment"), dict) else {}).get("grade")
        or "n/a"
    )
    output_language = final_state.get("output_language") or "n/a"
    mode = final_state.get("mode") or meta.get("mode") or "unknown"
    duration_seconds = None
    if start_ts is not None and end_ts is not None:
        duration_seconds = max((end_ts - start_ts).total_seconds(), 0.0)
    event_count = len(events)
    is_complete = (run_dir / "final_state.json").exists() and bool(final_state)
    age_seconds = max(0.0, datetime.now().timestamp() - (run_dir / "events.jsonl").stat().st_mtime)
    is_running = (not is_complete) and age_seconds <= 12.0
    status = "running" if is_running else ("complete" if is_complete else "incomplete")

    aux_nodes = [
        {"name": name, "events": count}
        for name, count in aux_counter.most_common()
        if name not in PRIMARY_NODE_META
    ]

    metrics = [
        {"label": "Run", "value": run_id},
        {"label": "Mode", "value": str(mode)},
        {"label": "Ticker", "value": str(target_ticker)},
        {"label": "Risk", "value": str(risk_grade)},
        {"label": "Iterations", "value": str(max(unique_iterations) if unique_iterations else 0)},
        {"label": "Duration", "value": _format_seconds(duration_seconds)},
        {"label": "Events", "value": str(len(events))},
        {"label": "Research Rounds", "value": str(research_rounds)},
        {"label": "Evidence Score", "value": str(evidence_score if evidence_score is not None else "n/a")},
        {"label": "Status", "value": status},
        {"label": "Language", "value": str(output_language)},
    ]

    return {
        "run_id": run_id,
        "title": "Agent Empire Replay",
        "subtitle": "AI investment activity map",
        "request": _short_text(request_text, max_len=220),
        "as_of": str(final_state.get("as_of") or meta.get("created_at") or ""),
        "metrics": metrics,
        "nodes": primary_nodes,
        "map_nodes": map_nodes,
        "standby_nodes": standby_nodes,
        "edges": edges,
        "timeline": timeline,
        "aux_nodes": aux_nodes,
        "analysis_tasks": analysis_tasks,
        "event_count": event_count,
        "is_running": is_running,
        "is_complete": is_complete,
        "status": status,
        "updated_at": str(events[-1].get("ts", "")) if events else "",
    }


def _json_for_script(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False).replace("</script>", "<\\/script>")


def _render_html(model: dict[str, Any]) -> str:
    template = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent Empire Replay</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;700&display=swap');

    :root {
      --bg: #08101d;
      --bg2: #14213d;
      --panel: rgba(10, 17, 30, 0.82);
      --panel-strong: rgba(7, 12, 22, 0.92);
      --line: rgba(126, 149, 181, 0.22);
      --text: #ecf4ff;
      --muted: #9db1cb;
      --accent-control: #ff8a3d;
      --accent-desk: #2fd5c4;
      --accent-research: #f4c152;
      --accent-aux: #8c96ad;
      --accent-danger: #ff5e6b;
      --glow: rgba(47, 213, 196, 0.24);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(244, 193, 82, 0.14), transparent 30%),
        radial-gradient(circle at 90% 15%, rgba(47, 213, 196, 0.16), transparent 28%),
        linear-gradient(180deg, #0a1220 0%, #08101d 52%, #111b2d 100%);
      font-family: "Space Grotesk", system-ui, sans-serif;
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      background-image:
        linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
      background-size: 28px 28px;
      pointer-events: none;
      opacity: 0.35;
    }

    .shell {
      position: relative;
      width: min(1720px, calc(100% - 20px));
      margin: 12px auto 24px;
      z-index: 1;
    }

    .hero {
      display: grid;
      grid-template-columns: 1.3fr 0.7fr;
      gap: 14px;
      margin-bottom: 14px;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 18px 60px rgba(0, 0, 0, 0.28);
      backdrop-filter: blur(14px);
    }

    .hero-main {
      padding: 22px 24px;
      overflow: hidden;
    }

    .hero-kicker {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--muted);
      font-family: "IBM Plex Mono", monospace;
    }

    .hero h1 {
      margin: 12px 0 10px;
      font-size: clamp(30px, 4vw, 52px);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }

    .hero p {
      margin: 0;
      max-width: 70ch;
      color: var(--muted);
      line-height: 1.6;
    }

    .hero-side {
      padding: 18px 18px 16px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    .metric {
      padding: 12px;
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .metric-label {
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      font-family: "IBM Plex Mono", monospace;
    }

    .metric-value {
      margin-top: 6px;
      font-size: 15px;
      font-weight: 600;
      word-break: break-word;
    }

    .hero-strip {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .hero-chip {
      padding: 7px 10px;
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(255, 255, 255, 0.03);
      color: var(--muted);
      font-size: 12px;
      font-family: "IBM Plex Mono", monospace;
    }

    .layout {
      display: grid;
      gap: 14px;
    }

    .support-grid {
      display: grid;
      grid-template-columns: minmax(300px, 0.78fr) minmax(0, 1.22fr);
      gap: 14px;
      align-items: start;
    }

    .insights-column {
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .arena {
      padding: 18px;
    }

    .section-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }

    .section-head-actions {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    .section-head h2,
    .side-section h2 {
      margin: 0;
      font-size: 14px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--muted);
      font-family: "IBM Plex Mono", monospace;
    }

    .section-subtle {
      color: var(--muted);
      font-size: 12px;
    }

    .arena-stage {
      position: relative;
      min-height: clamp(940px, 78vh, 1180px);
      border-radius: 24px;
      border: 1px solid rgba(255, 255, 255, 0.05);
      background:
        radial-gradient(circle at center, rgba(47, 213, 196, 0.08), transparent 30%),
        linear-gradient(180deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.01));
      overflow: hidden;
    }

    .arena-stage::before {
      content: "";
      position: absolute;
      inset: 0;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.05), transparent 30%),
        repeating-linear-gradient(
          180deg,
          transparent,
          transparent 14px,
          rgba(255, 255, 255, 0.015) 14px,
          rgba(255, 255, 255, 0.015) 15px
        );
      pointer-events: none;
    }

    .standby-dock-wrap {
      margin-top: 14px;
      padding: 14px 16px 16px;
      border-radius: 18px;
      border: 1px solid rgba(255, 255, 255, 0.06);
      background: rgba(255, 255, 255, 0.025);
    }

    .standby-head {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }

    .standby-headline {
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      font-family: "IBM Plex Mono", monospace;
    }

    .standby-note {
      color: var(--muted);
      font-size: 12px;
    }

    .standby-dock {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    .standby-chip {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 9px 12px;
      border-radius: 14px;
      border: 1px solid rgba(255, 255, 255, 0.06);
      background: rgba(7, 12, 22, 0.74);
      min-width: 0;
    }

    .standby-avatar {
      flex: 0 0 auto;
      width: 28px;
      height: 28px;
      display: grid;
      place-items: center;
      border-radius: 10px;
      color: #07111d;
      font: 700 11px "IBM Plex Mono", monospace;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.26), rgba(255, 255, 255, 0.1)),
        color-mix(in srgb, var(--standby-accent) 74%, #f0e3cf);
    }

    .standby-text {
      min-width: 0;
    }

    .standby-label {
      font-size: 13px;
      font-weight: 600;
    }

    .standby-meta {
      margin-top: 2px;
      color: var(--muted);
      font-size: 11px;
      font-family: "IBM Plex Mono", monospace;
      white-space: nowrap;
    }

    #zone-layer,
    #edge-layer,
    #character-layer,
    #node-layer {
      position: absolute;
      inset: 0;
    }

    #zone-layer {
      z-index: 0;
      pointer-events: none;
    }

    #edge-layer {
      width: 100%;
      height: 100%;
      overflow: visible;
      z-index: 1;
    }

    #character-layer {
      z-index: 2;
      pointer-events: none;
    }

    #node-layer {
      z-index: 3;
    }

    .edge {
      fill: none;
      stroke: rgba(157, 177, 203, 0.28);
      stroke-width: 0.35;
      stroke-linecap: round;
      stroke-dasharray: 1.8 1.2;
      transition: stroke 180ms ease, opacity 180ms ease, stroke-width 180ms ease;
      opacity: 0.7;
    }

    .edge.visited {
      stroke: rgba(244, 193, 82, 0.48);
      opacity: 0.95;
    }

    .edge.active {
      stroke: rgba(47, 213, 196, 0.92);
      stroke-width: 0.55;
      filter: drop-shadow(0 0 10px rgba(47, 213, 196, 0.45));
      animation: pulse-line 1.5s linear infinite;
    }

    .edge-label {
      font-family: "IBM Plex Mono", monospace;
      font-size: 1.2px;
      fill: rgba(236, 244, 255, 0.7);
      pointer-events: none;
    }

    .room-zone {
      position: absolute;
      width: 138px;
      height: 100px;
      transform: translate(-50%, -50%);
      pointer-events: none;
      opacity: 0.94;
      filter: saturate(1.08);
    }

    .room-zone::before {
      content: "";
      position: absolute;
      inset: 40px 12px 8px;
      border-radius: 999px;
      background: radial-gradient(circle at center, color-mix(in srgb, var(--room-accent) 28%, transparent), transparent 72%);
      filter: blur(3px);
      opacity: 0.9;
    }

    .room-rug {
      position: absolute;
      inset: auto 16px 10px;
      height: 26px;
      border-radius: 999px;
      background:
        radial-gradient(circle at 30% 35%, rgba(255, 255, 255, 0.26), transparent 44%),
        linear-gradient(90deg, color-mix(in srgb, var(--room-accent) 72%, #09111d), rgba(255, 255, 255, 0.06));
      opacity: 0.85;
    }

    .room-desk {
      position: absolute;
      left: 50%;
      top: 28px;
      width: 58px;
      height: 16px;
      transform: translateX(-50%);
      border-radius: 12px;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.18), rgba(255, 255, 255, 0.05)),
        color-mix(in srgb, var(--room-accent) 22%, #0b1525);
      border: 1px solid rgba(255, 255, 255, 0.08);
      box-shadow: 0 8px 18px rgba(0, 0, 0, 0.18);
    }

    .room-monitor {
      position: absolute;
      left: 50%;
      top: 10px;
      width: 22px;
      height: 18px;
      transform: translateX(-50%);
      border-radius: 7px 7px 5px 5px;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.24), rgba(255, 255, 255, 0.04)),
        #0d1b2c;
      border: 1px solid rgba(255, 255, 255, 0.08);
      box-shadow: 0 0 0 1px rgba(47, 213, 196, 0.08), 0 0 16px color-mix(in srgb, var(--room-accent) 22%, transparent);
    }

    .room-monitor::after {
      content: "";
      position: absolute;
      left: 50%;
      bottom: -5px;
      width: 10px;
      height: 4px;
      transform: translateX(-50%);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.24);
    }

    .room-label {
      position: absolute;
      left: 50%;
      bottom: -4px;
      transform: translateX(-50%);
      padding: 4px 7px;
      border-radius: 999px;
      font-size: 10px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: rgba(236, 244, 255, 0.78);
      font-family: "IBM Plex Mono", monospace;
      background: rgba(6, 11, 21, 0.72);
      border: 1px solid rgba(255, 255, 255, 0.06);
      white-space: nowrap;
    }

    @keyframes pulse-line {
      from { stroke-dashoffset: 0; }
      to { stroke-dashoffset: -9; }
    }

    .node {
      position: absolute;
      transform: translate(-50%, -50%);
      width: 188px;
      padding: 14px 15px 13px;
      border-radius: 18px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(7, 12, 22, 0.84);
      box-shadow: 0 10px 32px rgba(0, 0, 0, 0.26);
      transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease, opacity 180ms ease;
      cursor: grab;
      user-select: none;
      touch-action: none;
    }

    .node::after {
      content: "";
      position: absolute;
      inset: auto 14px 9px;
      height: 2px;
      border-radius: 999px;
      background: transparent;
    }

    .node.control::after {
      background: linear-gradient(90deg, var(--accent-control), transparent);
    }

    .node.desk::after {
      background: linear-gradient(90deg, var(--accent-desk), transparent);
    }

    .node.research::after {
      background: linear-gradient(90deg, var(--accent-research), transparent);
    }

    .node.idle {
      opacity: 0.72;
    }

    .node.visited {
      opacity: 1;
    }

    .node.active {
      transform: translate(-50%, -50%) scale(1.03);
      border-color: rgba(47, 213, 196, 0.65);
      box-shadow: 0 0 0 1px rgba(47, 213, 196, 0.18), 0 16px 44px rgba(47, 213, 196, 0.18);
    }

    .node.dragging {
      cursor: grabbing;
      z-index: 6;
      transition: none;
      box-shadow: 0 0 0 1px rgba(47, 213, 196, 0.22), 0 18px 52px rgba(47, 213, 196, 0.24);
    }

    .node.done {
      border-color: rgba(244, 193, 82, 0.44);
    }

    .node.error {
      border-color: rgba(255, 94, 107, 0.75);
      box-shadow: 0 0 0 1px rgba(255, 94, 107, 0.18), 0 16px 44px rgba(255, 94, 107, 0.18);
    }

    .node-top {
      display: flex;
      justify-content: space-between;
      align-items: start;
      gap: 10px;
    }

    .node-group {
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      font-family: "IBM Plex Mono", monospace;
    }

    .node-title {
      margin: 4px 0 0;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }

    .node-subtitle {
      margin-top: 2px;
      color: var(--muted);
      font-size: 12px;
    }

    .status-pill {
      min-width: 62px;
      padding: 6px 8px;
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(255, 255, 255, 0.04);
      font-size: 11px;
      text-align: center;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      font-family: "IBM Plex Mono", monospace;
    }

    .status-pill.active {
      color: #07111d;
      background: var(--accent-desk);
    }

    .status-pill.done {
      color: #1a1204;
      background: var(--accent-research);
    }

    .status-pill.error {
      color: #fff5f6;
      background: var(--accent-danger);
    }

    .status-pill.idle {
      color: var(--muted);
    }

    .node-stats {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin: 12px 0 10px;
    }

    .stat-box {
      padding: 8px;
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stat-key {
      font-size: 10px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      font-family: "IBM Plex Mono", monospace;
    }

    .stat-val {
      margin-top: 5px;
      font-size: 15px;
      font-weight: 600;
    }

    .node-summary {
      min-height: 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }

    .node-live-task {
      display: none;
      margin-top: 10px;
      padding-top: 9px;
      border-top: 1px solid rgba(255, 255, 255, 0.06);
    }

    .node-live-task.visible {
      display: block;
    }

    .node-live-kicker {
      font-size: 10px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: rgba(47, 213, 196, 0.88);
      font-family: "IBM Plex Mono", monospace;
    }

    .node-live-body {
      margin-top: 4px;
      color: var(--text);
      font-size: 12px;
      line-height: 1.45;
    }

    .character {
      position: absolute;
      width: 90px;
      height: 102px;
      transform: translate(-50%, -50%);
      transition:
        left 780ms cubic-bezier(0.22, 0.61, 0.36, 1),
        top 780ms cubic-bezier(0.22, 0.61, 0.36, 1),
        filter 180ms ease;
      pointer-events: none;
    }

    .character-shell {
      position: relative;
      width: 100%;
      height: 100%;
      animation: idle-bob 2.8s ease-in-out infinite;
      transform-origin: center bottom;
    }

    .character.walking {
      z-index: 5;
    }

    .character.walking .character-shell {
      animation: walk-bob 420ms ease-in-out infinite alternate;
    }

    .character.active .character-shell {
      animation: active-bob 840ms ease-in-out infinite;
    }

    .character.error .character-shell {
      animation: error-shake 320ms linear infinite;
      filter: drop-shadow(0 0 14px rgba(255, 94, 107, 0.36));
    }

    .character.done .character-shell {
      filter: drop-shadow(0 0 10px rgba(244, 193, 82, 0.24));
    }

    .character-shadow {
      position: absolute;
      left: 50%;
      bottom: 8px;
      width: 40px;
      height: 14px;
      transform: translateX(-50%);
      border-radius: 999px;
      background: rgba(0, 0, 0, 0.24);
      filter: blur(3px);
    }

    .character-body {
      position: absolute;
      left: 50%;
      top: 10px;
      width: 54px;
      height: 72px;
      transform: translateX(-50%);
    }

    .character-head {
      position: absolute;
      left: 50%;
      top: 0;
      width: 34px;
      height: 34px;
      transform: translateX(-50%);
      border-radius: 14px 14px 12px 12px;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.28), rgba(255, 255, 255, 0.08)),
        color-mix(in srgb, var(--char-accent) 64%, #f0e3cf);
      border: 1px solid rgba(255, 255, 255, 0.12);
      display: grid;
      place-items: center;
      font: 700 11px "IBM Plex Mono", monospace;
      color: #07111d;
      box-shadow: 0 6px 16px rgba(0, 0, 0, 0.18);
    }

    .character-head::before {
      content: "";
      position: absolute;
      inset: 4px 6px auto;
      height: 6px;
      border-radius: 999px;
      background: color-mix(in srgb, var(--char-accent) 75%, #08101d);
      opacity: 0.8;
    }

    .character-torso {
      position: absolute;
      left: 50%;
      top: 26px;
      width: 40px;
      height: 28px;
      transform: translateX(-50%);
      border-radius: 12px 12px 10px 10px;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.24), rgba(255, 255, 255, 0.06)),
        color-mix(in srgb, var(--char-accent) 70%, #0f1a2b);
      border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .character-torso::before,
    .character-torso::after {
      content: "";
      position: absolute;
      top: 6px;
      width: 10px;
      height: 18px;
      border-radius: 999px;
      background: color-mix(in srgb, var(--char-accent) 46%, #f0e3cf);
    }

    .character-torso::before {
      left: -7px;
      transform: rotate(22deg);
    }

    .character-torso::after {
      right: -7px;
      transform: rotate(-22deg);
    }

    .character-legs {
      position: absolute;
      left: 50%;
      top: 52px;
      width: 28px;
      height: 20px;
      transform: translateX(-50%);
      display: flex;
      justify-content: space-between;
    }

    .character-legs span {
      display: block;
      width: 9px;
      height: 20px;
      border-radius: 999px;
      background: color-mix(in srgb, var(--char-accent) 42%, #d6ddea);
      transform-origin: top center;
    }

    .character.walking .character-legs span:first-child {
      transform: rotate(15deg);
    }

    .character.walking .character-legs span:last-child {
      transform: rotate(-15deg);
    }

    .character-badge {
      position: absolute;
      left: 50%;
      top: 58px;
      transform: translateX(-50%);
      padding: 3px 6px;
      border-radius: 999px;
      background: rgba(6, 11, 21, 0.84);
      border: 1px solid rgba(255, 255, 255, 0.08);
      color: rgba(236, 244, 255, 0.8);
      font: 10px "IBM Plex Mono", monospace;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .character-callout {
      position: absolute;
      left: 50%;
      top: -8px;
      transform: translateX(-50%);
      max-width: 118px;
      padding: 4px 7px;
      border-radius: 10px;
      background: rgba(6, 11, 21, 0.88);
      border: 1px solid rgba(255, 255, 255, 0.08);
      color: rgba(236, 244, 255, 0.88);
      font-size: 10px;
      line-height: 1.35;
      text-align: center;
      opacity: 0;
      transition: opacity 160ms ease;
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
    }

    .character-callout.visible {
      opacity: 1;
    }

    .character-signal {
      position: absolute;
      left: 50%;
      top: 18px;
      width: 68px;
      height: 68px;
      transform: translateX(-50%);
      border-radius: 999px;
      border: 1px solid color-mix(in srgb, var(--char-accent) 56%, transparent);
      background: radial-gradient(circle at center, color-mix(in srgb, var(--char-accent) 26%, transparent), transparent 66%);
      opacity: 0;
      transition: opacity 180ms ease;
    }

    .character.active .character-signal,
    .character.walking .character-signal,
    .character.error .character-signal {
      opacity: 1;
    }

    @keyframes idle-bob {
      0%, 100% { transform: translateY(0); }
      50% { transform: translateY(-3px); }
    }

    @keyframes active-bob {
      0%, 100% { transform: translateY(0) scale(1); }
      50% { transform: translateY(-5px) scale(1.03); }
    }

    @keyframes walk-bob {
      0% { transform: translateY(0) rotate(-1deg); }
      100% { transform: translateY(-4px) rotate(1deg); }
    }

    @keyframes error-shake {
      0%, 100% { transform: translateX(0); }
      25% { transform: translateX(-2px); }
      75% { transform: translateX(2px); }
    }

    .side-column {
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .side-section {
      padding: 18px;
    }

    .controls {
      display: flex;
      gap: 8px;
      margin: 12px 0;
      flex-wrap: wrap;
    }

    button {
      appearance: none;
      border: 0;
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.06);
      color: var(--text);
      cursor: pointer;
      font: inherit;
    }

    button:hover {
      background: rgba(255, 255, 255, 0.1);
    }

    input[type="range"] {
      width: 100%;
      accent-color: var(--accent-desk);
    }

    .current-event {
      margin-top: 12px;
      padding: 14px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .current-event-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: baseline;
      margin-bottom: 8px;
    }

    .current-event-title {
      font-size: 16px;
      font-weight: 700;
    }

    .execution-node-line {
      margin-top: 3px;
      color: var(--muted);
      font-size: 11px;
      font-family: "IBM Plex Mono", monospace;
    }

    .execution-node-name {
      color: var(--text);
    }

    .current-event-meta {
      color: var(--muted);
      font-size: 12px;
      font-family: "IBM Plex Mono", monospace;
    }

    .current-event-body {
      color: var(--muted);
      line-height: 1.6;
      font-size: 13px;
    }

    .feed,
    .ledger {
      padding: 18px;
    }

    .agent-work-list,
    .feed-list,
    .edge-list,
    .aux-list {
      display: grid;
      gap: 10px;
    }

    .agent-work-item,
    .feed-item,
    .edge-item,
    .aux-item {
      padding: 12px 13px;
      border-radius: 14px;
      border: 1px solid rgba(255, 255, 255, 0.06);
      background: rgba(255, 255, 255, 0.025);
    }

    .agent-work-item.active,
    .feed-item.active {
      border-color: rgba(47, 213, 196, 0.5);
      background: rgba(47, 213, 196, 0.08);
    }

    .agent-work-item.walking {
      border-color: rgba(244, 193, 82, 0.4);
      background: rgba(244, 193, 82, 0.08);
    }

    .agent-work-item.error {
      border-color: rgba(255, 94, 107, 0.44);
      background: rgba(255, 94, 107, 0.08);
    }

    .agent-work-head,
    .feed-head,
    .edge-head,
    .aux-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: baseline;
      margin-bottom: 6px;
    }

    .agent-work-title,
    .feed-title,
    .edge-title,
    .aux-title {
      font-weight: 600;
    }

    .agent-work-meta,
    .feed-meta,
    .edge-meta,
    .aux-meta {
      color: var(--muted);
      font-size: 12px;
      font-family: "IBM Plex Mono", monospace;
    }

    .agent-work-body,
    .feed-body,
    .edge-body,
    .aux-body {
      color: var(--muted);
      line-height: 1.55;
      font-size: 13px;
    }

    @media (max-width: 1180px) {
      .hero,
      .support-grid {
        grid-template-columns: 1fr;
      }

      .arena-stage {
        min-height: 880px;
      }
    }

    @media (max-width: 720px) {
      .shell {
        width: min(100%, calc(100% - 16px));
        margin: 8px auto 18px;
      }

      .hero-main,
      .hero-side,
      .arena,
      .insights-column,
      .side-section,
      .feed,
      .ledger {
        padding: 16px;
      }

      .node {
        width: 158px;
        padding: 11px 12px 10px;
      }

      .room-zone {
        width: 118px;
        height: 88px;
      }

      .character {
        width: 78px;
        height: 92px;
      }

      .node-title {
        font-size: 16px;
      }

      .node-stats {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="panel hero-main">
        <div class="hero-kicker">Agent Empire Replay</div>
        <h1 id="hero-title"></h1>
        <p id="hero-request"></p>
      </div>
      <div class="panel hero-side">
        <div class="metrics" id="metrics"></div>
        <div class="hero-strip" id="task-strip"></div>
      </div>
    </section>

    <section class="layout">
      <div class="panel arena">
        <div class="section-head">
          <h2>Interaction Map</h2>
          <div class="section-head-actions">
            <button id="reset-layout" type="button">Reset Layout</button>
            <div class="section-subtle">드래그로 노드를 옮길 수 있습니다. 활동 상태와 흐름은 이벤트 로그로 리플레이합니다.</div>
          </div>
        </div>
        <div class="arena-stage" id="arena-stage">
          <div id="zone-layer"></div>
          <svg id="edge-layer" viewBox="0 0 100 100" preserveAspectRatio="none"></svg>
          <div id="character-layer"></div>
          <div id="node-layer"></div>
        </div>
        <div class="standby-dock-wrap">
          <div class="standby-head">
            <div class="standby-headline">Standby Units</div>
            <div class="standby-note">이번 run에서 움직이지 않은 에이전트와 system node는 여기로 축약합니다.</div>
          </div>
          <div class="standby-dock" id="standby-dock"></div>
        </div>
      </div>

      <div class="support-grid">
        <div class="side-column">
          <div class="panel side-section">
            <h2>Playback</h2>
            <div class="controls">
              <button id="play-toggle" type="button">Play</button>
              <button id="prev-step" type="button">Prev</button>
              <button id="next-step" type="button">Next</button>
            </div>
            <input id="timeline-scrubber" type="range" min="0" max="0" value="0">
            <div class="current-event" id="current-event"></div>
          </div>

          <div class="panel side-section">
            <div class="section-head">
              <h2>Live Work</h2>
              <div class="section-subtle">실행 중인 에이전트 또는 system node의 현재 작업</div>
            </div>
            <div class="agent-work-list" id="agent-work-list"></div>
          </div>

          <div class="panel side-section">
            <h2>Auxiliary Nodes</h2>
            <div class="aux-list" id="aux-list"></div>
          </div>
        </div>

        <div class="insights-column">
          <div class="panel feed">
            <div class="section-head">
              <h2>Ops Feed</h2>
              <div class="section-subtle" id="feed-count"></div>
            </div>
            <div class="feed-list" id="feed-list"></div>
          </div>

          <div class="panel ledger">
            <div class="section-head">
              <h2>Interaction Ledger</h2>
              <div class="section-subtle">집계된 노드 간 흐름</div>
            </div>
            <div class="edge-list" id="edge-list"></div>
          </div>
        </div>
      </div>
    </section>
  </div>

  <script>
    let model = __MODEL_JSON__;
    const state = {
      index: Math.max((model.timeline || []).length - 1, 0),
      playing: false,
      timer: null,
      liveSource: null,
      drag: null,
    };

    let nodesById = new Map((model.nodes || []).map((node) => [node.id, node]));
    let defaultLayout = {};
    const metricsRoot = document.getElementById("metrics");
    const taskStrip = document.getElementById("task-strip");
    const heroTitle = document.getElementById("hero-title");
    const heroRequest = document.getElementById("hero-request");
    const feedCount = document.getElementById("feed-count");
    const arenaStage = document.getElementById("arena-stage");
    const zoneLayer = document.getElementById("zone-layer");
    const edgeSvg = document.getElementById("edge-layer");
    const characterLayer = document.getElementById("character-layer");
    const nodeLayer = document.getElementById("node-layer");
    const standbyDock = document.getElementById("standby-dock");
    const agentWorkList = document.getElementById("agent-work-list");
    const feedList = document.getElementById("feed-list");
    const edgeList = document.getElementById("edge-list");
    const auxList = document.getElementById("aux-list");
    const currentEvent = document.getElementById("current-event");
    const scrubber = document.getElementById("timeline-scrubber");
    const playToggle = document.getElementById("play-toggle");
    const prevStep = document.getElementById("prev-step");
    const nextStep = document.getElementById("next-step");
    const resetLayoutButton = document.getElementById("reset-layout");
    const WORK_LABELS = {
      question_understanding: "Parsing request",
      orchestrator: "Coordinating desks",
      research_manager: "Managing research loop",
      macro: "Checking macro regime",
      fundamental: "Reviewing company setup",
      sentiment: "Reading narrative flow",
      quant: "Sizing trade",
      hedge_lite_builder: "Building hedge overlay",
      portfolio_construction_quant: "Assembling portfolio",
      risk_manager: "Running risk gates",
      report_writer: "Writing memo",
    };

    function layoutStorageKey() {
      return `agent-empire-layout:${model.run_id || "unknown"}`;
    }

    function mapNodes() {
      return (model.nodes || []).filter((node) => (node.render_mode || "card") === "card");
    }

    function standbyNodes() {
      return (model.nodes || []).filter((node) => node.render_mode === "standby");
    }

    function readSavedLayout() {
      try {
        const raw = window.localStorage.getItem(layoutStorageKey());
        if (!raw) {
          return {};
        }
        const parsed = JSON.parse(raw);
        return parsed && typeof parsed === "object" ? parsed : {};
      } catch (_err) {
        return {};
      }
    }

    function saveLayout() {
      try {
        const payload = {};
        for (const node of mapNodes()) {
          payload[node.id] = { x: node.x, y: node.y };
        }
        window.localStorage.setItem(layoutStorageKey(), JSON.stringify(payload));
      } catch (_err) {
        // ignore storage errors
      }
    }

    function clearSavedLayout() {
      try {
        window.localStorage.removeItem(layoutStorageKey());
      } catch (_err) {
        // ignore storage errors
      }
    }

    function applySavedLayout(targetModel) {
      const saved = readSavedLayout();
      for (const node of targetModel.nodes || []) {
        const override = saved[node.id];
        if (!override) {
          continue;
        }
        const x = clamp(Number(override.x), 8, 92);
        const y = clamp(Number(override.y), 8, 92);
        if (Number.isFinite(x) && Number.isFinite(y)) {
          node.x = x;
          node.y = y;
        }
      }
    }

    function snapshotDefaultLayout(sourceModel) {
      const snapshot = {};
      for (const node of sourceModel.nodes || []) {
        snapshot[node.id] = { x: node.x, y: node.y };
      }
      return snapshot;
    }

    function createMetric(metric) {
      const item = document.createElement("div");
      item.className = "metric";
      item.innerHTML = `
        <div class="metric-label">${metric.label}</div>
        <div class="metric-value">${metric.value}</div>
      `;
      return item;
    }

    function resetStaticLayout() {
      metricsRoot.innerHTML = "";
      taskStrip.innerHTML = "";
      zoneLayer.innerHTML = "";
      edgeSvg.innerHTML = "";
      characterLayer.innerHTML = "";
      nodeLayer.innerHTML = "";
      standbyDock.innerHTML = "";
      agentWorkList.innerHTML = "";
      edgeList.innerHTML = "";
      auxList.innerHTML = "";
    }

    function cleanTaskDetail(detail) {
      const text = String(detail || "").trim();
      if (!text || text === "no summary") {
        return "";
      }
      return text;
    }

    function executionNodeForEvent(event) {
      if (!event) {
        return "";
      }
      const rawNode = String(event.node_name || "").trim();
      const displayNode = String(event.node || "").trim();
      const agentId = String(event.agent_id || "").trim();
      if (!rawNode || !displayNode || !agentId || rawNode === displayNode) {
        return "";
      }
      return rawNode;
    }

    function taskLabelForNode(nodeId, fallbackLabel) {
      return WORK_LABELS[nodeId] || fallbackLabel || "Processing";
    }

    function taskTextForEvent(event, overrideLabel = "") {
      if (!event) {
        return "";
      }
      const base = overrideLabel || taskLabelForNode(event.node, event.label);
      const detail = cleanTaskDetail(event.detail);
      const executionNode = executionNodeForEvent(event);
      const parts = [];
      if (executionNode) {
        parts.push(`node=${executionNode}`);
      }
      if (detail) {
        parts.push(detail);
      }
      if (event.phase === "exit") {
        return parts.length ? `${base} complete · ${parts.join(" · ")}` : `${base} complete`;
      }
      return parts.length ? `${base} · ${parts.join(" · ")}` : base;
    }

    function createNode(node) {
      const summary = node.last_summary
        ? `<div class="node-summary">${node.last_summary}</div>`
        : "";
      const el = document.createElement("article");
      el.className = `node ${node.group} ${node.visited ? "visited" : "idle"}`;
      el.dataset.nodeId = node.id;
      el.style.left = `${node.x}%`;
      el.style.top = `${node.y}%`;
      el.innerHTML = `
        <div class="node-top">
          <div>
            <div class="node-group">${node.group_label}</div>
            <div class="node-title">${node.label}</div>
            <div class="node-subtitle">${node.subtitle}</div>
          </div>
          <div class="status-pill idle">idle</div>
        </div>
        <div class="node-stats">
          <div class="stat-box">
            <div class="stat-key">enter</div>
            <div class="stat-val">${node.enter_count}</div>
          </div>
          <div class="stat-box">
            <div class="stat-key">exit</div>
            <div class="stat-val">${node.exit_count}</div>
          </div>
          <div class="stat-box">
            <div class="stat-key">avg</div>
            <div class="stat-val">${node.avg_duration_label}</div>
          </div>
        </div>
        ${summary}
        <div class="node-live-task">
          <div class="node-live-kicker">current work</div>
          <div class="node-live-body"></div>
        </div>
      `;
      return el;
    }

    function createStandby(node) {
      const el = document.createElement("div");
      el.className = "standby-chip";
      el.style.setProperty("--standby-accent", node.accent || "#2fd5c4");
      el.innerHTML = `
        <div class="standby-avatar">${node.avatar || node.label.slice(0, 2)}</div>
        <div class="standby-text">
          <div class="standby-label">${node.label}</div>
          <div class="standby-meta">${node.group_label} · ${node.room_label || node.subtitle}</div>
        </div>
      `;
      return el;
    }

    function createZone(node) {
      const el = document.createElement("div");
      el.className = `room-zone ${node.group}`;
      el.dataset.zoneId = node.id;
      el.style.left = `${node.x}%`;
      el.style.top = `${Math.min(95, node.y + node.home_dy - 2)}%`;
      el.style.setProperty("--room-accent", node.accent || "#2fd5c4");
      el.innerHTML = `
        <div class="room-monitor"></div>
        <div class="room-desk"></div>
        <div class="room-rug"></div>
        <div class="room-label">${node.room_label || node.label}</div>
      `;
      return el;
    }

    function createCharacter(node) {
      const el = document.createElement("div");
      el.className = "character idle";
      el.dataset.charId = node.id;
      el.style.setProperty("--char-accent", node.accent || "#2fd5c4");
      el.innerHTML = `
        <div class="character-callout"></div>
        <div class="character-shell">
          <div class="character-signal"></div>
          <div class="character-shadow"></div>
          <div class="character-body">
            <div class="character-head"><span>${node.avatar || node.label.slice(0, 2)}</span></div>
            <div class="character-torso"></div>
            <div class="character-legs"><span></span><span></span></div>
            <div class="character-badge">${node.avatar || node.label.slice(0, 2)}</div>
          </div>
        </div>
      `;
      return el;
    }

    function createEdge(edge) {
      const source = nodesById.get(edge.source);
      const target = nodesById.get(edge.target);
      if (!source || !target) {
        return null;
      }
      const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
      group.dataset.edgeId = edge.id;

      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", source.x);
      line.setAttribute("y1", source.y);
      line.setAttribute("x2", target.x);
      line.setAttribute("y2", target.y);
      line.setAttribute("class", edge.count > 0 ? "edge visited" : "edge");
      group.appendChild(line);

      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("class", "edge-label");
      label.setAttribute("x", ((source.x + target.x) / 2).toFixed(2));
      label.setAttribute("y", ((source.y + target.y) / 2 - 1.2).toFixed(2));
      label.textContent = edge.label;
      group.appendChild(label);
      return group;
    }

    function pointerToArenaPercent(event) {
      const rect = arenaStage.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width) * 100;
      const y = ((event.clientY - rect.top) / rect.height) * 100;
      return {
        x: clamp(x, 8, 92),
        y: clamp(y, 8, 92),
      };
    }

    function setNodeElementPosition(nodeId) {
      const node = nodesById.get(nodeId);
      if (!node) {
        return;
      }
      const nodeEl = nodeLayer.querySelector(`[data-node-id="${nodeId}"]`);
      if (nodeEl) {
        nodeEl.style.left = `${node.x}%`;
        nodeEl.style.top = `${node.y}%`;
      }
      const zoneEl = zoneLayer.querySelector(`[data-zone-id="${nodeId}"]`);
      if (zoneEl) {
        zoneEl.style.left = `${node.x}%`;
        zoneEl.style.top = `${Math.min(95, node.y + node.home_dy - 2)}%`;
      }
    }

    function syncEdgeGeometry(activeEdge = "") {
      for (const edgeEl of edgeSvg.querySelectorAll("[data-edge-id]")) {
        const edgeId = edgeEl.dataset.edgeId;
        const edge = (model.edges || []).find((item) => item.id === edgeId);
        const line = edgeEl.querySelector(".edge");
        const label = edgeEl.querySelector(".edge-label");
        if (!edge || !line || !label) {
          continue;
        }
        const source = nodesById.get(edge.source);
        const target = nodesById.get(edge.target);
        if (!source || !target) {
          continue;
        }
        line.setAttribute("x1", source.x);
        line.setAttribute("y1", source.y);
        line.setAttribute("x2", target.x);
        line.setAttribute("y2", target.y);
        line.classList.remove("active");
        if (edgeId === activeEdge) {
          line.classList.add("active");
        }
        label.setAttribute("x", ((source.x + target.x) / 2).toFixed(2));
        label.setAttribute("y", ((source.y + target.y) / 2 - 1.2).toFixed(2));
      }
    }

    function syncCharacterGeometry(playback) {
      const charState = characterState(state.index, playback);
      for (const charEl of characterLayer.querySelectorAll(".character")) {
        const charId = charEl.dataset.charId;
        const nextState = charState.statuses[charId] || "idle";
        const nextPos = charState.positions[charId] || { x: 50, y: 50 };
        charEl.classList.remove("idle", "active", "done", "error", "walking");
        charEl.classList.add(nextState);
        charEl.style.left = `${nextPos.x}%`;
        charEl.style.top = `${nextPos.y}%`;
        const bubble = charEl.querySelector(".character-callout");
        if (bubble) {
          const text = charState.callouts[charId] || "";
          bubble.textContent = text;
          bubble.classList.toggle("visible", Boolean(text) && nextState !== "idle");
        }
      }
    }

    function updateNodePosition(nodeId, nextX, nextY) {
      const node = nodesById.get(nodeId);
      if (!node) {
        return;
      }
      node.x = clamp(nextX, 8, 92);
      node.y = clamp(nextY, 8, 92);
      setNodeElementPosition(nodeId);
      const playback = playbackState(state.index);
      syncEdgeGeometry(playback.activeEdge);
      syncCharacterGeometry(playback);
    }

    function attachNodeDrag(nodeEl) {
      nodeEl.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) {
          return;
        }
        const nodeId = nodeEl.dataset.nodeId;
        const node = nodesById.get(nodeId);
        if (!node) {
          return;
        }
        stopPlayback();
        const point = pointerToArenaPercent(event);
        state.drag = {
          nodeId,
          pointerId: event.pointerId,
          offsetX: point.x - node.x,
          offsetY: point.y - node.y,
        };
        nodeEl.classList.add("dragging");
        nodeEl.setPointerCapture(event.pointerId);
        event.preventDefault();
      });

      nodeEl.addEventListener("pointermove", (event) => {
        if (!state.drag || state.drag.pointerId !== event.pointerId) {
          return;
        }
        const point = pointerToArenaPercent(event);
        updateNodePosition(
          state.drag.nodeId,
          point.x - state.drag.offsetX,
          point.y - state.drag.offsetY,
        );
      });

      function finishDrag(event) {
        if (!state.drag || state.drag.pointerId !== event.pointerId) {
          return;
        }
        const dragNode = nodeLayer.querySelector(`[data-node-id="${state.drag.nodeId}"]`);
        if (dragNode) {
          dragNode.classList.remove("dragging");
        }
        saveLayout();
        state.drag = null;
      }

      nodeEl.addEventListener("pointerup", finishDrag);
      nodeEl.addEventListener("pointercancel", finishDrag);
    }

    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

    function nodeSeed(nodeId) {
      let out = 0;
      for (const char of String(nodeId || "")) {
        out += char.charCodeAt(0);
      }
      return out || 1;
    }

    function homePosition(node, tick = 0) {
      const seed = nodeSeed(node.id);
      const roamX = Math.sin(tick * 0.52 + seed * 0.11) * 0.9;
      const roamY = Math.cos(tick * 0.43 + seed * 0.07) * 0.55;
      return {
        x: clamp(node.x + (node.home_dx || 0) + roamX, 6, 94),
        y: clamp(node.y + (node.home_dy || 0) + roamY, 8, 95),
      };
    }

    function focusPosition(node, tick = 0) {
      const seed = nodeSeed(node.id);
      return {
        x: clamp(node.x + Math.sin(tick * 0.35 + seed * 0.03) * 0.55, 6, 94),
        y: clamp(node.y + (node.home_dy || 0) - 4.2, 8, 95),
      };
    }

    function lanePoint(sourceNode, targetNode, t, lane = 0) {
      const sx = sourceNode.x + (sourceNode.home_dx || 0);
      const sy = sourceNode.y + (sourceNode.home_dy || 0);
      const tx = targetNode.x + (targetNode.home_dx || 0);
      const ty = targetNode.y + (targetNode.home_dy || 0);
      const dx = tx - sx;
      const dy = ty - sy;
      const norm = Math.hypot(dx, dy) || 1;
      return {
        x: clamp(sx + dx * t + (-dy / norm) * lane, 6, 94),
        y: clamp(sy + dy * t + (dx / norm) * lane, 8, 95),
      };
    }

    function characterState(index, playback) {
      const positions = {};
      const statuses = {};
      const callouts = {};

      for (const node of mapNodes()) {
        positions[node.id] = homePosition(node, index);
        statuses[node.id] = playback.nodes[node.id] || "idle";
        callouts[node.id] = ["active", "walking", "error"].includes(statuses[node.id])
          ? (playback.tasks[node.id] || "")
          : "";
      }

      const activeEvent = playback.activeEvent;
      if (!activeEvent) {
        return { positions, statuses, callouts };
      }

      const eventNode = nodesById.get(activeEvent.node);
      if (eventNode) {
        positions[eventNode.id] = focusPosition(eventNode, index);
        statuses[eventNode.id] = activeEvent.errors && activeEvent.errors.length > 0
          ? "error"
          : activeEvent.phase === "enter"
            ? "active"
            : "done";
        callouts[eventNode.id] = playback.tasks[eventNode.id] || `${activeEvent.phase} · ${eventNode.label}`;
      }

      if (activeEvent.flow && activeEvent.flow.source && activeEvent.flow.target) {
        const sourceNode = nodesById.get(activeEvent.flow.source);
        const targetNode = nodesById.get(activeEvent.flow.target);
        if (sourceNode && targetNode) {
          positions[sourceNode.id] = lanePoint(sourceNode, targetNode, 0.38, -1.8);
          statuses[sourceNode.id] = activeEvent.errors && activeEvent.errors.length > 0 ? "error" : "walking";
          callouts[sourceNode.id] = playback.tasks[sourceNode.id] || `route to ${targetNode.label}`;

          if (sourceNode.id !== targetNode.id) {
            positions[targetNode.id] = activeEvent.phase === "enter"
              ? lanePoint(sourceNode, targetNode, 0.82, 1.8)
              : focusPosition(targetNode, index);
            statuses[targetNode.id] = activeEvent.errors && activeEvent.errors.length > 0
              ? "error"
              : activeEvent.phase === "enter"
                ? "active"
                : "done";
            callouts[targetNode.id] = playback.tasks[targetNode.id] || (
              activeEvent.phase === "enter"
                ? `receiving ${sourceNode.label}`
                : `${targetNode.label} completed`
            );
          }
        }
      }

      return { positions, statuses, callouts };
    }

    function playbackState(index) {
      const nodes = {};
      const tasks = {};
      const taskMeta = {};
      for (const node of mapNodes()) {
        nodes[node.id] = node.visited ? "done" : "idle";
        tasks[node.id] = "";
        taskMeta[node.id] = null;
      }
      let activeEdge = "";
      let activeEvent = model.timeline[index] || null;
      for (let i = 0; i <= index; i += 1) {
        const event = model.timeline[i];
        if (!event) continue;
        if (!nodes[event.node]) continue;
        tasks[event.node] = taskTextForEvent(event);
        taskMeta[event.node] = {
          iteration: event.iteration,
          phase: event.phase,
          ts: event.ts,
        };
        if (event.errors && event.errors.length > 0) {
          nodes[event.node] = "error";
        } else if (event.phase === "enter") {
          nodes[event.node] = "active";
        } else if (event.phase === "exit") {
          nodes[event.node] = "done";
        }
      }
      if (activeEvent && activeEvent.flow && activeEvent.flow.source && activeEvent.flow.target) {
        activeEdge = `${activeEvent.flow.source}->${activeEvent.flow.target}`;
        const sourceNode = nodesById.get(activeEvent.flow.source);
        const targetNode = nodesById.get(activeEvent.flow.target);
        if (sourceNode && targetNode) {
          tasks[sourceNode.id] = `handoff to ${targetNode.label}`;
          taskMeta[sourceNode.id] = {
            iteration: activeEvent.iteration,
            phase: "route",
            ts: activeEvent.ts,
          };
          if (sourceNode.id !== targetNode.id && activeEvent.phase === "enter") {
            tasks[targetNode.id] = `receiving ${sourceNode.label}`;
            taskMeta[targetNode.id] = {
              iteration: activeEvent.iteration,
              phase: "receive",
              ts: activeEvent.ts,
            };
          }
        }
      }
      if (activeEvent && nodes[activeEvent.node] !== "error") {
        nodes[activeEvent.node] = activeEvent.phase === "enter" ? "active" : "done";
        tasks[activeEvent.node] = taskTextForEvent(activeEvent);
        taskMeta[activeEvent.node] = {
          iteration: activeEvent.iteration,
          phase: activeEvent.phase,
          ts: activeEvent.ts,
        };
      }
      return { nodes, activeEvent, activeEdge, tasks, taskMeta };
    }

    function renderStatic() {
      heroTitle.textContent = `${model.title} · ${model.run_id}`;
      heroRequest.textContent = model.request || "n/a";
      feedCount.textContent = `${(model.timeline || []).length} events`;

      for (const metric of model.metrics || []) {
        metricsRoot.appendChild(createMetric(metric));
      }

      for (const task of model.analysis_tasks || []) {
        const chip = document.createElement("div");
        chip.className = "hero-chip";
        chip.textContent = task;
        taskStrip.appendChild(chip);
      }

      for (const node of mapNodes()) {
        zoneLayer.appendChild(createZone(node));
        characterLayer.appendChild(createCharacter(node));
        const nodeEl = createNode(node);
        attachNodeDrag(nodeEl);
        nodeLayer.appendChild(nodeEl);
      }

      for (const node of standbyNodes()) {
        standbyDock.appendChild(createStandby(node));
      }

      if (standbyDock.childElementCount === 0) {
        const row = document.createElement("div");
        row.className = "standby-chip";
        row.innerHTML = `
          <div class="standby-text">
            <div class="standby-label">No standby agents</div>
            <div class="standby-meta">이번 run에서는 주요 에이전트가 모두 맵에 참여했습니다.</div>
          </div>
        `;
        standbyDock.appendChild(row);
      }

      for (const edge of model.edges || []) {
        const el = createEdge(edge);
        if (el) edgeSvg.appendChild(el);

        const row = document.createElement("div");
        row.className = "edge-item";
        row.innerHTML = `
          <div class="edge-head">
            <div class="edge-title">${nodesById.get(edge.source)?.label || edge.source} → ${nodesById.get(edge.target)?.label || edge.target}</div>
            <div class="edge-meta">${edge.count} event(s)</div>
          </div>
          <div class="edge-body">집계 라우팅: ${edge.label || "0x"}</div>
        `;
        edgeList.appendChild(row);
      }

      if ((model.edges || []).length === 0) {
        const row = document.createElement("div");
        row.className = "edge-item";
        row.innerHTML = `<div class="edge-body">No interaction edges were inferred from this run.</div>`;
        edgeList.appendChild(row);
      }

      for (const aux of model.aux_nodes || []) {
        const row = document.createElement("div");
        row.className = "aux-item";
        row.innerHTML = `
          <div class="aux-head">
            <div class="aux-title">${aux.name}</div>
            <div class="aux-meta">${aux.events} event(s)</div>
          </div>
        `;
        auxList.appendChild(row);
      }

      if ((model.aux_nodes || []).length === 0) {
        const row = document.createElement("div");
        row.className = "aux-item";
        row.innerHTML = `<div class="aux-body">No auxiliary-only nodes in this run.</div>`;
        auxList.appendChild(row);
      }

      scrubber.max = Math.max((model.timeline || []).length - 1, 0);
      scrubber.value = state.index;
    }

    function applyModel(nextModel) {
      const previousTimelineLength = (model.timeline || []).length;
      const nextTimelineLength = (nextModel.timeline || []).length;
      const wasAtEnd = state.index >= Math.max(previousTimelineLength - 1, 0);
      const shouldFollowTail = wasAtEnd || state.playing;
      defaultLayout = snapshotDefaultLayout(nextModel);
      applySavedLayout(nextModel);
      model = nextModel;
      nodesById = new Map((model.nodes || []).map((node) => [node.id, node]));
      if (shouldFollowTail) {
        state.index = Math.max(nextTimelineLength - 1, 0);
      } else {
        state.index = Math.min(state.index, Math.max(nextTimelineLength - 1, 0));
      }
      resetStaticLayout();
      renderStatic();
      renderTimeline();
    }

    function connectLiveStream() {
      const canStream =
        Boolean(window.EventSource) &&
        (window.location.protocol === "http:" || window.location.protocol === "https:");
      if (!canStream || !model.run_id) {
        return;
      }
      const source = new EventSource(`/api/live/runs/${encodeURIComponent(model.run_id)}`);
      source.addEventListener("snapshot", (event) => {
        try {
          const payload = JSON.parse(event.data);
          if (payload && payload.model) {
            applyModel(payload.model);
          }
        } catch (_err) {
          // ignore malformed snapshot frames
        }
      });
      source.onerror = () => {};
      state.liveSource = source;
      window.addEventListener("beforeunload", () => {
        if (state.liveSource) {
          state.liveSource.close();
          state.liveSource = null;
        }
      }, { once: true });
    }

    function renderTimeline() {
      const playback = playbackState(state.index);
      const { nodes, activeEvent, activeEdge, tasks, taskMeta } = playback;

      for (const nodeEl of nodeLayer.querySelectorAll(".node")) {
        const nodeId = nodeEl.dataset.nodeId;
        const nextState = nodes[nodeId] || "idle";
        nodeEl.classList.remove("idle", "active", "done", "error");
        nodeEl.classList.add(nextState);
        const pill = nodeEl.querySelector(".status-pill");
        if (pill) {
          pill.className = `status-pill ${nextState}`;
          pill.textContent = nextState;
        }
        const taskBox = nodeEl.querySelector(".node-live-task");
        const taskBody = nodeEl.querySelector(".node-live-body");
        const taskText = ["active", "walking", "error"].includes(nextState) ? (tasks[nodeId] || "") : "";
        if (taskBody) {
          taskBody.textContent = taskText;
        }
        if (taskBox) {
          taskBox.classList.toggle("visible", Boolean(taskText));
        }
      }

      syncEdgeGeometry(activeEdge);
      syncCharacterGeometry(playback);

      agentWorkList.innerHTML = "";
      for (const node of mapNodes()) {
        const status = nodes[node.id] || "idle";
        if (!["active", "walking", "error"].includes(status)) {
          continue;
        }
        const meta = taskMeta[node.id];
        const row = document.createElement("div");
        row.className = `agent-work-item ${status}`;
        row.innerHTML = `
          <div class="agent-work-head">
            <div class="agent-work-title">${node.label}</div>
            <div class="agent-work-meta">${status} · iter ${meta?.iteration ?? 0}</div>
          </div>
          <div class="agent-work-body">${tasks[node.id] || taskLabelForNode(node.id, node.label)}<br>${meta?.ts || ""}</div>
        `;
        agentWorkList.appendChild(row);
      }

      if (agentWorkList.childElementCount === 0) {
        const row = document.createElement("div");
        row.className = "agent-work-item";
        row.innerHTML = `
          <div class="agent-work-body">No mapped agents or system nodes are currently active. Work detail appears here as new live events arrive.</div>
        `;
        agentWorkList.appendChild(row);
      }

      currentEvent.innerHTML = "";
      if (activeEvent) {
        const executionNode = executionNodeForEvent(activeEvent);
        const box = document.createElement("div");
        box.innerHTML = `
          <div class="current-event-head">
            <div>
              <div class="current-event-title">${activeEvent.label}</div>
              ${executionNode ? `<div class="execution-node-line">execution node · <span class="execution-node-name">${executionNode}</span></div>` : ""}
            </div>
            <div class="current-event-meta">${activeEvent.phase} · iter ${activeEvent.iteration}</div>
          </div>
          <div class="current-event-body">
            <div>${activeEvent.detail}</div>
            <div style="margin-top:8px;">${activeEvent.ts}</div>
          </div>
        `;
        currentEvent.appendChild(box);
      }

      feedList.innerHTML = "";
      const start = Math.max(state.index - 9, 0);
      const end = Math.min(state.index + 8, (model.timeline || []).length - 1);
      for (let i = start; i <= end; i += 1) {
        const item = model.timeline[i];
        if (!item) continue;
        const executionNode = executionNodeForEvent(item);
        const row = document.createElement("div");
        row.className = `feed-item ${i === state.index ? "active" : ""}`;
        row.innerHTML = `
          <div class="feed-head">
            <div class="feed-title">${item.label} · ${item.phase}</div>
            <div class="feed-meta">#${i + 1} · iter ${item.iteration}${executionNode ? ` · ${executionNode}` : ""}</div>
          </div>
          <div class="feed-body">${item.detail}<br>${item.ts}</div>
        `;
        row.addEventListener("click", () => {
          state.index = i;
          scrubber.value = i;
          stopPlayback();
          renderTimeline();
        });
        feedList.appendChild(row);
      }
    }

    function step(delta) {
      const limit = Math.max((model.timeline || []).length - 1, 0);
      state.index = Math.max(0, Math.min(limit, state.index + delta));
      scrubber.value = state.index;
      renderTimeline();
    }

    function stopPlayback() {
      state.playing = false;
      playToggle.textContent = "Play";
      if (state.timer) {
        clearInterval(state.timer);
        state.timer = null;
      }
    }

    function startPlayback() {
      if ((model.timeline || []).length <= 1) {
        return;
      }
      state.playing = true;
      playToggle.textContent = "Pause";
      state.timer = setInterval(() => {
        if (state.index >= (model.timeline || []).length - 1) {
          stopPlayback();
          return;
        }
        step(1);
      }, 950);
    }

    playToggle.addEventListener("click", () => {
      if (state.playing) {
        stopPlayback();
      } else {
        startPlayback();
      }
    });

    prevStep.addEventListener("click", () => {
      stopPlayback();
      step(-1);
    });

    nextStep.addEventListener("click", () => {
      stopPlayback();
      step(1);
    });

    scrubber.addEventListener("input", (event) => {
      stopPlayback();
      state.index = Number(event.target.value || 0);
      renderTimeline();
    });

    resetLayoutButton.addEventListener("click", () => {
      clearSavedLayout();
      for (const node of model.nodes || []) {
        const fallback = defaultLayout[node.id];
        if (!fallback) {
          continue;
        }
        node.x = fallback.x;
        node.y = fallback.y;
      }
      nodesById = new Map((model.nodes || []).map((node) => [node.id, node]));
      resetStaticLayout();
      renderStatic();
      renderTimeline();
    });

    defaultLayout = snapshotDefaultLayout(model);
    applySavedLayout(model);
    nodesById = new Map((model.nodes || []).map((node) => [node.id, node]));
    renderStatic();
    renderTimeline();
    connectLiveStream();
  </script>
</body>
</html>
"""
    return template.replace("__MODEL_JSON__", _json_for_script(model))


def build_dashboard_model(run_id: str, *, runs_dir: Path = RUNS_DIR) -> dict[str, Any]:
    return _build_dashboard_model(run_id, runs_dir=runs_dir)


def render_run_dashboard_html(run_id: str, *, runs_dir: Path = RUNS_DIR) -> str:
    model = _build_dashboard_model(run_id, runs_dir=runs_dir)
    return _render_html(model)


def write_run_dashboard(
    run_id: str,
    *,
    runs_dir: Path = RUNS_DIR,
    output_name: str = "agent_empire.html",
) -> Path:
    model = _build_dashboard_model(run_id, runs_dir=runs_dir)
    output_path = runs_dir / run_id / output_name
    output_path.write_text(_render_html(model), encoding="utf-8")
    return output_path


def _resolve_run_id(raw_run_id: str, runs_dir: Path) -> str:
    candidate = str(raw_run_id or "").strip()
    if candidate and candidate.lower() != "latest":
        return candidate
    latest_run = None
    latest_mtime = -1.0
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        events_path = run_dir / "events.jsonl"
        if not events_path.exists():
            continue
        mtime = events_path.stat().st_mtime
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_run = run_dir.name
    if latest_run is None:
        raise FileNotFoundError(f"no run with events.jsonl found under {runs_dir}")
    return latest_run


def resolve_run_id(raw_run_id: str, *, runs_dir: Path = RUNS_DIR) -> str:
    return _resolve_run_id(raw_run_id, runs_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render Agent Empire replay HTML for a run.")
    parser.add_argument("--run-id", default="latest", help="Run ID to render. Use 'latest' for the newest run.")
    parser.add_argument("--runs-dir", default="runs", help="Runs directory path.")
    parser.add_argument("--output", default="agent_empire.html", help="Output file name inside the run directory.")
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    run_id = _resolve_run_id(args.run_id, runs_dir)
    path = write_run_dashboard(run_id, runs_dir=runs_dir, output_name=args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
