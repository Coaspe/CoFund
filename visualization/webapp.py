"""
visualization/webapp.py
=======================
FastAPI app for browsing and replaying agent runs.
"""

from __future__ import annotations

import asyncio
import argparse
import json
import os
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

from .agent_empire import (
    build_dashboard_model,
    list_runs,
    render_run_dashboard_html,
    resolve_run_id,
)

_RUNS_DIR_ENV = "AGENT_EMPIRE_RUNS_DIR"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LAUNCH_LOG_DIR = _PROJECT_ROOT / ".cache" / "agent_empire_launches"
_RATE_LIMIT_MARKERS = (
    "429",
    "rate limit",
    "too many requests",
    "tokens per minute",
    "requests per minute",
)
_RUN_ID_RE = re.compile(r"Run ID:\s*([0-9a-fA-F-]{36})")
_LLM_RATE_LIMIT_RE = re.compile(r"\[(?P<router>LLM Router)\]\s*(?P<scope>[^:]+):.*\((?P<source>[^)]+)\)")
_API_RATE_LIMIT_RE = re.compile(r"\[(?P<router>API Router)\]\s*(?P<source>[^:]+):")


class LaunchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    mode: str = Field(default="mock")
    seed: int = Field(default=42)
    portfolio_context: dict[str, Any] | None = None


class LaunchPrepareRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    mode: str = Field(default="mock")
    seed: int = Field(default=42)
    portfolio_context: dict[str, Any] | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_for_script(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False).replace("</script>", "<\\/script>")


def _sse_payload(payload: dict, *, event: str = "snapshot") -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _process_exists(pid: Any) -> bool:
    try:
        numeric = int(pid)
    except (TypeError, ValueError):
        return False
    if numeric <= 0:
        return False
    try:
        os.kill(numeric, 0)
        return True
    except OSError:
        return False


def _signal_process(pid: Any, sig: signal.Signals) -> bool:
    try:
        numeric = int(pid)
    except (TypeError, ValueError):
        return False
    if numeric <= 0:
        return False
    for sender in (lambda: os.killpg(numeric, sig), lambda: os.kill(numeric, sig)):
        try:
            sender()
            return True
        except OSError:
            continue
    return False


def _read_launch_log_delta(record: dict[str, Any]) -> str:
    path_text = str(record.get("log_path") or "").strip()
    if not path_text:
        return ""
    path = Path(path_text)
    if not path.exists():
        return ""
    offset = int(record.get("log_offset") or 0)
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            handle.seek(offset)
            chunk = handle.read()
            record["log_offset"] = handle.tell()
            return chunk
    except OSError:
        return ""


def _extract_run_id(text: str) -> str:
    match = _RUN_ID_RE.search(str(text or ""))
    return str(match.group(1)).strip() if match else ""


def _extract_rate_limit_reason(text: str) -> str:
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if any(marker in lower for marker in _RATE_LIMIT_MARKERS):
            return line
    return ""


def _extract_rate_limit_metadata(text: str) -> dict[str, str]:
    reason = ""
    api_type = ""
    api_source = ""
    api_scope = ""
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if not any(marker in lower for marker in _RATE_LIMIT_MARKERS):
            continue
        if not reason:
            reason = line
        llm_match = _LLM_RATE_LIMIT_RE.search(line)
        if llm_match:
            api_type = "llm"
            api_source = str(llm_match.group("source") or "").strip()
            api_scope = str(llm_match.group("scope") or "").strip()
            break
        api_match = _API_RATE_LIMIT_RE.search(line)
        if api_match:
            api_type = "data"
            api_source = str(api_match.group("source") or "").strip()
            break
    return {
        "pending_reason": reason,
        "rate_limit_api_type": api_type,
        "rate_limit_api_source": api_source,
        "rate_limit_api_scope": api_scope,
    }


def _run_is_complete(run_id: str, runs_dir: Path) -> bool:
    run = str(run_id or "").strip()
    if not run:
        return False
    return bool(_read_json(runs_dir / run / "final_state.json"))


def _launch_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "launch_id": str(record.get("launch_id", "")),
        "run_id": str(record.get("run_id", "")),
        "status": str(record.get("status", "accepted")),
        "mode": str(record.get("mode", "")),
        "seed": int(record.get("seed", 0) or 0),
        "question": str(record.get("question", "")),
        "question_preview": str(record.get("question_preview", "")),
        "pid": record.get("pid"),
        "log_path": str(record.get("log_path", "")),
        "command": list(record.get("command", []) or []),
        "pending_reason": str(record.get("pending_reason", "")),
        "rate_limit_api_type": str(record.get("rate_limit_api_type", "")),
        "rate_limit_api_source": str(record.get("rate_limit_api_source", "")),
        "rate_limit_api_scope": str(record.get("rate_limit_api_scope", "")),
        "resume_strategy": str(record.get("resume_strategy", "")),
        "created_at": str(record.get("created_at", "")),
        "updated_at": str(record.get("updated_at", "")),
        "continue_count": int(record.get("continue_count", 0) or 0),
    }


def _refresh_launch_record(record: dict[str, Any], *, runs_dir: Path) -> None:
    chunk = _read_launch_log_delta(record)
    if chunk:
        run_id = _extract_run_id(chunk)
        if run_id and not str(record.get("run_id", "")).strip():
            record["run_id"] = run_id
            record["updated_at"] = _now_iso()
        if str(record.get("status", "")).strip() not in {"pending_rate_limit", "complete", "stopped"}:
            rate_limit = _extract_rate_limit_metadata(chunk)
            reason = str(rate_limit.get("pending_reason", "")).strip()
            if reason:
                record["resume_strategy"] = "signal" if _signal_process(record.get("pid"), signal.SIGSTOP) else "relaunch"
                record.update(rate_limit)
                record["status"] = "pending_rate_limit"
                record["updated_at"] = _now_iso()
                return
    if _run_is_complete(str(record.get("run_id", "")).strip(), runs_dir):
        if str(record.get("status", "")).strip() != "complete":
            record["status"] = "complete"
            record["pending_reason"] = ""
            record["rate_limit_api_type"] = ""
            record["rate_limit_api_source"] = ""
            record["rate_limit_api_scope"] = ""
            record["resume_strategy"] = ""
            record["updated_at"] = _now_iso()
        return
    if str(record.get("status", "")).strip() == "pending_rate_limit":
        return
    alive = _process_exists(record.get("pid"))
    next_status = "running" if alive else "stopped"
    if str(record.get("status", "")).strip() == "accepted" and alive:
        next_status = "running"
    if str(record.get("status", "")).strip() in {"accepted", "running", "resuming"} and next_status != record.get("status"):
        record["status"] = next_status
        record["updated_at"] = _now_iso()


def _overlay_runs_with_launches(runs: list[dict[str, Any]], launches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_run: dict[str, dict[str, Any]] = {}
    for launch in launches:
        run_id = str(launch.get("run_id", "")).strip()
        if not run_id:
            continue
        current = by_run.get(run_id)
        if current is None or str(launch.get("updated_at", "")) > str(current.get("updated_at", "")):
            by_run[run_id] = launch
    merged_runs: list[dict[str, Any]] = []
    for run in runs:
        item = dict(run)
        launch = by_run.get(str(run.get("run_id", "")).strip())
        if launch:
            item["launch_id"] = launch.get("launch_id")
            item["launch_status"] = launch.get("status")
            item["launched_from_web"] = True
            if launch.get("status") == "pending_rate_limit":
                item["status"] = "pending"
                item["is_running"] = False
                item["pending_reason"] = launch.get("pending_reason", "")
            elif launch.get("status") in {"accepted", "running", "resuming"}:
                item["status"] = "running"
                item["is_running"] = True
        merged_runs.append(item)
    return merged_runs


def _load_run_result(run_id: str, runs_dir: Path) -> dict[str, Any]:
    run_dir = runs_dir / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory not found: {run_dir}")

    final_state = _read_json(run_dir / "final_state.json")
    operator_summary = ""
    operator_summary_path = run_dir / "operator_summary.md"
    if operator_summary_path.exists():
        try:
            operator_summary = operator_summary_path.read_text(encoding="utf-8").strip()
        except OSError:
            operator_summary = ""

    is_complete = bool(final_state)
    report = str(final_state.get("final_report") or "").strip()
    risk = final_state.get("risk_assessment", {}) if isinstance(final_state.get("risk_assessment"), dict) else {}
    quant = final_state.get("technical_analysis", {}) if isinstance(final_state.get("technical_analysis"), dict) else {}
    recommendation = str(quant.get("recommendation") or quant.get("decision") or "").strip()
    risk_grade = str(risk.get("grade") or "").strip()
    risk_summary = str(risk.get("summary") or "").strip()

    content = report or operator_summary
    title = "Final Result"
    if report:
        first_line = next((line.strip() for line in report.splitlines() if line.strip()), "")
        if first_line:
            title = first_line.lstrip("# ").strip() or title

    return {
        "run_id": run_id,
        "available": is_complete and bool(content),
        "is_complete": is_complete,
        "title": title,
        "content": content,
        "risk_grade": risk_grade or "n/a",
        "risk_summary": risk_summary or "",
        "recommendation": recommendation or "n/a",
        "has_report": bool(report),
        "has_operator_summary": bool(operator_summary),
    }


def _preview_launch_requirements(
    *,
    question: str,
    mode: str,
    seed: int,
    portfolio_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from investment_team import preview_launch_requirements

    return preview_launch_requirements(
        question,
        portfolio_context=portfolio_context or {},
        mode=mode,
        seed=seed,
    )


def _launch_investment_team_default(
    *,
    question: str,
    mode: str,
    seed: int,
    runs_dir: Path,
    portfolio_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected_runs = (_PROJECT_ROOT / "runs").resolve()
    if runs_dir.resolve() != expected_runs:
        raise RuntimeError(
            f"default launcher only supports runs dir {expected_runs}; got {runs_dir.resolve()}"
        )

    _LAUNCH_LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = _LAUNCH_LOG_DIR / f"launch_{stamp}.log"
    command = [
        sys.executable,
        str(_PROJECT_ROOT / "investment_team.py"),
        "--mode",
        mode,
        "--seed",
        str(seed),
        "--question",
        question,
    ]
    if portfolio_context:
        command.extend([
            "--portfolio-context-json",
            json.dumps(portfolio_context, ensure_ascii=False),
        ])

    with log_path.open("ab") as handle:
        process = subprocess.Popen(
            command,
            cwd=str(_PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    return {
        "pid": process.pid,
        "log_path": str(log_path),
        "command": command,
    }


def _render_index_html(*, runs: list[dict], launches: list[dict], initial_run_id: str) -> str:
    seed = {
        "runs": runs,
        "launches": launches,
        "initial_run_id": initial_run_id,
    }
    return """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent Empire</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;700&display=swap');

    :root {
      --bg: #07111f;
      --bg2: #111d32;
      --panel: rgba(8, 15, 28, 0.84);
      --line: rgba(137, 157, 183, 0.16);
      --text: #ecf4ff;
      --muted: #96aac5;
      --accent: #2fd5c4;
      --accent2: #f4c152;
      --accent3: #ff8a3d;
      --danger: #ff5e6b;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(47, 213, 196, 0.14), transparent 30%),
        radial-gradient(circle at 90% 15%, rgba(244, 193, 82, 0.14), transparent 24%),
        linear-gradient(180deg, #09101d 0%, #0a1322 52%, #111d32 100%);
      font-family: "Space Grotesk", system-ui, sans-serif;
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      background-image:
        linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
      background-size: 30px 30px;
      pointer-events: none;
      opacity: 0.26;
    }
    .app {
      position: relative;
      z-index: 1;
      display: grid;
      grid-template-columns: 340px minmax(0, 1fr);
      min-height: 100vh;
    }
    .sidebar {
      border-right: 1px solid var(--line);
      background: rgba(6, 11, 21, 0.86);
      backdrop-filter: blur(16px);
      padding: 18px 16px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .brand {
      padding: 8px 8px 0;
    }
    .brand-kicker {
      font: 12px "IBM Plex Mono", monospace;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .brand h1 {
      margin: 10px 0 8px;
      font-size: 34px;
      line-height: 0.95;
      letter-spacing: -0.05em;
    }
    .brand p {
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
      font-size: 14px;
    }
    .toolbar {
      display: grid;
      gap: 10px;
    }
    .launch-panel {
      display: grid;
      gap: 10px;
      padding: 12px;
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.06);
      background: rgba(255,255,255,0.03);
    }
    .launch-head {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 10px;
    }
    .launch-head h2 {
      margin: 0;
      font: 13px "IBM Plex Mono", monospace;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .launch-head span {
      color: var(--muted);
      font-size: 12px;
    }
    .launch-grid {
      display: grid;
      gap: 8px;
    }
    .launch-row {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 8px;
    }
    label.field {
      display: grid;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
    }
    label.field span {
      font: 11px "IBM Plex Mono", monospace;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }
    .toolbar-row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    button, .link-button {
      appearance: none;
      border: 0;
      border-radius: 12px;
      padding: 11px 12px;
      background: rgba(255,255,255,0.06);
      color: var(--text);
      font: inherit;
      cursor: pointer;
      text-decoration: none;
      text-align: center;
    }
    button:hover, .link-button:hover {
      background: rgba(255,255,255,0.1);
    }
    textarea, select, input[type="number"] {
      width: 100%;
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 12px;
      background: rgba(8, 15, 28, 0.82);
      color: var(--text);
      font: inherit;
      padding: 11px 12px;
    }
    textarea {
      min-height: 96px;
      resize: vertical;
      line-height: 1.5;
    }
    .launch-status {
      min-height: 20px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }
    .launch-status.success {
      color: var(--accent);
    }
    .launch-status.warning {
      color: var(--accent3);
    }
    .launch-status.error {
      color: #ff98a1;
    }
    .clarification-panel {
      display: grid;
      gap: 12px;
      padding: 14px;
      border-radius: 16px;
      border: 1px solid rgba(47, 213, 196, 0.18);
      background: rgba(8, 15, 28, 0.74);
    }
    .clarification-title {
      font: 12px "IBM Plex Mono", monospace;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--accent);
    }
    .clarification-message {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }
    .toggle {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px;
      border-radius: 14px;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.05);
      color: var(--muted);
      font-size: 13px;
    }
    .toggle input {
      accent-color: var(--accent);
    }
    .sidebar-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 0 8px;
    }
    .sidebar-head h2 {
      margin: 0;
      font: 13px "IBM Plex Mono", monospace;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .sidebar-head span {
      color: var(--muted);
      font-size: 12px;
    }
    .run-list {
      display: grid;
      gap: 10px;
      overflow: auto;
      padding-right: 2px;
    }
    .run-card {
      border-radius: 16px;
      padding: 14px;
      border: 1px solid rgba(255,255,255,0.06);
      background: rgba(255,255,255,0.03);
      cursor: pointer;
      transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
    }
    .run-card:hover {
      transform: translateY(-1px);
      border-color: rgba(47, 213, 196, 0.35);
    }
    .run-card.active {
      border-color: rgba(47, 213, 196, 0.55);
      background: rgba(47, 213, 196, 0.08);
      box-shadow: 0 0 0 1px rgba(47, 213, 196, 0.12);
    }
    .run-top {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: baseline;
      margin-bottom: 8px;
    }
    .run-id {
      font: 12px "IBM Plex Mono", monospace;
      color: var(--text);
    }
    .run-mode {
      font: 11px "IBM Plex Mono", monospace;
      color: var(--muted);
      text-transform: uppercase;
    }
    .run-request {
      color: var(--muted);
      line-height: 1.5;
      font-size: 13px;
      min-height: 38px;
    }
    .run-meta {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 10px;
    }
    .chip {
      padding: 5px 8px;
      border-radius: 999px;
      font: 11px "IBM Plex Mono", monospace;
      color: var(--muted);
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.03);
    }
    .chip.running {
      color: #05131a;
      background: var(--accent);
      border-color: transparent;
    }
    .chip.complete {
      color: #1c1302;
      background: var(--accent2);
      border-color: transparent;
    }
    .chip.incomplete {
      color: #fff5f6;
      background: var(--danger);
      border-color: transparent;
    }
    .chip.pending {
      color: #281300;
      background: var(--accent3);
      border-color: transparent;
    }
    .main {
      padding: 18px;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 14px;
    }
    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(280px, 0.7fr);
      gap: 14px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 18px 60px rgba(0,0,0,0.25);
      backdrop-filter: blur(14px);
    }
    .hero-main, .hero-side {
      padding: 18px 20px;
    }
    .hero-main h2 {
      margin: 4px 0 10px;
      font-size: 30px;
      line-height: 1;
      letter-spacing: -0.04em;
    }
    .hero-main p {
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
    }
    .hero-kicker {
      font: 12px "IBM Plex Mono", monospace;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .hero-side {
      display: grid;
      gap: 12px;
      align-content: start;
    }
    .stat-line {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.05);
      font-size: 13px;
    }
    .stat-line span:last-child {
      color: var(--muted);
      font-family: "IBM Plex Mono", monospace;
      text-align: right;
    }
    .viewer {
      min-height: 0;
      overflow: hidden;
      padding: 12px;
    }
    .result-panel {
      padding: 18px 20px;
      display: grid;
      gap: 12px;
      align-content: start;
    }
    .result-head {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
    }
    .result-head h3 {
      margin: 0;
      font-size: 22px;
      letter-spacing: -0.03em;
    }
    .result-subtle {
      color: var(--muted);
      font-size: 12px;
      font-family: "IBM Plex Mono", monospace;
    }
    .result-meta {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .result-body {
      max-height: 380px;
      overflow: auto;
      padding: 14px;
      border-radius: 16px;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.05);
      color: var(--text);
      white-space: pre-wrap;
      line-height: 1.6;
      font-size: 14px;
    }
    .result-empty {
      color: var(--muted);
      line-height: 1.7;
      padding: 10px 2px 2px;
    }
    iframe {
      display: block;
      width: 100%;
      height: 1180px;
      min-height: 960px;
      border: 0;
      border-radius: 16px;
      background: #09111e;
      overflow: hidden;
    }
    .empty {
      padding: 30px;
      text-align: center;
      color: var(--muted);
      line-height: 1.7;
    }
    .modal-shell {
      position: fixed;
      inset: 0;
      z-index: 20;
      display: grid;
      place-items: center;
      padding: 24px;
      background: rgba(4, 9, 17, 0.68);
      backdrop-filter: blur(12px);
    }
    .modal-shell[hidden] {
      display: none;
    }
    .modal-card {
      width: min(560px, 100%);
      padding: 22px;
      border-radius: 22px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(7, 16, 29, 0.96);
      box-shadow: 0 24px 80px rgba(0,0,0,0.35);
      display: grid;
      gap: 12px;
    }
    .modal-kicker {
      font: 12px "IBM Plex Mono", monospace;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--accent3);
    }
    .modal-card h3 {
      margin: 0;
      font-size: 28px;
      letter-spacing: -0.04em;
    }
    .modal-card p {
      margin: 0;
      color: var(--muted);
      line-height: 1.7;
      font-size: 14px;
      white-space: pre-wrap;
    }
    .modal-actions {
      display: flex;
      gap: 10px;
      justify-content: flex-end;
      flex-wrap: wrap;
    }
    .button-accent {
      background: linear-gradient(135deg, var(--accent), #5be1d3);
      color: #041219;
    }
    .button-danger {
      background: rgba(255, 94, 107, 0.16);
      color: #ffd8db;
      border: 1px solid rgba(255, 94, 107, 0.2);
    }
    @media (max-width: 1220px) {
      .app {
        grid-template-columns: 1fr;
      }
      .sidebar {
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      .hero {
        grid-template-columns: 1fr;
      }
      iframe {
        height: 980px;
        min-height: 760px;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-kicker">FastAPI Control Room</div>
        <h1>Agent Empire</h1>
        <p>런별 에이전트 상호작용을 탐색하고, 선택한 run의 리플레이 화면을 바로 열어봅니다.</p>
      </div>

      <div class="toolbar">
        <div class="toolbar-row">
          <button id="refresh-button" type="button">Refresh</button>
          <a class="link-button" id="open-current" href="#" target="_blank" rel="noopener noreferrer">Open Run</a>
        </div>
        <label class="toggle">
          <span>Auto refresh sidebar and selected run</span>
          <input id="auto-refresh" type="checkbox">
        </label>
      </div>

      <section class="launch-panel">
        <div class="launch-head">
          <h2>Launch Run</h2>
          <span>investment_team.py</span>
        </div>
        <div class="launch-grid">
          <label class="field">
            <span>Question</span>
            <textarea id="launch-question" placeholder="예: SPY를 지금 매수해도 될지 6개월 관점에서 분석해줘"></textarea>
          </label>
          <div class="launch-row">
            <label class="field">
              <span>Mode</span>
              <select id="launch-mode">
                <option value="mock">mock</option>
                <option value="live">live</option>
              </select>
            </label>
            <label class="field">
              <span>Seed</span>
              <input id="launch-seed" type="number" value="42">
            </label>
          </div>
          <div class="clarification-panel" id="launch-clarification" hidden>
            <div class="clarification-title">Position Review Intake</div>
            <div class="clarification-message" id="launch-clarification-message">보유 수량과 평단을 입력하면 position review를 정밀하게 계산합니다.</div>
            <div class="launch-row">
              <label class="field">
                <span>Ticker</span>
                <input id="clarify-ticker" type="text" placeholder="예: NVDA">
              </label>
              <label class="field">
                <span>Shares</span>
                <input id="clarify-shares" type="number" min="0" step="0.000001" placeholder="예: 25">
              </label>
            </div>
            <div class="launch-row">
              <label class="field">
                <span>Avg Cost</span>
                <input id="clarify-avg-cost" type="number" min="0" step="0.000001" placeholder="예: 132.5">
              </label>
              <label class="field">
                <span>Currency</span>
                <select id="clarify-currency">
                  <option value="USD">USD</option>
                  <option value="KRW">KRW</option>
                </select>
              </label>
            </div>
          </div>
          <button id="launch-button" type="button">Launch</button>
          <div class="launch-status" id="launch-status">질문을 입력하고 새 run을 시작할 수 있습니다.</div>
        </div>
      </section>

      <div class="sidebar-head">
        <h2>Recent Runs</h2>
        <span id="run-count">0</span>
      </div>
      <div class="run-list" id="run-list"></div>
    </aside>

    <main class="main">
      <section class="hero">
        <div class="panel hero-main">
          <div class="hero-kicker">Selected Run</div>
          <h2 id="selected-title">No run selected</h2>
          <p id="selected-request">runs 디렉터리에서 리플레이 가능한 run을 찾는 중입니다.</p>
        </div>
        <div class="panel hero-side" id="selected-stats"></div>
      </section>

      <section class="panel result-panel">
        <div class="result-head">
          <h3 id="result-title">Final Result</h3>
          <div class="result-subtle" id="result-status">waiting</div>
        </div>
        <div class="result-meta" id="result-meta"></div>
        <div class="result-body" id="result-body" hidden></div>
        <div class="result-empty" id="result-empty">complete 된 run을 선택하면 최종 보고서나 operator summary가 여기 표시됩니다.</div>
      </section>

      <section class="panel viewer" id="viewer-panel">
        <iframe id="run-frame" title="Agent Empire Replay"></iframe>
        <div class="empty" id="empty-state" hidden>표시할 run이 없습니다.</div>
      </section>
    </main>
  </div>
  <div class="modal-shell" id="launch-modal" hidden>
    <div class="modal-card">
      <div class="modal-kicker">Rate Limit Pending</div>
      <h3>계속 진행할까요?</h3>
      <p id="launch-modal-message">live run에서 API rate limit이 감지되면 여기서 계속 여부를 확인합니다.</p>
      <div class="modal-actions">
        <button id="launch-stop-button" type="button" class="button-danger">중단</button>
        <button id="launch-continue-button" type="button" class="button-accent">계속 진행</button>
      </div>
    </div>
  </div>

  <script>
    const seed = __SEED_JSON__;
    const ACTIVE_LAUNCH_STORAGE_KEY = "agent-empire-active-launch";
    const state = {
      runs: Array.isArray(seed.runs) ? seed.runs : [],
      launches: Array.isArray(seed.launches) ? seed.launches : [],
      selectedRunId: seed.initial_run_id || "",
      lastSelectedUpdatedAt: "",
      autoRefresh: true,
      timer: null,
      resultRequestSeq: 0,
      activeLaunchId: "",
      pendingClarification: null,
    };
    try {
      state.activeLaunchId = window.sessionStorage.getItem(ACTIVE_LAUNCH_STORAGE_KEY) || "";
    } catch (_err) {
      state.activeLaunchId = "";
    }

    const runListEl = document.getElementById("run-list");
    const runCountEl = document.getElementById("run-count");
    const selectedTitleEl = document.getElementById("selected-title");
    const selectedRequestEl = document.getElementById("selected-request");
    const selectedStatsEl = document.getElementById("selected-stats");
    const resultTitleEl = document.getElementById("result-title");
    const resultStatusEl = document.getElementById("result-status");
    const resultMetaEl = document.getElementById("result-meta");
    const resultBodyEl = document.getElementById("result-body");
    const resultEmptyEl = document.getElementById("result-empty");
    const runFrameEl = document.getElementById("run-frame");
    const emptyStateEl = document.getElementById("empty-state");
    const refreshButtonEl = document.getElementById("refresh-button");
    const autoRefreshEl = document.getElementById("auto-refresh");
    const openCurrentEl = document.getElementById("open-current");
    const launchQuestionEl = document.getElementById("launch-question");
    const launchModeEl = document.getElementById("launch-mode");
    const launchSeedEl = document.getElementById("launch-seed");
    const launchButtonEl = document.getElementById("launch-button");
    const launchStatusEl = document.getElementById("launch-status");
    const launchClarificationEl = document.getElementById("launch-clarification");
    const launchClarificationMessageEl = document.getElementById("launch-clarification-message");
    const clarifyTickerEl = document.getElementById("clarify-ticker");
    const clarifySharesEl = document.getElementById("clarify-shares");
    const clarifyAvgCostEl = document.getElementById("clarify-avg-cost");
    const clarifyCurrencyEl = document.getElementById("clarify-currency");
    const launchModalEl = document.getElementById("launch-modal");
    const launchModalMessageEl = document.getElementById("launch-modal-message");
    const launchContinueButtonEl = document.getElementById("launch-continue-button");
    const launchStopButtonEl = document.getElementById("launch-stop-button");

    function setFrameHeight(value) {
      const numeric = Number(value || 0);
      if (!Number.isFinite(numeric) || numeric <= 0) {
        return;
      }
      const clamped = Math.max(760, Math.min(6400, Math.ceil(numeric)));
      runFrameEl.style.height = `${clamped}px`;
    }

    function syncFrameHeight(retries = 0) {
      if (runFrameEl.hidden) {
        return;
      }
      try {
        const doc = runFrameEl.contentWindow && runFrameEl.contentWindow.document;
        if (!doc) {
          throw new Error("iframe document unavailable");
        }
        const root = doc.documentElement;
        const body = doc.body;
        const height = Math.max(
          root ? root.scrollHeight : 0,
          root ? root.offsetHeight : 0,
          body ? body.scrollHeight : 0,
          body ? body.offsetHeight : 0,
        );
        if (height > 0) {
          setFrameHeight(height + 12);
          return;
        }
      } catch (_err) {
        // same-origin expected; ignore transient load timing issues
      }
      if (retries < 8) {
        window.setTimeout(() => syncFrameHeight(retries + 1), 180);
      }
    }

    function formatDate(value) {
      if (!value) return "n/a";
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return value;
      return date.toLocaleString();
    }

    function setLaunchStatus(message, tone = "") {
      launchStatusEl.className = `launch-status ${tone}`.trim();
      launchStatusEl.textContent = message;
    }

    function collectLaunchPortfolioContext() {
      const ticker = String(clarifyTickerEl.value || "").trim().toUpperCase();
      const sharesRaw = String(clarifySharesEl.value || "").trim();
      const avgCostRaw = String(clarifyAvgCostEl.value || "").trim();
      const currency = String(clarifyCurrencyEl.value || "USD").trim().toUpperCase() || "USD";
      const context = {};
      const holdings = [];
      const hasShares = sharesRaw !== "";
      const hasAvgCost = avgCostRaw !== "";
      if (ticker || hasShares || hasAvgCost) {
        holdings.push({
          ticker,
          shares: hasShares ? Number(sharesRaw) : null,
          avg_cost: hasAvgCost ? Number(avgCostRaw) : null,
          currency,
        });
      }
      if (holdings.length) {
        context.holdings = holdings;
        context.primary_tickers = ticker ? [ticker] : [];
      }
      return context;
    }

    function resetClarification() {
      state.pendingClarification = null;
      launchClarificationEl.hidden = true;
      launchClarificationMessageEl.textContent = "보유 수량과 평단을 입력하면 position review를 정밀하게 계산합니다.";
    }

    function renderClarification(preview) {
      const clarification = preview && preview.clarification ? preview.clarification : null;
      if (!clarification || !clarification.required) {
        resetClarification();
        return;
      }
      state.pendingClarification = clarification;
      launchClarificationEl.hidden = false;
      launchClarificationMessageEl.textContent = clarification.message || "추가 입력이 필요합니다.";
      if (clarification.target_ticker && !String(clarifyTickerEl.value || "").trim()) {
        clarifyTickerEl.value = clarification.target_ticker;
      }
      if (clarification.currency) {
        clarifyCurrencyEl.value = clarification.currency;
      }
    }

    async function prepareLaunch(question, mode, seed, portfolioContext) {
      const response = await fetch("/api/launch/prepare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          mode,
          seed,
          portfolio_context: portfolioContext,
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "launch prepare failed");
      }
      return payload;
    }

    function setActiveLaunchId(value) {
      state.activeLaunchId = value || "";
      try {
        if (state.activeLaunchId) {
          window.sessionStorage.setItem(ACTIVE_LAUNCH_STORAGE_KEY, state.activeLaunchId);
        } else {
          window.sessionStorage.removeItem(ACTIVE_LAUNCH_STORAGE_KEY);
        }
      } catch (_err) {
        // ignore storage failures
      }
    }

    function launchById(launchId) {
      return state.launches.find((item) => item.launch_id === launchId) || null;
    }

    function launchForRun(runId) {
      if (!runId) {
        return null;
      }
      const candidates = state.launches.filter((item) => item.run_id === runId);
      candidates.sort((left, right) => String(right.updated_at || "").localeCompare(String(left.updated_at || "")));
      return candidates[0] || null;
    }

    function displayedRunStatus(run) {
      const launch = launchForRun(run && run.run_id);
      if (launch && launch.status === "pending_rate_limit") {
        return "pending";
      }
      if (launch && ["accepted", "running", "resuming"].includes(launch.status)) {
        return "running";
      }
      return run && run.status ? run.status : "incomplete";
    }

    function pendingLaunchForModal() {
      if (!state.activeLaunchId) {
        return null;
      }
      const active = launchById(state.activeLaunchId);
      return active && active.status === "pending_rate_limit" ? active : null;
    }

    function renderLaunchModal() {
      const launch = pendingLaunchForModal();
      if (!launch) {
        launchModalEl.hidden = true;
        return;
      }
      const apiLabel = launch.rate_limit_api_type === "llm"
        ? `LLM API: ${launch.rate_limit_api_source || "unknown"}${launch.rate_limit_api_scope ? ` (${launch.rate_limit_api_scope})` : ""}`
        : launch.rate_limit_api_type === "data"
          ? `Data API: ${launch.rate_limit_api_source || "unknown"}`
          : "API: unknown";
      const strategy = launch.resume_strategy === "signal"
        ? "프로세스를 일시정지해 둔 상태입니다."
        : "현재 프로세스가 이미 멈춰 재실행 방식으로 이어집니다.";
      launchModalMessageEl.textContent = [
        `질문: ${launch.question_preview || launch.question || "n/a"}`,
        apiLabel,
        `사유: ${launch.pending_reason || "API rate limit detected"}`,
        strategy,
      ].join("\\n");
      launchModalEl.hidden = false;
    }

    function renderResultPlaceholder(message, status = "waiting") {
      resultTitleEl.textContent = "Final Result";
      resultStatusEl.textContent = status;
      resultMetaEl.innerHTML = "";
      resultBodyEl.hidden = true;
      resultBodyEl.textContent = "";
      resultEmptyEl.hidden = false;
      resultEmptyEl.textContent = message;
    }

    function renderSelectedResult(result) {
      if (!result) {
        renderResultPlaceholder("결과를 불러오는 중입니다.", "loading");
        return;
      }
      resultTitleEl.textContent = result.title || "Final Result";
      resultStatusEl.textContent = result.is_complete ? "complete" : "pending";
      resultMetaEl.innerHTML = `
        <div class="chip">${result.risk_grade || "risk n/a"}</div>
        <div class="chip">${result.recommendation || "n/a"}</div>
        <div class="chip">${result.has_report ? "final_report" : (result.has_operator_summary ? "operator_summary" : "no_report")}</div>
      `;
      if (result.available && result.content) {
        resultBodyEl.hidden = false;
        resultBodyEl.textContent = result.content;
        resultEmptyEl.hidden = true;
        resultEmptyEl.textContent = "";
        return;
      }
      resultBodyEl.hidden = true;
      resultBodyEl.textContent = "";
      resultEmptyEl.hidden = false;
      resultEmptyEl.textContent = result.is_complete
        ? "이 run에는 표시할 final report가 없습니다."
        : "run이 complete 되면 final report나 operator summary가 여기 표시됩니다.";
    }

    function selectedRun() {
      return state.runs.find((item) => item.run_id === state.selectedRunId) || null;
    }

    function applyRunsSnapshot(nextRuns, { keepSelection = true } = {}) {
      const previousSelectedId = state.selectedRunId;
      state.runs = Array.isArray(nextRuns) ? nextRuns : [];

      if (keepSelection && previousSelectedId && state.runs.some((run) => run.run_id === previousSelectedId)) {
        state.selectedRunId = previousSelectedId;
      } else {
        state.selectedRunId = state.runs[0] ? state.runs[0].run_id : "";
      }

      renderRunList();
      const selected = selectedRun();
      const shouldReloadFrame =
        !selected ||
        selected.run_id !== previousSelectedId;
      if (selected) {
        state.lastSelectedUpdatedAt = String(selected.updated_at || "");
      } else {
        state.lastSelectedUpdatedAt = "";
      }
      renderSelectedRun(shouldReloadFrame);
      fetchSelectedResult(state.selectedRunId);
    }

    function applyLaunchSnapshot(nextLaunches) {
      state.launches = Array.isArray(nextLaunches) ? nextLaunches : [];
      const active = launchById(state.activeLaunchId);
      if (state.activeLaunchId && (!active || ["complete", "stopped"].includes(active.status))) {
        setActiveLaunchId("");
      }
      const currentActive = launchById(state.activeLaunchId);
      if (currentActive && currentActive.run_id && state.runs.some((run) => run.run_id === currentActive.run_id) && !state.selectedRunId) {
        state.selectedRunId = currentActive.run_id;
      }
      renderRunList();
      renderSelectedRun(false);
      renderLaunchModal();
    }

    function setSelectedRun(runId, options = {}) {
      state.selectedRunId = runId || "";
      const selected = selectedRun();
      state.lastSelectedUpdatedAt = selected ? String(selected.updated_at || "") : "";
      const url = new URL(window.location.href);
      if (state.selectedRunId) {
        url.searchParams.set("run", state.selectedRunId);
      } else {
        url.searchParams.delete("run");
      }
      history.replaceState({}, "", url);
      renderRunList();
      renderSelectedRun(options.reloadFrame !== false);
      fetchSelectedResult(state.selectedRunId);
    }

    function renderRunList() {
      runListEl.innerHTML = "";
      runCountEl.textContent = String(state.runs.length);
      for (const run of state.runs) {
        const status = displayedRunStatus(run);
        const card = document.createElement("article");
        card.className = `run-card ${run.run_id === state.selectedRunId ? "active" : ""}`;
        card.innerHTML = `
          <div class="run-top">
            <div class="run-id">${run.run_id}</div>
            <div class="run-mode">${run.mode || "unknown"}</div>
          </div>
          <div class="run-request">${run.request || "No request preview"}</div>
          <div class="run-meta">
            <div class="chip ${status || "incomplete"}">${status || "n/a"}</div>
            <div class="chip">${run.target_ticker || "n/a"}</div>
            <div class="chip">events ${run.event_count}</div>
            <div class="chip">${run.risk_grade || "risk n/a"}</div>
          </div>
        `;
        card.addEventListener("click", () => setSelectedRun(run.run_id));
        runListEl.appendChild(card);
      }
      if (state.runs.length === 0) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "No runs with events.jsonl were found.";
        runListEl.appendChild(empty);
      }
    }

    function renderSelectedRun(reloadFrame = true) {
      const run = selectedRun();
      if (!run) {
        selectedTitleEl.textContent = "No run selected";
        selectedRequestEl.textContent = "runs 디렉터리에서 리플레이 가능한 run이 없습니다.";
        selectedStatsEl.innerHTML = "";
        renderResultPlaceholder("표시할 run이 없습니다.", "empty");
        runFrameEl.hidden = true;
        emptyStateEl.hidden = false;
        openCurrentEl.href = "#";
        return;
      }

      selectedTitleEl.textContent = `${run.run_id} · ${run.target_ticker || "n/a"}`;
      selectedRequestEl.textContent = run.request || "No request preview available.";
      const launch = launchForRun(run.run_id);
      const status = displayedRunStatus(run);
      selectedStatsEl.innerHTML = `
        <div class="stat-line"><span>Status</span><span>${status || "unknown"}</span></div>
        <div class="stat-line"><span>Mode</span><span>${run.mode || "unknown"}</span></div>
        <div class="stat-line"><span>Events</span><span>${run.event_count}</span></div>
        <div class="stat-line"><span>Risk</span><span>${run.risk_grade || "n/a"}</span></div>
        <div class="stat-line"><span>Updated</span><span>${formatDate(run.updated_at)}</span></div>
        ${launch ? `<div class="stat-line"><span>Launch</span><span>${launch.status}</span></div>` : ""}
      `;
      openCurrentEl.href = `/runs/${encodeURIComponent(run.run_id)}`;
      runFrameEl.hidden = false;
      emptyStateEl.hidden = true;
      if (reloadFrame) {
        setFrameHeight(1180);
        runFrameEl.src = `/runs/${encodeURIComponent(run.run_id)}?ts=${Date.now()}`;
      } else {
        syncFrameHeight();
      }
    }

    async function fetchSelectedResult(runId) {
      if (!runId) {
        renderResultPlaceholder("표시할 run이 없습니다.", "empty");
        return;
      }
      const requestSeq = ++state.resultRequestSeq;
      renderResultPlaceholder("결과를 불러오는 중입니다.", "loading");
      try {
        const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/result`);
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "result fetch failed");
        }
        if (requestSeq !== state.resultRequestSeq || runId !== state.selectedRunId) {
          return;
        }
        renderSelectedResult(payload);
      } catch (error) {
        if (requestSeq !== state.resultRequestSeq || runId !== state.selectedRunId) {
          return;
        }
        renderResultPlaceholder(`결과를 불러오지 못했습니다: ${error.message}`, "error");
      }
    }

    async function fetchRuns({ keepSelection = true } = {}) {
      const response = await fetch("/api/runs");
      if (!response.ok) {
        return;
      }
      const payload = await response.json();
      applyRunsSnapshot(payload.runs, { keepSelection });
      applyLaunchSnapshot(payload.launches || []);
    }

    function stopAutoRefresh() {
      if (state.timer) {
        if (typeof state.timer.close === "function") {
          state.timer.close();
        } else {
          clearInterval(state.timer);
        }
        state.timer = null;
      }
    }

    function syncAutoRefresh() {
      state.autoRefresh = autoRefreshEl.checked;
      stopAutoRefresh();
      if (!state.autoRefresh) {
        return;
      }
      if (window.EventSource) {
        const source = new EventSource("/api/live/runs?limit=60");
        source.addEventListener("snapshot", (event) => {
          try {
            const payload = JSON.parse(event.data);
            applyRunsSnapshot(payload.runs, { keepSelection: true });
            applyLaunchSnapshot(payload.launches || []);
          } catch (_err) {
            // ignore malformed frames
          }
        });
        source.onerror = () => {};
        state.timer = source;
        return;
      }
      state.timer = setInterval(() => {
        fetchRuns({ keepSelection: true });
      }, 2500);
    }

    async function launchRun() {
      const question = String(launchQuestionEl.value || "").trim();
      const mode = String(launchModeEl.value || "mock").trim();
      const seed = Number(launchSeedEl.value || 42);
      const portfolioContext = collectLaunchPortfolioContext();
      if (!question) {
        setLaunchStatus("질문을 입력해야 run을 시작할 수 있습니다.", "error");
        launchQuestionEl.focus();
        return;
      }

      launchButtonEl.disabled = true;
      setLaunchStatus("질문 의도와 추가 입력 필요 여부를 확인 중입니다...", "");
      try {
        const preview = await prepareLaunch(question, mode, seed, portfolioContext);
        renderClarification(preview);
        if (preview.needs_clarification) {
          const fields = Array.isArray(preview.clarification && preview.clarification.fields)
            ? preview.clarification.fields.join(", ")
            : "shares, avg_cost";
          setLaunchStatus(`추가 입력이 필요합니다: ${fields}`, "warning");
          return;
        }
        const response = await fetch("/api/launch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, mode, seed, portfolio_context: portfolioContext }),
        });
        const payload = await response.json();
        if (!response.ok) {
          if (payload.detail && payload.detail.status === "needs_clarification") {
            renderClarification(payload.detail);
            const fields = Array.isArray(payload.detail.clarification && payload.detail.clarification.fields)
              ? payload.detail.clarification.fields.join(", ")
              : "shares, avg_cost";
            setLaunchStatus(`추가 입력이 필요합니다: ${fields}`, "warning");
            return;
          }
          throw new Error(payload.detail || "launch failed");
        }
        setActiveLaunchId(payload.launch_id || "");
        resetClarification();
        setLaunchStatus(
          `Launch accepted. pid=${payload.pid} · live 상태와 rate limit pending을 계속 추적합니다.`,
          "success",
        );
        await fetchRuns({ keepSelection: true });
      } catch (error) {
        setLaunchStatus(`Launch failed: ${error.message}`, "error");
      } finally {
        launchButtonEl.disabled = false;
      }
    }

    async function continueLaunch() {
      const launch = pendingLaunchForModal();
      if (!launch) {
        launchModalEl.hidden = true;
        return;
      }
      launchContinueButtonEl.disabled = true;
      launchStopButtonEl.disabled = true;
      try {
        const response = await fetch(`/api/launches/${encodeURIComponent(launch.launch_id)}/continue`, {
          method: "POST",
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "continue failed");
        }
        setActiveLaunchId(launch.launch_id);
        setLaunchStatus("Rate limit pending run을 다시 진행했습니다.", "success");
        await fetchRuns({ keepSelection: true });
      } catch (error) {
        setLaunchStatus(`Continue failed: ${error.message}`, "error");
      } finally {
        launchContinueButtonEl.disabled = false;
        launchStopButtonEl.disabled = false;
      }
    }

    async function stopLaunch() {
      const launch = pendingLaunchForModal();
      if (!launch) {
        launchModalEl.hidden = true;
        return;
      }
      launchContinueButtonEl.disabled = true;
      launchStopButtonEl.disabled = true;
      try {
        const response = await fetch(`/api/launches/${encodeURIComponent(launch.launch_id)}/stop`, {
          method: "POST",
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "stop failed");
        }
        setActiveLaunchId("");
        setLaunchStatus("Pending run을 중단했습니다.", "warning");
        await fetchRuns({ keepSelection: true });
      } catch (error) {
        setLaunchStatus(`Stop failed: ${error.message}`, "error");
      } finally {
        launchContinueButtonEl.disabled = false;
        launchStopButtonEl.disabled = false;
      }
    }

    refreshButtonEl.addEventListener("click", () => {
      fetchRuns({ keepSelection: true });
    });

    autoRefreshEl.addEventListener("change", syncAutoRefresh);
    launchQuestionEl.addEventListener("input", () => {
      resetClarification();
    });
    launchButtonEl.addEventListener("click", launchRun);
    launchContinueButtonEl.addEventListener("click", continueLaunch);
    launchStopButtonEl.addEventListener("click", stopLaunch);
    runFrameEl.addEventListener("load", () => {
      syncFrameHeight();
      window.setTimeout(() => syncFrameHeight(), 250);
      window.setTimeout(() => syncFrameHeight(), 1000);
    });
    window.addEventListener("resize", () => {
      window.setTimeout(() => syncFrameHeight(), 120);
    });

    autoRefreshEl.checked = true;
    renderRunList();
    renderLaunchModal();
    if ((!state.selectedRunId || !selectedRun()) && state.runs[0]) {
      state.selectedRunId = state.runs[0].run_id;
    }
    renderSelectedRun(true);
    fetchSelectedResult(state.selectedRunId);
    syncAutoRefresh();
  </script>
</body>
</html>
""".replace("__SEED_JSON__", _json_for_script(seed))


def create_app(
    *,
    runs_dir: str | Path = "runs",
    launch_run: Callable[..., dict[str, Any]] | None = None,
) -> FastAPI:
    app = FastAPI(title="Agent Empire", docs_url="/api/docs", redoc_url=None)
    runs_path = Path(runs_dir)
    launcher = launch_run or _launch_investment_team_default
    launch_records: dict[str, dict[str, Any]] = {}

    def _refresh_launches() -> list[dict[str, Any]]:
        for record in launch_records.values():
            _refresh_launch_record(record, runs_dir=runs_path)
        launches = [_launch_snapshot(record) for record in launch_records.values()]
        launches.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
        return launches

    def _runs_payload(limit: int) -> dict[str, Any]:
        launches = _refresh_launches()
        runs = _overlay_runs_with_launches(list_runs(runs_path, limit=limit), launches)
        return {"runs": runs, "launches": launches}

    def _get_launch_or_404(launch_id: str) -> dict[str, Any]:
        record = launch_records.get(launch_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"launch not found: {launch_id}")
        _refresh_launch_record(record, runs_dir=runs_path)
        return record

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/runs")
    def api_runs(limit: int = Query(default=40, ge=1, le=200)) -> dict[str, Any]:
        return _runs_payload(limit)

    @app.get("/api/runs/{run_id}")
    def api_run(run_id: str) -> dict:
        try:
            resolved = resolve_run_id(run_id, runs_dir=runs_path)
            return build_dashboard_model(resolved, runs_dir=runs_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/result")
    def api_run_result(run_id: str) -> dict[str, Any]:
        try:
            resolved = resolve_run_id(run_id, runs_dir=runs_path)
            return _load_run_result(resolved, runs_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/launch/prepare")
    def api_launch_prepare(payload: LaunchPrepareRequest) -> dict[str, Any]:
        question = str(payload.question or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="question is required")
        mode = str(payload.mode or "mock").strip().lower()
        if mode not in {"mock", "live"}:
            raise HTTPException(status_code=400, detail="mode must be 'mock' or 'live'")
        preview = _preview_launch_requirements(
            question=question,
            mode=mode,
            seed=int(payload.seed),
            portfolio_context=payload.portfolio_context or {},
        )
        return {
            "status": "needs_clarification" if preview.get("needs_clarification") else "ready",
            **preview,
        }

    @app.post("/api/launch")
    def api_launch(payload: LaunchRequest) -> dict[str, Any]:
        question = str(payload.question or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="question is required")
        mode = str(payload.mode or "mock").strip().lower()
        if mode not in {"mock", "live"}:
            raise HTTPException(status_code=400, detail="mode must be 'mock' or 'live'")
        preview = _preview_launch_requirements(
            question=question,
            mode=mode,
            seed=int(payload.seed),
            portfolio_context=payload.portfolio_context or {},
        )
        if preview.get("needs_clarification"):
            raise HTTPException(
                status_code=409,
                detail={
                    "status": "needs_clarification",
                    **preview,
                },
            )
        try:
            launch_info = launcher(
                question=question,
                mode=mode,
                seed=int(payload.seed),
                runs_dir=runs_path,
                portfolio_context=payload.portfolio_context or {},
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        launch_id = str(uuid4())
        launch_records[launch_id] = {
            "launch_id": launch_id,
            "question": question,
            "question_preview": question if len(question) <= 108 else question[:107].rstrip() + "…",
            "mode": mode,
            "seed": int(payload.seed),
            "portfolio_context": payload.portfolio_context or {},
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "status": "accepted",
            "pending_reason": "",
            "resume_strategy": "",
            "continue_count": 0,
            "log_offset": 0,
            "run_id": "",
            **launch_info,
        }
        return {
            "status": "accepted",
            "launch_id": launch_id,
            "mode": mode,
            "seed": int(payload.seed),
            **launch_info,
        }

    @app.get("/api/launches")
    def api_launches() -> dict[str, list[dict[str, Any]]]:
        return {"launches": _refresh_launches()}

    @app.post("/api/launches/{launch_id}/continue")
    def api_continue_launch(launch_id: str) -> dict[str, Any]:
        record = _get_launch_or_404(launch_id)
        if str(record.get("status", "")).strip() != "pending_rate_limit":
            raise HTTPException(status_code=409, detail="launch is not pending on rate limit")
        resumed = False
        if str(record.get("resume_strategy", "")).strip() == "signal":
            resumed = _signal_process(record.get("pid"), signal.SIGCONT)
        if resumed:
            record["status"] = "running"
            record["pending_reason"] = ""
            record["resume_strategy"] = "signal"
            record["updated_at"] = _now_iso()
            return _launch_snapshot(record)
        try:
            launch_info = launcher(
                question=str(record.get("question", "")),
                mode=str(record.get("mode", "live")),
                seed=int(record.get("seed", 42)),
                runs_dir=runs_path,
                portfolio_context=record.get("portfolio_context") if isinstance(record.get("portfolio_context"), dict) else {},
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        record.update(
            {
                "status": "accepted",
                "pending_reason": "",
                "resume_strategy": "",
                "updated_at": _now_iso(),
                "continue_count": int(record.get("continue_count", 0) or 0) + 1,
                "log_offset": 0,
                "run_id": "",
                **launch_info,
            }
        )
        return _launch_snapshot(record)

    @app.post("/api/launches/{launch_id}/stop")
    def api_stop_launch(launch_id: str) -> dict[str, Any]:
        record = _get_launch_or_404(launch_id)
        _signal_process(record.get("pid"), signal.SIGTERM)
        record["status"] = "stopped"
        record["pending_reason"] = ""
        record["resume_strategy"] = ""
        record["updated_at"] = _now_iso()
        return _launch_snapshot(record)

    @app.get("/api/live/runs")
    async def api_live_runs(
        limit: int = Query(default=60, ge=1, le=200),
        once: bool = Query(default=False),
    ) -> StreamingResponse:
        async def event_stream():
            previous_signature = ""
            while True:
                runs = list_runs(runs_path, limit=limit)
                launches = _refresh_launches()
                runs = _overlay_runs_with_launches(runs, launches)
                signature = json.dumps(
                    [
                        (
                            item.get("run_id"),
                            item.get("status"),
                            item.get("event_count"),
                            item.get("updated_at"),
                        )
                        for item in runs
                    ] + [
                        (
                            item.get("launch_id"),
                            item.get("status"),
                            item.get("run_id"),
                            item.get("updated_at"),
                        )
                        for item in launches
                    ],
                    ensure_ascii=False,
                )
                if signature != previous_signature:
                    previous_signature = signature
                    yield _sse_payload({"runs": runs, "launches": launches})
                else:
                    yield ": keep-alive\n\n"
                if once:
                    break
                await asyncio.sleep(2.0)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/api/live/runs/{run_id}")
    async def api_live_run(
        run_id: str,
        once: bool = Query(default=False),
    ) -> StreamingResponse:
        try:
            resolved = resolve_run_id(run_id, runs_dir=runs_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        async def event_stream():
            previous_signature = ""
            while True:
                model = build_dashboard_model(resolved, runs_dir=runs_path)
                signature = json.dumps(
                    (
                        model.get("event_count"),
                        model.get("status"),
                        model.get("updated_at"),
                    ),
                    ensure_ascii=False,
                )
                if signature != previous_signature:
                    previous_signature = signature
                    yield _sse_payload({"model": model})
                else:
                    yield ": keep-alive\n\n"
                if once:
                    break
                await asyncio.sleep(2.0)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/runs/latest")
    def latest_run_page() -> RedirectResponse:
        try:
            resolved = resolve_run_id("latest", runs_dir=runs_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return RedirectResponse(url=f"/runs/{resolved}")

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_page(run_id: str) -> HTMLResponse:
        try:
            resolved = resolve_run_id(run_id, runs_dir=runs_path)
            return HTMLResponse(render_run_dashboard_html(resolved, runs_dir=runs_path))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/", response_class=HTMLResponse)
    def index(run: str | None = None) -> HTMLResponse:
        payload = _runs_payload(60)
        runs = payload["runs"]
        initial_run_id = ""
        if run:
            initial_run_id = run
        elif runs:
            initial_run_id = str(runs[0].get("run_id", ""))
        return HTMLResponse(_render_index_html(runs=runs, launches=payload["launches"], initial_run_id=initial_run_id))

    return app


def create_default_app() -> FastAPI:
    return create_app(runs_dir=os.environ.get(_RUNS_DIR_ENV, "runs"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the Agent Empire FastAPI app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    import uvicorn

    runs_dir = str(Path(args.runs_dir).resolve())
    if args.reload:
        os.environ[_RUNS_DIR_ENV] = runs_dir
        uvicorn.run(
            "visualization.webapp:create_default_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
        )
    else:
        uvicorn.run(
            create_app(runs_dir=runs_dir),
            host=args.host,
            port=args.port,
            reload=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
