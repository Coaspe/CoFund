"""
telemetry.py — JSONL 이벤트 로깅 + 런 감사 시스템
===================================================
CHANGELOG:
  v1.0 (2026-02-22) — 신규 생성. init_run, log_event, save_final_state.

Iron Rule R2: 모든 노드 이벤트를 JSONL로 기록하여 재현성/감사 보장.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


RUNS_DIR = Path("runs")


def init_run(run_id: str, mode: str = "mock") -> Path:
    """
    runs/{run_id}/ 폴더를 생성하고 메타 정보를 기록한다.

    Returns:
        생성된 런 디렉토리 Path.
    """
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_id": run_id,
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    return run_dir


def log_event(
    run_id: str,
    node_name: str,
    iteration: int,
    *,
    phase: str = "exit",
    inputs_summary: Optional[dict] = None,
    outputs_summary: Optional[dict] = None,
    errors: Optional[list] = None,
) -> None:
    """
    runs/{run_id}/events.jsonl에 노드 이벤트를 한 줄 추가.

    Args:
        run_id:          런 ID
        node_name:       노드 이름 (e.g., "orchestrator", "risk_manager")
        iteration:       현재 iteration
        phase:           "enter" | "exit"
        inputs_summary:  입력 요약 (핵심 필드만)
        outputs_summary: 출력 요약 (핵심 필드만)
        errors:          에러 목록
    """
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "node_name": node_name,
        "iteration": iteration,
        "phase": phase,
        "inputs_summary": _safe_truncate(inputs_summary),
        "outputs_summary": _safe_truncate(outputs_summary),
        "errors": errors or [],
    }

    events_path = run_dir / "events.jsonl"
    with open(events_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def save_final_state(run_id: str, state: dict) -> Path:
    """
    runs/{run_id}/final_state.json에 최종 상태를 저장.

    Returns:
        저장된 파일 Path.
    """
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    path = run_dir / "final_state.json"
    # 대용량 필드(보고서 등) 잘리지 않게 전체 저장
    serializable = _make_serializable(state)
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2, default=str))
    return path


def _safe_truncate(obj: Any, max_str_len: int = 500) -> Any:
    """JSONL 기록 시 과도하게 큰 문자열을 잘라낸다."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj[:max_str_len] + ("..." if len(obj) > max_str_len else "")
    if isinstance(obj, dict):
        return {k: _safe_truncate(v, max_str_len) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_truncate(v, max_str_len) for v in obj[:20]]
    return obj


def _make_serializable(obj: Any) -> Any:
    """numpy/기타 비직렬화 객체를 JSON-safe로 변환."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return obj
