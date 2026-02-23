"""
storage/pit_store.py — Point-in-Time Snapshot Store
====================================================
모든 provider 결과, LLM I/O, Gate trace, 최종 보고서를
runs/{run_id}/ 아래에 Run-ID 기준으로 저장합니다.

Directory layout:
  runs/{run_id}/raw/{provider}/{request_hash}.json
  runs/{run_id}/features/{agent}.json
  runs/{run_id}/decisions/risk_gate_trace.json
  runs/{run_id}/llm_io/{agent}_prompt.txt
  runs/{run_id}/llm_io/{agent}_response.txt
  runs/{run_id}/final_report/report.md
  runs/{run_id}/config/config_hash.txt
  runs/{run_id}/config/env.example
  runs/{run_id}/config/config.json
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ── 루트 경로 (파일 기준) ────────────────────────────────────────────────────

_HERE = Path(__file__).parent.parent  # project root
RUNS_ROOT = _HERE / "runs"


def _run_dir(run_id: str) -> Path:
    return RUNS_ROOT / run_id


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── Provider 스냅샷 ──────────────────────────────────────────────────────────

def make_request_hash(provider: str, endpoint: str, params: dict, as_of: str) -> str:
    """
    as_of를 포함한 결정론적 해시.
    동일한 (provider, endpoint, params, as_of) → 항상 동일한 hash.
    """
    payload = {
        "provider": provider,
        "endpoint": endpoint,
        "params": params,
        "as_of": as_of,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:24]


def save_snapshot(
    run_id: str,
    provider: str,
    request_hash: str,
    payload: dict,
) -> Path:
    """Provider 결과를 raw 스냅샷으로 저장."""
    dest_dir = _ensure(_run_dir(run_id) / "raw" / provider)
    path = dest_dir / f"{request_hash}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return path


def load_snapshot(run_id: str, provider: str, request_hash: str) -> Optional[dict]:
    """저장된 스냅샷 로드. 없으면 None."""
    path = _run_dir(run_id) / "raw" / provider / f"{request_hash}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def check_lookahead(
    source_timestamp: Optional[str],
    as_of: str,
    mode: str = "live",
) -> Optional[str]:
    """
    source_timestamp > as_of 이면 lookahead_violation 반환.
    backtest 모드: raise RuntimeError (fail-fast)
    live 모드: 경고 문자열 반환
    """
    if not source_timestamp:
        return None
    try:
        ts = datetime.fromisoformat(source_timestamp.replace("Z", "+00:00"))
        ao = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        if ts > ao:
            msg = f"lookahead_violation: source_timestamp {source_timestamp} > as_of {as_of}"
            if mode == "backtest":
                raise RuntimeError(f"[PIT] {msg}")
            return "lookahead_violation"
    except (ValueError, TypeError):
        pass
    return None


# ── Features 저장 ────────────────────────────────────────────────────────────

def save_features(run_id: str, agent: str, features: dict) -> Path:
    dest_dir = _ensure(_run_dir(run_id) / "features")
    path = dest_dir / f"{agent}.json"
    path.write_text(json.dumps(features, ensure_ascii=False, indent=2, default=str))
    return path


# ── Risk Gate Trace ──────────────────────────────────────────────────────────

def save_gate_trace(run_id: str, gate_trace: list) -> Path:
    dest_dir = _ensure(_run_dir(run_id) / "decisions")
    path = dest_dir / "risk_gate_trace.json"
    path.write_text(json.dumps(gate_trace, ensure_ascii=False, indent=2, default=str))
    return path


def save_positions(run_id: str, proposed: dict, final: dict) -> Path:
    dest_dir = _ensure(_run_dir(run_id) / "decisions")
    path = dest_dir / "positions.json"
    path.write_text(json.dumps(
        {"positions_proposed": proposed, "positions_final": final},
        ensure_ascii=False, indent=2, default=str,
    ))
    return path


# ── LLM I/O 로그 ─────────────────────────────────────────────────────────────

def save_llm_io(run_id: str, agent_name: str, prompt: str, response: str) -> None:
    dest_dir = _ensure(_run_dir(run_id) / "llm_io")
    (dest_dir / f"{agent_name}_prompt.txt").write_text(prompt, encoding="utf-8")
    (dest_dir / f"{agent_name}_response.txt").write_text(response, encoding="utf-8")


# ── 최종 보고서 ───────────────────────────────────────────────────────────────

def save_final_report(run_id: str, report_md: str) -> Path:
    dest_dir = _ensure(_run_dir(run_id) / "final_report")
    path = dest_dir / "report.md"
    path.write_text(report_md, encoding="utf-8")
    return path


# ── Config 스냅샷 (Amendment 5) ───────────────────────────────────────────────

def save_config_snapshot(run_id: str, extra_config: Optional[dict] = None) -> str:
    """
    config/ 아래에 config.json + .env.example 사본 + config_hash 저장.
    Returns: config_hash (hex)
    """
    dest_dir = _ensure(_run_dir(run_id) / "config")

    # .env.example 복사
    env_example = _HERE / ".env.example"
    if env_example.exists():
        shutil.copy(env_example, dest_dir / "env.example")

    # config.json 저장
    config_data = extra_config or {}
    config_json = json.dumps(config_data, sort_keys=True, ensure_ascii=False, indent=2)
    (dest_dir / "config.json").write_text(config_json)

    # config_hash
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]
    (dest_dir / "config_hash.txt").write_text(config_hash)

    return config_hash
