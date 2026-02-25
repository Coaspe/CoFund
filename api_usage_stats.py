"""
api_usage_stats.py — daily API usage counters
=============================================
Tracks per-API request/success/failure counts by UTC date.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

_LOCK = threading.Lock()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _today_utc() -> str:
    return _utc_now().strftime("%Y-%m-%d")


def _stats_root() -> Path:
    custom = str(os.environ.get("API_USAGE_STATS_DIR", "")).strip()
    if custom:
        return Path(custom)
    cache_dir = str(os.environ.get("CACHE_DIR", ".cache")).strip() or ".cache"
    return Path(cache_dir) / "api_usage"


def get_today_stats_path() -> Path:
    root = _stats_root()
    return root / f"{_today_utc()}.json"


def _empty_payload(date_utc: str) -> dict:
    return {
        "date_utc": date_utc,
        "updated_at_utc": "",
        "apis": {},
    }


def _load_payload(path: Path, date_utc: str) -> dict:
    if not path.exists():
        return _empty_payload(date_utc)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_payload(date_utc)
    if not isinstance(data, dict):
        return _empty_payload(date_utc)
    if str(data.get("date_utc", "")) != date_utc:
        return _empty_payload(date_utc)
    apis = data.get("apis")
    if not isinstance(apis, dict):
        data["apis"] = {}
    return data


def _write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    tmp_path.write_text(text + "\n", encoding="utf-8")
    tmp_path.replace(path)


def record_api_request(api_name: str, *, success: bool, category: str = "") -> None:
    """
    Append one API request result to today's stats file.
    Fail-open: never raises.
    """
    name = str(api_name or "").strip().lower() or "unknown"
    cat = str(category or "").strip().lower()
    date_utc = _today_utc()
    now_utc = _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")
    path = get_today_stats_path()

    try:
        with _LOCK:
            payload = _load_payload(path, date_utc)
            apis = payload.setdefault("apis", {})
            row = apis.setdefault(
                name,
                {
                    "category": cat,
                    "requests": 0,
                    "success": 0,
                    "failure": 0,
                    "last_status": "",
                    "last_updated_at_utc": "",
                },
            )
            if not row.get("category") and cat:
                row["category"] = cat
            row["requests"] = int(row.get("requests", 0)) + 1
            if success:
                row["success"] = int(row.get("success", 0)) + 1
                row["last_status"] = "success"
            else:
                row["failure"] = int(row.get("failure", 0)) + 1
                row["last_status"] = "failure"
            row["last_updated_at_utc"] = now_utc
            payload["updated_at_utc"] = now_utc
            _write_payload(path, payload)
    except Exception:
        return

