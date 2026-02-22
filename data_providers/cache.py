"""
data_providers/cache.py — SQLite disk cache with TTL
=====================================================
Key includes provider + endpoint + params hash.
Stores fetched_at timestamp for freshness scoring.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class DiskCache:
    def __init__(self, cache_dir: str = ".cache", default_ttl: int = 3600):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._dir / "provider_cache.db"
        self._default_ttl = default_ttl
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key       TEXT PRIMARY KEY,
                    value     TEXT NOT NULL,
                    fetched_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
            """)

    @staticmethod
    def make_key(provider: str, endpoint: str, params: dict | None = None) -> str:
        raw = f"{provider}:{endpoint}:{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        now = time.time()
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute(
                "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None or row[1] < now:
            return None
        return json.loads(row[0])

    def set(self, key: str, value: Any, ttl_sec: int | None = None):
        now = time.time()
        ttl = ttl_sec if ttl_sec is not None else self._default_ttl
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, fetched_at, expires_at) VALUES (?, ?, ?, ?)",
                (key, json.dumps(value, default=str), now, now + ttl),
            )

    def clear(self):
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("DELETE FROM cache")
