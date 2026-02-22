"""
data_providers/rate_limiter.py — Token-bucket QPS limiter
=========================================================
Simple per-provider rate limiter. Blocks until a token is available.
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    def __init__(self, qps: float = 2.0, burst: int = 5):
        self._interval = 1.0 / max(qps, 0.01)
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._burst, self._tokens + elapsed / self._interval)
            self._last_refill = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return

            sleep_time = (1.0 - self._tokens) * self._interval
        time.sleep(sleep_time)
        with self._lock:
            self._tokens = 0.0
            self._last_refill = time.monotonic()
