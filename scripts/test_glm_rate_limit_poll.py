#!/usr/bin/env python3
"""
scripts/test_glm_rate_limit_poll.py — GLM(Z.ai) rate-limit polling test
========================================================================
라우터/폴백을 우회하고 Z.ai GLM endpoint를 직접 호출합니다.
rate limit(429/관련 메시지) 또는 일시적 네트워크 오류가 나면
지정된 간격으로 재시도하며, 성공 응답이 올 때까지 반복합니다.

사용법:
  ./.venv/bin/python scripts/test_glm_rate_limit_poll.py
  ./.venv/bin/python scripts/test_glm_rate_limit_poll.py --poll-interval 10
  ./.venv/bin/python scripts/test_glm_rate_limit_poll.py --max-attempts 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


_RATE_LIMIT_TOKENS = (
    "429",
    "rate limit",
    "too many requests",
    "quota",
    "resource exhausted",
    "tokens per minute",
    "tpm",
    "rpm",
)


def _get_zai_key() -> str:
    return (
        os.environ.get("ZAI_API_KEY", "")
        or os.environ.get("ZHIPU_API_KEY", "")
        or os.environ.get("BIGMODEL_API_KEY", "")
    ).strip()


def _build_endpoint(base_url: str) -> str:
    base = (base_url or "").strip().rstrip("/")
    if not base:
        base = "https://api.z.ai/api/paas/v4"
    return f"{base}/chat/completions"


def _is_rate_limit(status_code: int, body_text: str) -> bool:
    if status_code == 429:
        return True
    text = (body_text or "").lower()
    return any(token in text for token in _RATE_LIMIT_TOKENS)


def _retry_after_seconds(resp: requests.Response, default_wait: float) -> float:
    raw = (resp.headers.get("Retry-After", "") or "").strip()
    if not raw:
        return default_wait
    try:
        value = float(raw)
        if value > 0:
            return value
    except ValueError:
        pass
    return default_wait


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="glm-4.7-flash")
    parser.add_argument("--prompt", default="Reply with 'OK' only.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=40.0)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="0 means unlimited retry until success",
    )
    args = parser.parse_args()

    key = _get_zai_key()
    if not key:
        print("❌ Z.ai API key not found. Set ZAI_API_KEY (or ZHIPU_API_KEY/BIGMODEL_API_KEY).")
        return 1

    endpoint = _build_endpoint(os.environ.get("ZAI_BASE_URL", ""))
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    print("=" * 60)
    print("GLM Rate Limit Poll Test (no router / no fallback)")
    print("=" * 60)
    print(f"endpoint       : {endpoint}")
    print(f"model          : {args.model}")
    print(f"poll_interval  : {args.poll_interval}s")
    print(f"max_attempts   : {args.max_attempts if args.max_attempts > 0 else 'unlimited'}")
    print(f"timeout(sec)   : {args.timeout}")

    attempt = 0
    while True:
        attempt += 1
        print(f"\n[{_now()}] attempt #{attempt}")
        started = time.perf_counter()

        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=args.timeout)
        except requests.RequestException as exc:
            elapsed = time.perf_counter() - started
            print(f"⚠️ request failed ({elapsed:.2f}s): {exc}")
            if args.max_attempts > 0 and attempt >= args.max_attempts:
                print("❌ max attempts reached.")
                return 2
            print(f"↻ retry in {args.poll_interval:.1f}s ...")
            time.sleep(args.poll_interval)
            continue

        elapsed = time.perf_counter() - started
        status = int(resp.status_code)
        body_text = resp.text.strip()
        print(f"HTTP {status} ({elapsed:.2f}s)")

        if status < 400:
            try:
                data = resp.json()
            except json.JSONDecodeError:
                print("❌ success status but non-JSON response:")
                print(body_text[:2000])
                return 3

            content = ""
            choices = data.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                msg = choices[0].get("message", {})
                if isinstance(msg, dict):
                    content = str(msg.get("content", "")).strip()

            print("\n✅ success response:")
            print(content or "(empty content)")
            usage = data.get("usage")
            if usage is not None:
                print("\nusage:")
                print(json.dumps(usage, indent=2, ensure_ascii=False))
            return 0

        if _is_rate_limit(status, body_text):
            wait_sec = _retry_after_seconds(resp, args.poll_interval)
            print("⚠️ rate limit detected.")
            print(body_text[:600] if body_text else "(empty body)")
            if args.max_attempts > 0 and attempt >= args.max_attempts:
                print("❌ max attempts reached.")
                return 4
            print(f"↻ retry in {wait_sec:.1f}s ...")
            time.sleep(wait_sec)
            continue

        if status >= 500:
            print("⚠️ server-side temporary error, retrying ...")
            print(body_text[:600] if body_text else "(empty body)")
            if args.max_attempts > 0 and attempt >= args.max_attempts:
                print("❌ max attempts reached.")
                return 5
            print(f"↻ retry in {args.poll_interval:.1f}s ...")
            time.sleep(args.poll_interval)
            continue

        print("❌ non-retryable error:")
        print(body_text[:2000] if body_text else "(empty body)")
        return 6


if __name__ == "__main__":
    sys.exit(main())

