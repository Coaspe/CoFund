#!/usr/bin/env python3
"""
scripts/test_perplexity_only.py — Perplexity direct query test
===============================================================
라우터/폴백을 우회하고 Perplexity endpoint만 직접 호출합니다.

사용법:
  ./.venv/bin/python scripts/test_perplexity_only.py
  ./.venv/bin/python scripts/test_perplexity_only.py --prompt "Latest NVDA earnings summary with sources"
  ./.venv/bin/python scripts/test_perplexity_only.py --model sonar --timeout 90
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _get_api_key() -> str:
    return str(os.environ.get("PERPLEXITY_API_KEY", "")).strip()


def _build_endpoint(base_url: str) -> str:
    base = (base_url or "").strip().rstrip("/")
    if not base:
        base = "https://api.perplexity.ai"
    return f"{base}/chat/completions"


def _extract_content(data: dict) -> tuple[str, str]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", ""
    choice = choices[0] if isinstance(choices[0], dict) else {}
    finish_reason = str(choice.get("finish_reason", "") or "").strip()
    msg = choice.get("message")
    if not isinstance(msg, dict):
        return "", finish_reason
    content = str(msg.get("content", "") or "").strip()
    return content, finish_reason


def _extract_sources(data: dict) -> tuple[list[str], list[dict]]:
    citations: list[str] = []
    for c in data.get("citations", []) or []:
        if isinstance(c, str) and c.strip():
            citations.append(c.strip())
        elif isinstance(c, dict):
            u = str(c.get("url", "") or c.get("link", "")).strip()
            if u:
                citations.append(u)

    search_results: list[dict] = []
    raw = data.get("search_results")
    if isinstance(raw, list):
        for row in raw:
            if not isinstance(row, dict):
                continue
            search_results.append(
                {
                    "title": str(row.get("title", "") or "").strip(),
                    "url": str(row.get("url", "") or row.get("link", "")).strip(),
                    "date": str(row.get("date", "") or row.get("published_at", "")).strip(),
                }
            )
    return citations, search_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get("PERPLEXITY_MODEL", "sonar"),
    )
    parser.add_argument(
        "--prompt",
        default="Summarize latest NVDA earnings in 3 bullets with sources.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--show-json", action="store_true")
    args = parser.parse_args()

    key = _get_api_key()
    if not key:
        print("❌ PERPLEXITY_API_KEY not found in environment.")
        return 1

    endpoint = _build_endpoint(os.environ.get("PERPLEXITY_BASE_URL", ""))
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": "You are a factual research assistant. Provide concise answers.",
            },
            {"role": "user", "content": args.prompt},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    print("=" * 60)
    print("Perplexity Direct Test (no router / no fallback)")
    print("=" * 60)
    print(f"endpoint    : {endpoint}")
    print(f"model       : {args.model}")
    print(f"prompt      : {args.prompt}")
    print(f"timeout(sec): {args.timeout}")

    started = time.perf_counter()
    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=args.timeout)
    except requests.RequestException as exc:
        elapsed = time.perf_counter() - started
        print(f"\n❌ Request failed after {elapsed:.2f}s")
        print(f"error: {exc}")
        return 2

    elapsed = time.perf_counter() - started
    print(f"\nHTTP status : {resp.status_code}")
    print(f"latency(sec): {elapsed:.2f}")

    body_text = (resp.text or "").strip()
    if resp.status_code >= 400:
        print("\n❌ Perplexity API error response:")
        print(body_text[:3000])
        return 3

    try:
        data = resp.json()
    except json.JSONDecodeError:
        print("\n❌ Non-JSON response:")
        print(body_text[:3000])
        return 4

    content, finish_reason = _extract_content(data)
    citations, search_results = _extract_sources(data)

    print("\n✅ Perplexity response:")
    print(content or "(empty content)")
    if finish_reason:
        print(f"\nfinish_reason: {finish_reason}")

    if citations:
        print("\nCitations:")
        for i, u in enumerate(citations[:10], start=1):
            print(f"{i}. {u}")

    if search_results:
        print("\nSearch Results:")
        for i, row in enumerate(search_results[:10], start=1):
            title = row.get("title", "") or "(no title)"
            url = row.get("url", "") or "(no url)"
            date = row.get("date", "") or "-"
            print(f"{i}. [{date}] {title}")
            print(f"   {url}")

    usage = data.get("usage")
    if usage is not None:
        print("\nusage:")
        print(json.dumps(usage, indent=2, ensure_ascii=False))

    if args.show_json:
        print("\nraw_json:")
        print(json.dumps(data, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
