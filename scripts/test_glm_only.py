#!/usr/bin/env python3
"""
scripts/test_glm_only.py — GLM(Z.ai) direct query test
=======================================================
라우터/폴백을 완전히 우회하고 Z.ai GLM endpoint만 직접 호출합니다.

사용법:
  ./.venv/bin/python scripts/test_glm_only.py
  ./.venv/bin/python scripts/test_glm_only.py --prompt "Reply with OK only."
  ./.venv/bin/python scripts/test_glm_only.py --model glm-4.7-flash --timeout 30
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="glm-4.7-flash")
    parser.add_argument("--prompt", default="Reply with 'OK' only.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=40.0)
    parser.add_argument("--show-thinking", action="store_true")
    parser.add_argument("--stream", action="store_true")
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
    if args.stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}

    print("=" * 60)
    print("GLM Direct Test (no router / no fallback)")
    print("=" * 60)
    print(f"endpoint    : {endpoint}")
    print(f"model       : {args.model}")
    print(f"prompt      : {args.prompt}")
    print(f"timeout(sec): {args.timeout}")

    started = time.perf_counter()
    try:
        resp = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=args.timeout,
            stream=args.stream,
        )
    except requests.RequestException as exc:
        elapsed = time.perf_counter() - started
        print(f"\n❌ Request failed after {elapsed:.2f}s")
        print(f"error: {exc}")
        return 2

    elapsed = time.perf_counter() - started
    print(f"\nHTTP status : {resp.status_code}")
    print(f"latency(sec): {elapsed:.2f}")

    body_text = resp.text.strip()
    if resp.status_code >= 400:
        print("\n❌ GLM API error response:")
        print(body_text[:2000])
        return 3

    if args.stream:
        finish_reason = ""
        content = ""
        reasoning_content = ""
        usage = None
        thinking_started = False
        content_started = False

        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = str(raw_line).strip()
            if not line.startswith("data:"):
                continue
            data_line = line[5:].strip()
            if data_line == "[DONE]":
                break
            try:
                chunk = json.loads(data_line)
            except json.JSONDecodeError:
                continue

            if isinstance(chunk.get("usage"), dict):
                usage = chunk.get("usage")

            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0] if isinstance(choices[0], dict) else {}
            if not finish_reason and choice.get("finish_reason"):
                finish_reason = str(choice.get("finish_reason", "")).strip()

            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                continue

            piece_reasoning = str(delta.get("reasoning_content", "") or "")
            if piece_reasoning and args.show_thinking:
                if not thinking_started:
                    print("\n🧠 reasoning_content (stream):")
                    thinking_started = True
                print(piece_reasoning, end="", flush=True)
            reasoning_content += piece_reasoning

            piece_content = str(delta.get("content", "") or "")
            if piece_content:
                if not content_started:
                    print("\n\n✅ GLM response (stream):")
                    content_started = True
                print(piece_content, end="", flush=True)
            content += piece_content

        if args.show_thinking and thinking_started:
            print("")
        if content_started:
            print("")
        if not content_started:
            print("\n✅ GLM response:")
            print("(empty content)")
        if finish_reason:
            print(f"\nfinish_reason: {finish_reason}")
        if args.show_thinking and not reasoning_content:
            print("\n🧠 reasoning_content:")
            print("(empty reasoning_content)")
        if usage is not None:
            print("\nusage:")
            print(json.dumps(usage, indent=2, ensure_ascii=False))
        return 0

    try:
        data = resp.json()
    except json.JSONDecodeError:
        print("\n❌ Non-JSON response:")
        print(body_text[:2000])
        return 4

    choice = None
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]

    content = ""
    reasoning_content = ""
    finish_reason = ""
    if isinstance(choice, dict):
        finish_reason = str(choice.get("finish_reason", "")).strip()
        msg = choice.get("message", {})
        if isinstance(msg, dict):
            content = str(msg.get("content", "")).strip()
            reasoning_content = str(msg.get("reasoning_content", "")).strip()

    print("\n✅ GLM response:")
    print(content or "(empty content)")
    if finish_reason:
        print(f"\nfinish_reason: {finish_reason}")
    if args.show_thinking:
        print("\n🧠 reasoning_content:")
        print(reasoning_content or "(empty reasoning_content)")

    usage = data.get("usage")
    if usage is not None:
        print("\nusage:")
        print(json.dumps(usage, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
