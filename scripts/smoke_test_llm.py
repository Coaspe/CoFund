#!/usr/bin/env python3
"""
scripts/smoke_test_llm.py — LLM Router Smoke Test
===================================================
각 에이전트의 LLM 호출이 성공/실패(fallback)하는지 1회씩 확인합니다.

사용법:
  python scripts/smoke_test_llm.py
"""

from __future__ import annotations

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm.router import get_llm, AGENT_LLM_CONFIG, _get_zai_key, _get_groq_key, _get_gemini_key


def main():
    SEP = "=" * 60
    print(SEP)
    print("🔍 LLM Router Smoke Test")
    print(SEP)

    # 환경변수 상태
    print("\n📋 API Key Status:")
    zai_ok = bool(_get_zai_key())
    groq_ok = bool(_get_groq_key())
    gemini_ok = bool(_get_gemini_key())
    openai_ok = bool(os.environ.get("OPENAI_API_KEY", ""))
    print(f"   ZAI_API_KEY:     {'✅ SET' if zai_ok else '❌ NOT SET'}")
    print(f"   GROQ_API_KEY:    {'✅ SET' if groq_ok else '❌ NOT SET'}")
    print(f"   GOOGLE_API_KEY:  {'✅ SET' if gemini_ok else '❌ NOT SET'}")
    print(f"   OPENAI_API_KEY:  {'✅ SET' if openai_ok else '⚪ NOT SET (legacy)'}")

    print(f"\n{'─' * 60}")
    print("🧪 Per-Agent LLM Availability:")
    print(f"{'─' * 60}")

    results = {}
    for agent_name, config in AGENT_LLM_CONFIG.items():
        enabled = config.get("enabled", True)
        provider = config["provider"]
        model = config["model"]

        if not enabled:
            status = "⚪ DISABLED (Python-only)"
            llm = None
        else:
            llm = get_llm(agent_name)
            if llm is not None:
                status = f"✅ {provider.upper()} → {model}"
            else:
                status = f"⚠️ MOCK FALLBACK (no {provider} key)"

        results[agent_name] = llm is not None
        print(f"   {agent_name:15s} | {status}")

    # 간단한 연결 테스트 (API 키가 있는 경우에만)
    print(f"\n{'─' * 60}")
    print("🔌 Connection Test (agents with available LLM):")
    print(f"{'─' * 60}")

    tested = 0
    for agent_name in ["macro", "report_writer"]:
        llm = get_llm(agent_name)
        if llm is None:
            print(f"   {agent_name:15s} | ⏭️ SKIPPED (no LLM available)")
            continue

        try:
            from langchain_core.messages import HumanMessage
            resp = llm.invoke([HumanMessage(content="Say 'OK' and nothing else.")])
            content = resp.content.strip()[:50]
            print(f"   {agent_name:15s} | ✅ Response: '{content}'")
            tested += 1
        except Exception as exc:
            print(f"   {agent_name:15s} | ❌ ERROR: {exc}")

    if tested == 0:
        print("   (No agents tested — all running in Mock mode)")

    # 요약
    print(f"\n{SEP}")
    print("📊 Summary:")
    active_llm = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"   LLM active: {active_llm}/{total} agents")
    print(f"   Mock mode:  {total - active_llm}/{total} agents")

    if not zai_ok and not groq_ok and not gemini_ok:
        print(f"\n   💡 전체 Mock 모드로 실행됩니다.")
        print(f"      Z.ai 사용: export ZAI_API_KEY='...'")
        print(f"      Groq 사용: export GROQ_API_KEY='gsk_...'")
        print(f"      Gemini 사용: export GOOGLE_API_KEY='AI...'")

    print(f"\n{'✅ Smoke test 완료!' if True else ''}")


if __name__ == "__main__":
    main()
