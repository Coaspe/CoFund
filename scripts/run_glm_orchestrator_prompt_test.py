#!/usr/bin/env python3
"""
Run GLM direct test with a prebuilt orchestrator-style prompt.

Usage:
  ./.venv/bin/python scripts/run_glm_orchestrator_prompt_test.py
  ./.venv/bin/python scripts/run_glm_orchestrator_prompt_test.py --timeout 300
  ./.venv/bin/python scripts/run_glm_orchestrator_prompt_test.py --model glm-4.7-flash
  ./.venv/bin/python scripts/run_glm_orchestrator_prompt_test.py --max-tokens 512
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_PROMPT = """당신은 헤지펀드의 총괄 PM/CIO다.
아래 입력을 바탕으로 반드시 JSON만 출력하라. 설명/마크다운/코드블록 금지.

출력 스키마:
{
  "current_iteration": <int>,
  "action_type": "initial_delegation | scale_down | add_hedge | pivot_strategy | fallback_abort",
  "investment_brief": {
    "rationale": "<판단 근거>",
    "target_universe": ["<TICKER>", "..."]
  },
  "desk_tasks": {
    "macro": {"horizon_days": <int>, "focus_areas": ["..."]},
    "fundamental": {"horizon_days": <int>, "focus_areas": ["..."]},
    "sentiment": {"horizon_days": <int>, "focus_areas": ["..."]},
    "quant": {"horizon_days": <int>, "risk_budget": "Conservative|Moderate|Aggressive", "focus_areas": ["..."]}
  }
}

입력:
[현재 시각] 2026-02-25T02:46:26Z
[iteration_count] 0
[사용자 원본 요청] 애플(AAPL) 주식을 지금 매수해도 괜찮을까요? 6개월 투자 관점에서 분석해 주세요.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="glm-4.7-flash")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    target = repo_root / "scripts" / "test_glm_only.py"

    cmd = [
        sys.executable,
        str(target),
        "--model",
        str(args.model),
        "--timeout",
        str(args.timeout),
        "--max-tokens",
        str(args.max_tokens),
        "--prompt",
        str(args.prompt),
    ]

    print("Running:", " ".join(cmd[:4]), "...", flush=True)
    return subprocess.call(cmd, cwd=str(repo_root))


if __name__ == "__main__":
    raise SystemExit(main())
