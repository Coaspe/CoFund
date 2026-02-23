"""
backtest/runner.py — PIT-based Deterministic Backtest Runner
============================================================
재현 가능한 포인트-인-타임 백테스트.

사용법:
  ./.venv/bin/python backtest/runner.py \\
    --start 2024-01-01 --end 2024-06-30 \\
    --universe AAPL MSFT --mode mock --seed 42

출력:
  runs/backtest_{run_id}/backtest_results.csv
  runs/backtest_{run_id}/backtest_summary.json
  runs/backtest_{run_id}/config/config_hash.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from schemas.common import create_initial_state, generate_run_id, now_iso
from storage.pit_store import save_config_snapshot, save_gate_trace, save_positions, save_final_report

# ── 비용 모델 ─────────────────────────────────────────────────────────────────
COST_BPS = 5       # 편도 거래비용 (기본 5bp)
SLIPPAGE_BPS = 2   # 슬리피지 (기본 2bp)
TOTAL_COST_PER_TURN = (COST_BPS + SLIPPAGE_BPS) / 10_000.0  # per unit

# ── 리밸런싱 주기 ─────────────────────────────────────────────────────────────
REBALANCE_FREQ_DAYS = {"daily": 1, "weekly": 5, "biweekly": 10, "monthly": 21}


def _get_rebalance_dates(start: str, end: str, freq: str) -> list[str]:
    """리밸런싱 날짜 목록 생성."""
    step = REBALANCE_FREQ_DAYS.get(freq, 5)
    s = datetime.fromisoformat(start)
    e = datetime.fromisoformat(end)
    dates = []
    cur = s
    while cur <= e:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=step)
    return dates


def _run_pipeline_for_date(
    as_of: str,
    universe: list[str],
    mode: str,
    seed: int,
    run_id: str,
) -> dict:
    """단일 날짜에 대해 7-Agent 파이프라인 실행 (mock 모드)."""
    # 재현성을 위해 seed + date hash로 결정론적 RNG
    date_seed = seed ^ int(hashlib.sha256(as_of.encode()).hexdigest()[:8], 16)
    rng = random.Random(date_seed)

    # Mock 포지션 생성 (결정론적)
    positions: dict[str, float] = {}
    for ticker in universe:
        # 재현 가능한 mock 포지션
        alloc = rng.uniform(0.03, 0.15)
        direction = rng.choice([1, 1, 1, -1])  # 75% 롱
        positions[ticker] = round(alloc * direction, 4)

    # 정규화
    total = sum(abs(w) for w in positions.values())
    if total > 1.0:
        positions = {t: round(w / total, 4) for t, w in positions.items()}

    # Gate trace (mock)
    gate_trace = [
        {"gate": i, "name": f"gate{i}", "flags": [], "feedback_required": False,
         "positions_after": dict(positions)}
        for i in range(1, 6)
    ]

    # 저장
    save_gate_trace(run_id, gate_trace)
    save_positions(run_id, positions, positions)

    return {
        "positions_final": positions,
        "gate_trace": gate_trace,
        "as_of": as_of,
    }


def _compute_pnl(
    positions_prev: dict[str, float],
    positions_curr: dict[str, float],
    price_returns: dict[str, float],  # {ticker -> period return}
) -> tuple[float, float]:
    """Gross PnL과 Turnover 계산."""
    gross_ret = sum(
        positions_prev.get(t, 0.0) * r
        for t, r in price_returns.items()
    )
    turnover = sum(
        abs(positions_curr.get(t, 0.0) - positions_prev.get(t, 0.0))
        for t in set(positions_curr) | set(positions_prev)
    ) / 2.0
    return gross_ret, turnover


def run(
    start_date: str,
    end_date: str,
    universe: list[str],
    rebalance_freq: str = "weekly",
    initial_capital: float = 1_000_000.0,
    mode: str = "mock",
    seed: int = 42,
) -> tuple[list[dict], dict]:
    """
    백테스트 실행.

    Returns:
        (results_rows, summary_dict)
    """
    run_id = f"backtest_{generate_run_id()[:8]}"
    config = {
        "start_date": start_date,
        "end_date": end_date,
        "universe": universe,
        "rebalance_freq": rebalance_freq,
        "initial_capital": initial_capital,
        "mode": mode,
        "seed": seed,
        "cost_bps": COST_BPS,
        "slippage_bps": SLIPPAGE_BPS,
    }
    config_hash = save_config_snapshot(run_id, config)
    print(f"[Backtest] run_id={run_id}  config_hash={config_hash}")

    rebalance_dates = _get_rebalance_dates(start_date, end_date, rebalance_freq)
    # Mock price return 생성기 (결정론적)
    price_rng = random.Random(seed + 1)

    results: list[dict] = []
    positions_prev: dict[str, float] = {}
    portfolio_value = initial_capital
    peak_value = initial_capital
    max_dd = 0.0
    cum_gross = 0.0

    for i, date in enumerate(rebalance_dates):
        as_of = f"{date}T00:00:00+00:00"

        # 파이프라인 실행
        pipeline_result = _run_pipeline_for_date(as_of, universe, mode, seed, run_id)
        positions_curr = pipeline_result["positions_final"]

        # Mock 기간 수익률 (결정론적)
        period_returns = {t: price_rng.normalvariate(0.001, 0.015) for t in universe}

        # PnL 계산
        gross_ret, turnover = _compute_pnl(positions_prev, positions_curr, period_returns)
        cost = turnover * TOTAL_COST_PER_TURN
        net_ret = gross_ret - cost

        portfolio_value *= (1 + net_ret)
        peak_value = max(peak_value, portfolio_value)
        drawdown = (peak_value - portfolio_value) / peak_value
        max_dd = max(max_dd, drawdown)
        cum_gross += gross_ret

        row = {
            "date": date,
            "gross_ret": round(gross_ret, 6),
            "net_ret": round(net_ret, 6),
            "turnover": round(turnover, 4),
            "drawdown": round(drawdown, 4),
            "portfolio_value": round(portfolio_value, 2),
            "cost": round(cost, 6),
        }
        for t in universe:
            row[f"weight_{t}"] = positions_curr.get(t, 0.0)

        results.append(row)
        positions_prev = positions_curr
        print(f"  [{date}] net_ret={net_ret:.4%}  turnover={turnover:.2%}  val={portfolio_value:,.0f}")

    # Summary
    n = len(results)
    if n < 2:
        cagr = 0.0
        sharpe = 0.0
    else:
        total_ret = portfolio_value / initial_capital - 1.0
        years = (datetime.fromisoformat(end_date) - datetime.fromisoformat(start_date)).days / 365.25
        cagr = (1 + total_ret) ** (1 / max(years, 0.01)) - 1
        rets = [r["net_ret"] for r in results]
        avg_r = sum(rets) / n
        std_r = math.sqrt(sum((r - avg_r) ** 2 for r in rets) / n) if n > 1 else 1e-9
        sharpe = (avg_r / max(std_r, 1e-9)) * math.sqrt(52)  # weekly → annualized

    hit_rate = sum(1 for r in results if r["net_ret"] > 0) / max(n, 1)
    avg_turnover = sum(r["turnover"] for r in results) / max(n, 1)

    summary = {
        "run_id": run_id,
        "config_hash": config_hash,
        "start_date": start_date,
        "end_date": end_date,
        "universe": universe,
        "seed": seed,
        "n_periods": n,
        "cagr": round(cagr, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "hit_rate": round(hit_rate, 4),
        "avg_turnover": round(avg_turnover, 4),
        "final_portfolio_value": round(portfolio_value, 2),
    }
    return results, summary


def _save_results(results: list[dict], summary: dict, run_id: str) -> None:
    """결과를 CSV와 JSON으로 저장."""
    from pathlib import Path
    out_dir = Path("runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    if results:
        import csv
        csv_path = out_dir / "backtest_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"  → {csv_path}")

    # JSON summary
    json_path = out_dir / "backtest_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  → {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PIT Backtest Runner")
    parser.add_argument("--start", required=True, help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--universe", nargs="+", required=True, help="티커 목록")
    parser.add_argument("--freq", default="weekly", choices=list(REBALANCE_FREQ_DAYS.keys()))
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    parser.add_argument("--mode", default="mock", choices=["mock", "live"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results, summary = run(
        start_date=args.start,
        end_date=args.end,
        universe=args.universe,
        rebalance_freq=args.freq,
        initial_capital=args.capital,
        mode=args.mode,
        seed=args.seed,
    )
    _save_results(results, summary, summary["run_id"])
    print("\n[Backtest Summary]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
