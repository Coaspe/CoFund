"""
report_agent.py — ⑦ Report Writer Agent (IC Memo Expert)
=========================================================
투자 심의 위원회(IC) 메모 작성 전문가.

핵심 기능:
  A. 데이터 큐레이션: 6개 에이전트 JSON → 정보 계층화된 IC 메모
  B. 시나리오 대응  : APPROVE → Alpha Thesis 중심 / REJECT → 방어적 서사
  C. 가정 기록     : 핵심 가정(assumptions)을 명시해 추후 추적 가능

의존 패키지:
  pip install langchain-openai langgraph pydantic

실행:
  python report_agent.py
"""

from __future__ import annotations

import json
import os
import re
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")

try:
    from langchain_core.messages import SystemMessage, HumanMessage
    HAS_LC = True
except ImportError:
    HAS_LC = False

from schemas.common import InvestmentState
from llm.router import get_llm


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 시스템 프롬프트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REPORT_SYSTEM_PROMPT = """\
당신은 세계 최고의 헤지펀드 투자 심의 위원회(IC) 서기이자 리포트 전문가입니다.
당신의 목표는 파편화된 분석 데이터를 하나의 설득력 있는 투자 서사(Narrative)로 엮어내는 것입니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A. 표준 투자 메모 목차 (Standard IC Memo Structure)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. **Executive Summary:** 결론(Buy/Hold/Sell), 최종 비중, 한 문장 핵심 논거.
2. **Alpha Thesis:** 퀀트의 엣지(Z-score, p-value)와 매크로/펀더멘털의 긍정적 요인 결합.
3. **Risk & Mitigation:** 리스크 매니저가 지적한 경고 사항 및 이를 어떻게 통제(Scaling/Hedge)했는지 설명.
4. **Sizing & Execution:** 켈리 공식 및 CVaR 한도 기반의 최종 사이징 근거.
5. **Key Assumptions & Review Triggers:** 이번 결정의 핵심 가정들과 재검토 조건.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
B. 시나리오별 뉘앙스 처리
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- [APPROVE 시]: 투자의 당위성과 기대 수익률, 그리고 리스크 대비 엣지의 통계적 우위를 강조하라.
- [FALLBACK/REJECT 시]: '지금은 하지 않는 이유'를 최상단에 명시하라. 리스크 매니저의 반려 사유를 \
방어적으로 설명하고, 어떤 조건(Trigger)이 충족되면 다시 검토할지 가이드라인을 제시하라.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
C. 포맷팅 가이드라인
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- 표(Table): Ticker, Action, Weight, Z-score, 1d-CVaR, Regime 등 핵심 수치는 반드시 표로 정리하라.
- 강조: 중요한 수치나 결론은 **볼드체** 또는 인용구(Blockquote)를 활용하라.
- 간결성: 전문적인 톤을 유지하되 불필요한 수식어는 배제하고 데이터 중심의 문장을 작성하라.

출력은 반드시 완성된 마크다운 보고서 문자열만 반환하라. JSON이 아닌 순수 마크다운이다."""


def _build_report_human_msg(state: dict) -> str:
    """InvestmentState 전체를 요약하여 LLM Human 메시지로 조립."""

    def _desk_narrative(desk_key: str) -> str:
        desk = state.get(desk_key, {})
        lines = []
        kd = desk.get("key_drivers") or []
        if kd:
            lines.append("  Key Drivers: " + " | ".join(kd))
        ww = desk.get("what_to_watch") or []
        if ww:
            lines.append("  What to watch:\n" + "\n".join(f"    - {w}" for w in ww))
        sn = desk.get("scenario_notes") or {}
        if sn:
            lines.append(f"  Scenarios:\n    Bull: {sn.get('bull','')}\n    Base: {sn.get('base','')}\n    Bear: {sn.get('bear','')}")
        dq = desk.get("data_quality") or {}
        missing = dq.get("missing_fields") or []
        if missing:
            lines.append(f"  Data gaps: {', '.join(str(m) for m in missing)}")
        limitations = desk.get("limitations") or []
        if limitations:
            lines.append(f"  Limitations: {'; '.join(limitations[:2])}")
        return "\n".join(lines) if lines else "  (데이터 없음)"

    def _evidence_section(state: dict) -> str:
        """evidence_store를 요약 렌더. 링크/제목/날짜만."""
        ev_store = state.get("evidence_store", {})
        ev_score = state.get("evidence_score", 0)
        if not ev_store:
            return f"[Evidence] score={ev_score} (수집된 증거 없음)"
        lines = [f"[Evidence Sources] (score={ev_score}, items={len(ev_store)})"]
        for _hash, item in list(ev_store.items())[:10]:
            url = item.get("url", "")
            title = item.get("title", "(제목 없음)")
            pub = item.get("published_at", "")
            tier = item.get("trust_tier", 0)
            lines.append(f"  - [{title}]({url}) | {pub} | trust={tier}")
        return "\n".join(lines)

    parts = [
        f"[작성 시각] {datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"[사용자 요청] {state.get('user_request', '')}",
        f"[종목] {state.get('target_ticker', 'N/A')}",
        f"[반복 횟수] {state.get('iteration_count', 0)}",
        "",
        "[Orchestrator 지시]",
        f"```json\n{json.dumps(state.get('orchestrator_directives', {}), ensure_ascii=False, indent=2)}\n```",
        "",
        "[Macro 분석]",
        f"```json\n{json.dumps(state.get('macro_analysis', {}), ensure_ascii=False, indent=2)}\n```",
        "",
        "[Fundamental 분석]",
        f"```json\n{json.dumps(state.get('fundamental_analysis', {}), ensure_ascii=False, indent=2)}\n```",
        "",
        "[Sentiment 분석]",
        f"```json\n{json.dumps(state.get('sentiment_analysis', {}), ensure_ascii=False, indent=2)}\n```",
        "",
        "[Quant 분석]",
        f"```json\n{json.dumps(state.get('technical_analysis', {}), ensure_ascii=False, indent=2)}\n```",
        "",
        "[Macro 핵심 내용]",
        _desk_narrative("macro_analysis"),
        "",
        "[Macro Market Inputs]",
        _build_macro_market_inputs_section(state.get("macro_analysis", {}), lang="ko"),
        "",
        "[Macro Translation]",
        _build_macro_translation_section(state.get("macro_analysis", {}), lang="ko"),
        "",
        "[Fundamental 핵심 내용]",
        _desk_narrative("fundamental_analysis"),
        "",
        "[Sentiment 핵심 내용]",
        _desk_narrative("sentiment_analysis"),
        "",
        "[Risk Assessment]",
        f"```json\n{json.dumps(state.get('risk_assessment', {}), ensure_ascii=False, indent=2)}\n```",
        "",
        _evidence_section(state),
    ]
    if state.get("user_action_required"):
        parts.extend([
            "",
            "[User Action Required]",
            f"```json\n{json.dumps({'research_stop_reason': state.get('research_stop_reason', ''), 'user_action_items': state.get('user_action_items', [])}, ensure_ascii=False, indent=2)}\n```",
        ])
    return "\n".join(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM 호출 (실제 / Mock)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _call_llm(state: dict) -> str:
    """LLM 호출 → 마크다운 보고서 문자열. API 키 없으면 Mock."""
    llm = get_llm("report_writer")
    if llm is None or not HAS_LC:
        print("   [LLM] API 키 없음 → 규칙 기반 Mock 보고서 생성")
        return _mock_generate_report(state)

    try:
        msgs = [
            SystemMessage(content=REPORT_SYSTEM_PROMPT),
            HumanMessage(content=_build_report_human_msg(state)),
        ]
        resp = llm.invoke(msgs)
        return resp.content
    except Exception as exc:
        print(f"   [LLM] ⚠️ 호출 실패 → Mock: {exc}")
        return _mock_generate_report(state)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 규칙 기반 Mock 보고서 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _determine_scenario(state: dict) -> str:
    """APPROVE / REJECT 시나리오 판별."""
    risk = state.get("risk_assessment", {})
    grade = risk.get("grade", "")
    # risk_agent.py 형식: risk_decision 안에 orchestrator_feedback
    risk_decision = risk.get("risk_decision", risk)
    fb = risk_decision.get("orchestrator_feedback", {})
    fb_required = fb.get("required", False)

    # orchestrator가 fallback_abort를 내렸는지 확인
    orch = state.get("orchestrator_directives", {})
    action_type = orch.get("action_type", "")

    if action_type == "fallback_abort":
        return "REJECT"
    if grade == "High" and fb_required:
        return "REJECT"
    return "APPROVE"


def _safe_get(d: dict, *keys: str, default: Any = "N/A") -> Any:
    """중첩 dict 안전 접근."""
    cur = d
    for k in keys:
        if isinstance(cur, dict):
            cur = cur.get(k, default)
        else:
            return default
    return cur


def _metric_as_of_map(evidence: list[dict] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in evidence or []:
        if not isinstance(item, dict):
            continue
        metric = str(item.get("metric", "") or "").strip()
        ts = str(item.get("as_of", "") or item.get("published_at", "")).strip()
        if metric and ts and metric not in out:
            out[metric] = ts
    return out


def _macro_metric_value(macro: dict, key: str) -> Any:
    if not isinstance(macro, dict):
        return None
    if key in macro and macro.get(key) is not None:
        return macro.get(key)
    indicators = macro.get("indicators", {})
    if isinstance(indicators, dict):
        return indicators.get(key)
    return None


def _build_macro_market_inputs_section(macro: dict, *, lang: str = "ko") -> str:
    evidence_map = _metric_as_of_map(macro.get("evidence", []))
    metric_rows = [
        ("2Y Treasury", "dgs2"),
        ("10Y Treasury", "dgs10"),
        ("Fed Funds", "fed_funds_rate"),
        ("SOFR", "sofr_rate"),
        ("Real 10Y Yield", "real_10y_yield"),
        ("Dollar Index", "dollar_index"),
        ("VIX (FRED)", "vix_level"),
        ("WTI Spot", "wti_spot"),
        ("Brent Spot", "brent_spot"),
        ("WTI Front", "wti_front_month"),
        ("Brent Front", "brent_front_month"),
        ("Fed Funds Fut Front", "fed_funds_futures_front_implied_rate"),
        ("Fed Funds Fut 3M", "fed_funds_futures_3m_implied_rate"),
        ("Fed Funds Fut 6M", "fed_funds_futures_6m_implied_rate"),
        ("Fed Funds Fut 6M Change(bp)", "fed_funds_futures_implied_change_6m_bp"),
        ("SOFR Fut Front", "sofr_futures_front_implied_rate"),
        ("SOFR Fut 3M", "sofr_futures_3m_implied_rate"),
        ("SOFR Fut 6M", "sofr_futures_6m_implied_rate"),
        ("SOFR Fut 6M Change(bp)", "sofr_futures_implied_change_6m_bp"),
        ("SOFR-FF 6M Basis(bp)", "sofr_ff_6m_basis_bp"),
        ("2Y-FFR Proxy(bp)", "cuts_priced_proxy_2y_ffr_bp"),
    ]
    rows = []
    for label, key in metric_rows:
        value = _macro_metric_value(macro, key)
        if value is None:
            continue
        rows.append((label, value, evidence_map.get(key, "")))
    if not rows:
        return ""
    if lang == "en":
        lines = ["", "## Macro Market Inputs", "", "| Metric | Value | As Of |", "|---|---:|---|"]
    else:
        lines = ["", "## Macro Market Inputs", "", "| Metric | Value | As Of |", "|---|---:|---|"]
    for label, value, ts in rows:
        lines.append(f"| {label} | {value} | {ts or 'n/a'} |")
    return "\n".join(lines) + "\n"


def _build_macro_translation_section(macro: dict, *, lang: str = "ko") -> str:
    transmission = macro.get("transmission_map", {}) if isinstance(macro, dict) else {}
    portfolio = macro.get("portfolio_implications", {}) if isinstance(macro, dict) else {}
    triggers = macro.get("monitoring_triggers", []) if isinstance(macro, dict) else []
    lines: list[str] = ["", "## Macro Translation"]
    if isinstance(transmission, dict) and transmission:
        lines.append("- Transmission map:")
        for key in ("growth_beta", "policy_rates", "rates_pricing", "credit", "inflation_real_assets", "usd", "volatility", "commodity_shock_watch"):
            item = transmission.get(key)
            if not isinstance(item, dict):
                continue
            lines.append(
                f"  - {key}: signal={item.get('signal')}, value={item.get('current_value')}, state={item.get('current_state')}"
            )
    if isinstance(portfolio, dict) and portfolio.get("targets"):
        lines.append("- Portfolio implications:")
        for row in portfolio.get("targets", [])[:6]:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"  - {row.get('ticker')}: {row.get('stance')} / score={row.get('macro_fit_score')} / bucket={row.get('bucket')}"
            )
    if isinstance(triggers, list) and triggers:
        lines.append("- Monitoring triggers:")
        for row in triggers[:5]:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"  - {row.get('name')}: {row.get('metric')} {row.get('trigger')} (current={row.get('current_value')})"
            )
    return "\n".join(lines) + "\n" if len(lines) > 1 else ""


def _mock_generate_report(state: dict) -> str:
    """시나리오에 따라 IC 메모 마크다운을 생성하는 규칙 기반 Mock."""
    scenario = _determine_scenario(state)
    ticker = state.get("target_ticker", "N/A")
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    iteration = state.get("iteration_count", 0)

    macro = state.get("macro_analysis", {})
    funda = state.get("fundamental_analysis", {})
    senti = state.get("sentiment_analysis", {})
    quant = state.get("technical_analysis", {})
    risk = state.get("risk_assessment", {})
    risk_decision = risk.get("risk_decision", risk)
    orch = state.get("orchestrator_directives", {})

    # ── 공통 수치 추출 ────────────────────────────────────────────────────
    regime = macro.get("regime", macro.get("sector_cycle", "N/A"))
    gdp = macro.get("gdp_growth", "N/A")
    rate = macro.get("interest_rate", "N/A")

    rev_growth = funda.get("revenue_growth", "N/A")
    roe = funda.get("roe", "N/A")
    debt_equity = funda.get("debt_to_equity", funda.get("debt_equity", "N/A"))
    pe = funda.get("pe_ratio", "N/A")

    sent_score = senti.get("sentiment_score", "N/A")
    sent_label = senti.get("overall_sentiment", "N/A")

    quant_decision = quant.get("decision", quant.get("macd_signal", "N/A"))
    z_score = quant.get("z_score", quant.get("momentum_score", "N/A"))
    alloc_pct = quant.get("final_allocation_pct", quant.get("final_weight", "N/A"))
    cvar_asset = quant.get("asset_cvar_99_daily", "N/A")

    # Risk 수치
    per_ticker = risk_decision.get("per_ticker_decisions", {})
    ticker_dec = per_ticker.get(ticker, {})
    final_weight = ticker_dec.get("final_weight", alloc_pct)
    decision_label = ticker_dec.get("decision", "N/A")
    risk_flags = ticker_dec.get("flags", risk.get("flags", []))
    risk_rationale = ticker_dec.get("rationale_short", risk.get("recommendation", ""))

    port_actions = risk_decision.get("portfolio_actions", {})
    hedges = port_actions.get("hedge_recommendations", [])
    gna = port_actions.get("gross_net_adjustment", {})

    fb = risk_decision.get("orchestrator_feedback", {})
    fb_reasons = fb.get("reasons", [])
    fb_detail = fb.get("detail", "")

    port_summary = risk.get("risk_payload", {}).get("portfolio_risk_summary", {})
    cvar_port = port_summary.get("portfolio_cvar_1d", "N/A")
    leverage = port_summary.get("leverage_ratio", "N/A")
    stress_loss = port_summary.get("stress_test_loss_pct", "N/A")
    hhi = port_summary.get("herfindahl_index", "N/A")

    # ── APPROVE 시나리오 ──────────────────────────────────────────────────
    if scenario == "APPROVE":
        action_word = "**BUY**" if quant_decision in ("LONG", "bullish") else "**HOLD**"
        report = f"""\
# 📊 Investment Committee Memo — {ticker}

> **Date:** {ts} | **Prepared by:** AI IC Report Writer | **Iterations:** {iteration}

---

## 1. Executive Summary

| Item | Value |
|------|-------|
| **Verdict** | {action_word} |
| **Final Weight** | **{_fmt_pct(final_weight)}** |
| **Risk Grade** | {risk.get('grade', 'N/A')} |
| **Macro Regime** | {regime} |

> **핵심 논거:** {ticker}는 퀀트 모델의 매수 신호(Z-score: {z_score})와 견고한 펀더멘털(매출 성장 {rev_growth}%, ROE {roe}%)을 기반으로, \
현행 리스크 한도 내에서 **{_fmt_pct(final_weight)}** 비중의 롱 포지션이 승인됨.

---

## 2. Alpha Thesis

### 퀀트 엣지
- **Quant Signal:** {quant_decision} | Z-score: **{z_score}**
- **Asset CVaR (99%, 1d):** {_fmt_pct(cvar_asset)}
- **추천 비중:** {_fmt_pct(alloc_pct)}

### 매크로 환경
- **GDP 성장률:** {gdp}% | **기준금리:** {rate}%
- **레짐:** {regime}

### 펀더멘털
| 지표 | 값 |
|------|-----|
| 매출 성장률 | {rev_growth}% |
| ROE | {roe}% |
| 부채비율 | {debt_equity} |
| PER | {pe} |

### 센티먼트
- **Overall:** {sent_label} (Score: {sent_score})

---

## 3. Risk & Mitigation

| Risk Flag | 설명 |
|-----------|------|
{_fmt_flags_table(risk_flags)}

{f'> **리스크 매니저 판단:** {risk_rationale}' if risk_rationale else ''}

{_fmt_hedges(hedges)}

{_fmt_gna(gna)}

---

## 4. Sizing & Execution

| 항목 | 값 |
|------|-----|
| Quant 추천 비중 | {_fmt_pct(alloc_pct)} |
| Risk-adjusted 최종 비중 | **{_fmt_pct(final_weight)}** |
| Portfolio CVaR (1d, 99%) | {_fmt_pct(cvar_port)} |
| Leverage | {leverage} |
| Stress Test Loss | {_fmt_pct(stress_loss)} |
| HHI (집중도) | {hhi} |

---

## 5. Key Assumptions & Review Triggers

1. **매크로 레짐** — 현재 '{regime}' 레짐이 유지된다고 가정. 레짐이 'recession'/'crisis'로 전환 시 즉시 재검토.
2. **펀더멘털** — 다음 분기 실적 발표까지 현재 매출 성장률({rev_growth}%)이 유지된다고 가정.
3. **유동성** — 일평균 거래대금이 충분하여 포지션 진입/청산에 슬리피지가 미미하다고 가정.
4. **재검토 트리거** — Z-score 반전(0 이상 → 0 이하), CVaR 한도 초과, 또는 구조적 리스크 플래그 발생 시.

---

{_build_macro_market_inputs_section(macro, lang="ko")}
{_build_macro_translation_section(macro, lang="ko")}

*본 보고서는 AI 투자 분석 시스템에 의해 자동 생성되었습니다. 최종 투자 결정은 반드시 인간 심의위원의 승인이 필요합니다.*

{_fmt_research_appendix(state)}"""

    # ── REJECT / FALLBACK 시나리오 ────────────────────────────────────────
    else:
        orch_rationale = _safe_get(orch, "investment_brief", "rationale", default="")
        orch_universe = _safe_get(orch, "investment_brief", "target_universe", default=[])
        action_type = orch.get("action_type", "reject")

        report = f"""\
# 🚫 Investment Committee Memo — {ticker} (REJECTED)

> **Date:** {ts} | **Prepared by:** AI IC Report Writer | **Iterations:** {iteration}

---

## 1. Executive Summary — 투자 불가 결정

> ⚠️ **결론: {ticker} 신규 매수 불가 (NO TRADE)**
>
> 리스크 위원회가 {iteration}회 반복 심의 후에도 리스크 한도 내 진입을 승인하지 못함. \
> 펀드 자본 보전을 최우선으로 신규 매수를 보류하고 방어 자산으로 전환을 권고.

| Item | Value |
|------|-------|
| **Verdict** | **NO TRADE** |
| **Action** | {action_type} |
| **Risk Grade** | {risk.get('grade', 'High')} |
| **Macro Regime** | {regime} |
| **대안 유니버스** | {', '.join(orch_universe) if orch_universe else 'CASH'} |

---

## 2. 반려 사유 (Rejection Rationale)

### Risk Manager 피드백
{f'> {fb_detail}' if fb_detail else '> 리스크 한도 초과로 인한 반려.'}

| 반려 사유 | 상세 |
|-----------|------|
{_fmt_reasons_table(fb_reasons)}

### 리스크 수치 요약

| 항목 | 현재값 | 한도 | 위반 |
|------|--------|------|------|
| Portfolio CVaR (1d) | {_fmt_pct(cvar_port)} | 1.50% | {'⚠️ 초과' if _is_over(cvar_port, 0.015) else '✅ 정상'} |
| Stress Test Loss | {_fmt_pct(stress_loss)} | 10.00% | {'⚠️ 초과' if _is_over(stress_loss, 0.10) else '✅ 정상'} |
| Leverage | {leverage} | 2.00 | {'⚠️ 초과' if _is_over(leverage, 2.0) else '✅ 정상'} |
| HHI (집중도) | {hhi} | 0.25 | {'⚠️ 초과' if _is_over(hhi, 0.25) else '✅ 정상'} |

### 각 종목 최종 결정

| Ticker | Decision | Final Weight | Flags |
|--------|----------|--------------|-------|
{_fmt_per_ticker_table(per_ticker)}

---

## 3. Orchestrator 에스컬레이션 이력

| 반복 | 조치 | 설명 |
|------|------|------|
| 1 | Scale / Hedge | 비중 축소 또는 숏 헷지 추가 시도 |
| 2 | Pivot | 방어 섹터로 전환 시도 |
| 3+ | **Fallback Abort** | 신규 매수 포기, 현금 관망 |

> **CIO 판단:** {orch_rationale if orch_rationale else '리스크 한도 내 진입 불가로 매수 포기.'}

---

## 4. 재검토 조건 (Review Triggers)

아래 조건 **하나 이상** 충족 시 {ticker} 투자를 재검토할 수 있음:

1. **매크로 레짐 전환** — '{regime}'에서 'expansion' 또는 'recovery'로 전환 시.
2. **CVaR 정상화** — 포트폴리오 CVaR이 한도(1.5%) 이하로 하락 시.
3. **스트레스 테스트 통과** — 과거 위기 시나리오 예상 손실이 10% 이하로 개선 시.
4. **펀더멘털 개선** — 구조적 리스크 플래그(부도 위험, 회계 이슈, 규제 제재) 해소 시.
5. **센티먼트 반전** — 부정적 센티먼트({sent_label}, {sent_score})가 중립 이상으로 전환 시.

---

## 5. 권고 대안

| 자산 | 유형 | 근거 |
|------|------|------|
| CASH | 현금 | 자본 보전, 기회 대기 |
| SHY | 단기 국채 ETF | 금리 수익 + 극저 변동성 |
| TLT | 장기 국채 ETF | Duration 방어 (금리 인하 기대 시) |
| XLU | 유틸리티 섹터 ETF | 방어 섹터, 배당 수익 |

---

{_build_macro_market_inputs_section(macro, lang="ko")}
{_build_macro_translation_section(macro, lang="ko")}

*본 보고서는 AI 투자 분석 시스템에 의해 자동 생성되었습니다. 최종 투자 결정은 반드시 인간 심의위원의 승인이 필요합니다.*

{_fmt_research_appendix(state)}"""

    return report


# ── 포맷팅 헬퍼 ───────────────────────────────────────────────────────────

def _fmt_pct(val: Any) -> str:
    """숫자를 % 문자열로 변환. N/A면 그대로."""
    if val is None or val == "N/A":
        return "N/A"
    try:
        return f"{float(val) * 100:.2f}%" if float(val) < 1.0 else f"{float(val):.2f}%"
    except (TypeError, ValueError):
        return str(val)


def _is_over(val: Any, threshold: float) -> bool:
    """수치 비교. N/A → False."""
    if val is None or val == "N/A":
        return False
    try:
        return float(val) > threshold
    except (TypeError, ValueError):
        return False


def _fmt_flags_table(flags: list) -> str:
    if not flags:
        return "| (없음) | 모든 리스크 게이트 통과 |"
    return "\n".join(f"| `{f}` | — |" for f in flags)


def _fmt_reasons_table(reasons: list) -> str:
    if not reasons:
        return "| (없음) | — |"
    return "\n".join(f"| `{r}` | — |" for r in reasons)


def _fmt_hedges(hedges: list) -> str:
    if not hedges:
        return ""
    lines = ["### 헷지 권고", "", "| Type | Direction | Notional | Reason |", "|------|-----------|----------|--------|"]
    for h in hedges:
        lines.append(
            f"| {h.get('type', '')} | {h.get('direction', '')} | "
            f"{_fmt_pct(h.get('notional_suggestion', ''))} | {h.get('reason', '')} |"
        )
    return "\n".join(lines)


def _fmt_gna(gna: Optional[dict]) -> str:
    if not gna:
        return ""
    return (
        "### Gross/Net Exposure 조정\n\n"
        f"| 항목 | 목표 |\n|------|------|\n"
        f"| Gross Exposure | {gna.get('target_gross_exposure', 'N/A')} |\n"
        f"| Net Exposure | {gna.get('target_net_exposure', 'N/A')} |\n"
        f"| 사유 | {gna.get('reason', '')} |"
    )


def _fmt_per_ticker_table(per_ticker: dict) -> str:
    if not per_ticker:
        return "| — | — | — | — |"
    lines = []
    for t, d in per_ticker.items():
        flags_str = ", ".join(d.get("flags", [])) or "—"
        lines.append(
            f"| {t} | {d.get('decision', 'N/A')} | "
            f"{_fmt_pct(d.get('final_weight', 'N/A'))} | {flags_str} |"
        )
    return "\n".join(lines)


def _fmt_research_appendix(state: dict) -> str:
    ev_score = state.get("evidence_score", 0)
    ev_store = state.get("evidence_store", {}) or {}

    lines = [
        "## 6. Research Evidence & Watchlist",
        "",
        f"- **Evidence Score:** {ev_score}",
        f"- **Evidence Items:** {len(ev_store)}",
        "",
        "### Evidence Sources (link/title/date)",
    ]
    if ev_store:
        for item in list(ev_store.values())[:10]:
            lines.append(
                f"- [{item.get('title','(제목 없음)')}]({item.get('url','')}) | {item.get('published_at','')}"
            )
    else:
        lines.append("- (none)")

    lines.extend(["", "### Desk Drivers / Watch / Data Quality"])
    for desk_key, label in (
        ("macro_analysis", "Macro"),
        ("fundamental_analysis", "Fundamental"),
        ("sentiment_analysis", "Sentiment"),
    ):
        desk = state.get(desk_key, {}) or {}
        lines.append(f"- **{label} Key Drivers:** " + " | ".join(desk.get("key_drivers", [])[:3] or ["(none)"]))
        lines.append(f"- **{label} What to Watch:** " + " | ".join(desk.get("what_to_watch", [])[:3] or ["(none)"]))
        sn = desk.get("scenario_notes", {}) or {}
        lines.append(
            f"- **{label} Scenario Notes:** bull={sn.get('bull','')} / base={sn.get('base','')} / bear={sn.get('bear','')}"
        )
        dq = desk.get("data_quality", {}) or {}
        lines.append(
            f"- **{label} Data Quality:** missing_pct={dq.get('missing_pct', 'N/A')}, "
            f"freshness_days={dq.get('freshness_days', 'N/A')}, warnings={dq.get('warnings', [])}"
        )
    if state.get("user_action_required"):
        lines.extend(["", "### User Action Required"])
        lines.append(f"- **Research Stop Reason:** {state.get('research_stop_reason', '')}")
        for item in (state.get("user_action_items", []) or [])[:5]:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- [{item.get('code','unknown')}] {item.get('suggested_action','수동 점검 필요')} "
                f"(desk={item.get('desk','')}, detail={item.get('detail','')})"
            )
    return "\n".join(lines)


def _output_language(state: dict) -> str:
    lang = str(state.get("output_language", "")).strip().lower()
    if lang in {"ko", "en"}:
        return lang
    text = str(state.get("user_request", ""))
    return "ko" if re.search(r"[가-힣]", text) else "en"


def _has_cvar_breach_flag(flags: list[Any]) -> bool:
    for f in flags or []:
        s = str(f).strip().lower()
        if "cvar" in s and ("breach" in s or "limit" in s or "exceed" in s):
            return True
    return False


def _build_fidelity_report(state: dict) -> str:
    lang = _output_language(state)
    ticker = str(state.get("target_ticker", "N/A"))
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    iteration = int(state.get("iteration_count", 0))
    intent = str(state.get("intent", ""))
    analysis_mode = str(state.get("analysis_execution_mode", "")).strip()
    universe = [str(t).strip().upper() for t in (state.get("universe", []) or []) if str(t).strip()]
    if not universe:
        universe = list((_safe_get(state.get("orchestrator_directives", {}), "investment_brief", "target_universe", default=[])) or [])
    hedge_lite = state.get("hedge_lite", {}) if isinstance(state.get("hedge_lite"), dict) else {}
    hedge_rows = hedge_lite.get("hedges", {}) if isinstance(hedge_lite.get("hedges"), dict) else {}
    analyzed = [ticker] if ticker and ticker != "N/A" else []
    if str(analysis_mode).startswith("B_") and hedge_rows:
        analyzed.extend([str(t).strip().upper() for t in hedge_rows.keys()])
    analyzed = [t for t in dict.fromkeys(analyzed) if t]
    unanalyzed = [t for t in universe if t not in set(analyzed)]

    macro = state.get("macro_analysis", {}) or {}
    funda = state.get("fundamental_analysis", {}) or {}
    senti = state.get("sentiment_analysis", {}) or {}
    quant = state.get("technical_analysis", {}) or {}
    risk = state.get("risk_assessment", {}) or {}
    risk_decision = risk.get("risk_decision", risk) if isinstance(risk, dict) else {}
    per_ticker = risk_decision.get("per_ticker_decisions", {}) if isinstance(risk_decision, dict) else {}
    ticker_dec = per_ticker.get(ticker, {}) if isinstance(per_ticker, dict) else {}

    quant_decision = str(quant.get("decision", quant.get("llm_decision", {}).get("decision", "HOLD")))
    quant_alloc = quant.get("final_allocation_pct", quant.get("llm_decision", {}).get("final_allocation_pct", 0.0))
    quant_reason = str(quant.get("llm_decision", {}).get("cot_reasoning", "")).strip()

    risk_flags = list(ticker_dec.get("flags", [])) if isinstance(ticker_dec, dict) else []
    risk_grade = str(risk.get("grade", "N/A"))
    risk_decision_label = str(ticker_dec.get("decision", "n/a"))
    risk_weight = ticker_dec.get("final_weight", quant_alloc)
    risk_rationale = str(ticker_dec.get("rationale_short", "")).strip()
    positions_final = state.get("positions_final", {}) if isinstance(state.get("positions_final"), dict) else {}
    positions_proposed = state.get("positions_proposed", {}) if isinstance(state.get("positions_proposed"), dict) else {}

    cvar_limit = (
        ((risk.get("risk_payload", {}) or {}).get("risk_limits", {}) or {}).get("max_portfolio_cvar_1d")
        or ((quant.get("quant_payload", {}) or {}).get("portfolio_risk_parameters", {}) or {}).get("max_portfolio_cvar_limit")
    )
    portfolio_cvar = ((risk.get("risk_payload", {}) or {}).get("portfolio_risk_summary", {}) or {}).get("portfolio_cvar_1d")
    cvar_breach = _has_cvar_breach_flag(risk_flags)
    cvar_line = (
        f"- 포트폴리오 CVaR 한도 이슈: **있음** (flags={risk_flags})"
        if cvar_breach
        else "- 포트폴리오 CVaR 한도 이슈: **없음** (risk flags 기준)"
    )

    mode_note = ""
    if unanalyzed:
        mode_note = (
            "\n- 분석 미실행 유니버스: " + ", ".join(unanalyzed) +
            " (state 근거 없음, 가정으로만 취급)"
        )

    def _build_hedge_lite_summary_ko() -> str:
        if not str(analysis_mode).startswith("B_"):
            return ""
        rows = hedge_rows if isinstance(hedge_rows, dict) else {}
        if not rows:
            return "\n## Hedge Lite Summary\n- 헤지 후보 lite 분석 미실행(가정)\n"

        lines = ["", "## Hedge Lite Summary"]
        lines.append(
            f"- 빌드 상태: {hedge_lite.get('status', 'unknown')} / reason={hedge_lite.get('reason', '')} / seed={hedge_lite.get('seed', 'n/a')}"
        )
        selected = hedge_lite.get("selected_hedges", []) or []
        lines.append(f"- 선택 헤지: {selected if selected else '없음'}")
        for ht in [t for t in universe if t != ticker]:
            row = rows.get(ht)
            if not isinstance(row, dict):
                lines.append(f"- {ht}: 분석 미실행(가정)")
                continue
            status = str(row.get("status", "unknown"))
            w = positions_final.get(ht, positions_proposed.get(ht, 0.0))
            if status != "ok":
                lines.append(f"- {ht}: 헤지 후보 lite 분석 실패(데이터 부족), weight={_fmt_pct(w)}")
                continue
            lines.append(
                f"- {ht}: score={row.get('score')}, corr_main60={row.get('corr_with_main_60d')}, "
                f"vol_shift={row.get('vol_shift_20d_vs_60d')}, ret5d={row.get('ret_5d')}, weight={_fmt_pct(w)}"
            )
        return "\n".join(lines) + "\n"

    def _build_hedge_lite_summary_en() -> str:
        if not str(analysis_mode).startswith("B_"):
            return ""
        rows = hedge_rows if isinstance(hedge_rows, dict) else {}
        if not rows:
            return "\n## Hedge Lite Summary\n- Hedge lite analysis not executed (assumption only).\n"

        lines = ["", "## Hedge Lite Summary"]
        lines.append(
            f"- Build status: {hedge_lite.get('status', 'unknown')} / reason={hedge_lite.get('reason', '')} / seed={hedge_lite.get('seed', 'n/a')}"
        )
        selected = hedge_lite.get("selected_hedges", []) or []
        lines.append(f"- Selected hedges: {selected if selected else 'none'}")
        for ht in [t for t in universe if t != ticker]:
            row = rows.get(ht)
            if not isinstance(row, dict):
                lines.append(f"- {ht}: analysis not executed (assumption only).")
                continue
            status = str(row.get("status", "unknown"))
            w = positions_final.get(ht, positions_proposed.get(ht, 0.0))
            if status != "ok":
                lines.append(f"- {ht}: hedge lite failed (insufficient data), weight={_fmt_pct(w)}")
                continue
            lines.append(
                f"- {ht}: score={row.get('score')}, corr_main60={row.get('corr_with_main_60d')}, "
                f"vol_shift={row.get('vol_shift_20d_vs_60d')}, ret5d={row.get('ret_5d')}, weight={_fmt_pct(w)}"
            )
        return "\n".join(lines) + "\n"

    if lang == "en":
        return (
            f"# IC Memo ({ticker})\n\n"
            f"- Timestamp: {ts}\n"
            f"- Iteration: {iteration}\n"
            f"- Intent: {intent}\n"
            f"- Universe: {universe}\n"
            f"- Quant decision: {quant_decision} / alloc={quant_alloc}\n"
            f"- Risk decision: {risk_decision_label} / weight={risk_weight} / grade={risk_grade}\n"
            f"- CVaR breach claim allowed: {cvar_breach}\n"
            f"{mode_note}\n\n"
            f"## Evidence-backed reason fields\n"
            f"- quant.llm_decision.cot_reasoning: {quant_reason or '(empty)'}\n"
            f"- risk_decision.per_ticker_decisions[{ticker}].flags: {risk_flags}\n"
            f"- risk_decision.per_ticker_decisions[{ticker}].rationale_short: {risk_rationale or '(empty)'}\n"
            f"- portfolio_cvar_1d(state): {portfolio_cvar} / limit(state): {cvar_limit}\n"
            f"{_build_hedge_lite_summary_en()}"
        )

    return (
        f"# 투자위원회 메모 ({ticker})\n\n"
        f"- 작성시각: {ts}\n"
        f"- 반복횟수: {iteration}\n"
        f"- 의도(intent): {intent or 'n/a'}\n"
        f"- 분석 유니버스(state.universe): {universe}\n"
        f"- Quant 결정(state): {quant_decision} / 비중={_fmt_pct(quant_alloc)}\n"
        f"- Risk 결정(state): {risk_decision_label} / 최종비중={_fmt_pct(risk_weight)} / 등급={risk_grade}\n"
        f"{cvar_line}\n"
        f"- portfolio_cvar_1d(state): {_fmt_pct(portfolio_cvar)} / limit(state): {_fmt_pct(cvar_limit)}"
        f"{mode_note}\n\n"
        f"## 결론 근거(필드 인용)\n"
        f"- `technical_analysis.llm_decision.cot_reasoning`: {quant_reason or '(없음)'}\n"
        f"- `risk_assessment.risk_decision.per_ticker_decisions[{ticker}].flags`: {risk_flags}\n"
        f"- `risk_assessment.risk_decision.per_ticker_decisions[{ticker}].rationale_short`: {risk_rationale or '(없음)'}\n"
        f"- `fundamental_analysis.analysis_mode`: {funda.get('analysis_mode', 'n/a')} / `asset_type`: {funda.get('asset_type', 'n/a')}\n"
        f"- `sentiment_analysis.data_quality.warnings`: {senti.get('data_quality', {}).get('warnings', [])}\n"
        f"- `research_stop_reason`: {state.get('research_stop_reason', '')}\n\n"
        f"{_build_hedge_lite_summary_ko()}"
        f"## 감사 메모\n"
        f"- 본 보고서는 state 값만 사용해 생성되며, state에 없는 인과는 추가하지 않음.\n"
        f"- CVaR 한도 초과 문구는 `cvar_limit_breach` 계열 플래그가 있을 때만 허용.\n"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LangGraph 노드 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def report_writer_node(state: InvestmentState) -> dict:
    """
    ⑦ Report Writer LangGraph 노드.

    실행 흐름:
      1. InvestmentState 전체를 읽는다.
      2. 시나리오(APPROVE / REJECT)를 판별한다.
      3. LLM(또는 Mock)에게 IC 메모 생성을 요청한다.
      4. final_report 필드에 마크다운 보고서를 저장한다.

    Returns:
        {"final_report": str}
    """
    print(f"\n{'=' * 60}")
    print("⑦ REPORT WRITER (IC Memo Expert)")
    print(f"{'=' * 60}")
    print(f"   [입력] 종목: {state.get('target_ticker', 'N/A')}")
    print(f"   [입력] iteration: {state.get('iteration_count', 0)}")

    scenario = _determine_scenario(state)
    print(f"   [시나리오] {scenario}")
    print("   [Report] state-fidelity 모드로 보고서 생성")
    report = _build_fidelity_report(state)

    line_count = report.count("\n") + 1
    print(f"   [결과] IC 메모 생성 완료 ({line_count} lines)")

    return {"final_report": report}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# __main__ — Mock 시뮬레이션
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    SEP = "=" * 65
    DASH = "─" * 65

    print(SEP)
    print("⑦ REPORT WRITER — Mock 시뮬레이션")
    print(SEP)

    # ══════════════════════════════════════════════════════════════════════
    # 시나리오 1: APPROVE — 모든 지표 양호, 승인
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{DASH}")
    print("📋 시나리오 1: APPROVE — Alpha Thesis 중심 보고서")
    print(DASH)

    state_approve: dict = {
        "user_request": "AAPL 6개월 투자 관점에서 분석해 줘",
        "target_ticker": "AAPL",
        "analysis_tasks": ["macro_analysis", "fundamental_analysis", "sentiment_analysis", "technical_analysis"],
        "iteration_count": 1,
        "orchestrator_directives": {
            "current_iteration": 0,
            "action_type": "initial_delegation",
            "investment_brief": {
                "rationale": "금리 인하 기대감에 따른 테크주 롱 포지션 검토 개시",
                "target_universe": ["AAPL"],
            },
            "desk_tasks": {
                "macro": {"horizon_days": 30, "focus_areas": ["금리 인하 확률"]},
                "fundamental": {"horizon_days": 90, "focus_areas": ["현금흐름 안정성"]},
                "sentiment": {"horizon_days": 5, "focus_areas": ["옵션 스큐"]},
                "quant": {"horizon_days": 10, "risk_budget": "Moderate", "focus_areas": ["Z-score"]},
            },
        },
        "macro_analysis": {
            "regime": "expansion",
            "gdp_growth": 2.8,
            "interest_rate": 4.50,
            "sector_cycle": "expansion",
        },
        "fundamental_analysis": {
            "sector": "Technology",
            "revenue_growth": 12.5,
            "roe": 28.3,
            "debt_to_equity": 1.2,
            "pe_ratio": 25.4,
            "risk_flags": [],
        },
        "sentiment_analysis": {
            "overall_sentiment": "positive",
            "sentiment_score": 0.62,
        },
        "technical_analysis": {
            "decision": "LONG",
            "final_allocation_pct": 0.08,
            "z_score": -1.85,
            "asset_cvar_99_daily": 0.022,
        },
        "risk_assessment": {
            "grade": "Low",
            "risk_decision": {
                "per_ticker_decisions": {
                    "AAPL": {
                        "final_weight": 0.08,
                        "decision": "approve",
                        "flags": [],
                        "rationale_short": "모든 리스크 게이트 통과 — 원안 승인.",
                    }
                },
                "portfolio_actions": {
                    "hedge_recommendations": [],
                    "gross_net_adjustment": None,
                },
                "orchestrator_feedback": {
                    "required": False,
                    "reasons": [],
                    "detail": "모든 Gate 통과.",
                },
            },
            "risk_payload": {
                "portfolio_risk_summary": {
                    "portfolio_cvar_1d": 0.0088,
                    "leverage_ratio": 0.08,
                    "stress_test_loss_pct": 0.045,
                    "herfindahl_index": 1.0,
                },
            },
        },
        "final_report": "",
    }

    result1 = report_writer_node(state_approve)
    print(f"\n{DASH}")
    print("📄 시나리오 1 — IC 메모 (APPROVE)")
    print(DASH)
    print(result1["final_report"])

    # ══════════════════════════════════════════════════════════════════════
    # 시나리오 2: REJECT/FALLBACK — 리스크 한도 초과, 거래 무산
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n\n{DASH}")
    print("📋 시나리오 2: REJECT — 리스크 한도 초과, Fallback")
    print(DASH)

    state_reject: dict = {
        "user_request": "AAPL 지금 사도 돼?",
        "target_ticker": "AAPL",
        "analysis_tasks": ["macro_analysis", "fundamental_analysis", "sentiment_analysis", "technical_analysis"],
        "iteration_count": 3,
        "orchestrator_directives": {
            "current_iteration": 3,
            "action_type": "fallback_abort",
            "investment_brief": {
                "rationale": (
                    "3회 반복에도 리스크 위원회 승인 획득 실패. "
                    "AAPL 신규 매수를 포기하고 현금 관망 또는 방어 섹터 ETF 대안을 권고함."
                ),
                "target_universe": ["CASH", "SHY", "TLT", "XLU"],
            },
            "desk_tasks": {
                "macro": {"horizon_days": 30, "focus_areas": ["방어적 자산배분"]},
                "fundamental": {"horizon_days": 90, "focus_areas": ["방어 섹터 ETF"]},
                "sentiment": {"horizon_days": 7, "focus_areas": ["VIX"]},
                "quant": {"horizon_days": 10, "risk_budget": "Conservative", "focus_areas": ["최소 변동성"]},
            },
        },
        "macro_analysis": {
            "regime": "recession",
            "gdp_growth": -0.5,
            "interest_rate": 5.25,
        },
        "fundamental_analysis": {
            "sector": "Technology",
            "revenue_growth": 5.2,
            "risk_flags": ["regulatory_action"],
            "debt_to_equity": 1.8,
            "pe_ratio": 32.1,
            "roe": 18.5,
        },
        "sentiment_analysis": {
            "overall_sentiment": "negative",
            "sentiment_score": -0.35,
        },
        "technical_analysis": {
            "decision": "LONG",
            "final_allocation_pct": 0.02,
            "z_score": -2.5,
            "asset_cvar_99_daily": 0.028,
        },
        "risk_assessment": {
            "grade": "High",
            "risk_decision": {
                "per_ticker_decisions": {
                    "AAPL": {
                        "final_weight": 0.0,
                        "decision": "reject_local",
                        "flags": ["cvar_limit_breach", "stress_test_violation", "macro_headwind", "regulatory_action"],
                        "rationale_short": "CVaR, 스트레스 테스트, 구조적 리스크로 인해 비중 0 처리.",
                    }
                },
                "portfolio_actions": {
                    "hedge_recommendations": [
                        {
                            "type": "index_hedge",
                            "direction": "short",
                            "notional_suggestion": 0.15,
                            "reason": "CVaR 한도 초과 방어",
                        }
                    ],
                    "gross_net_adjustment": {
                        "target_gross_exposure": 0.8,
                        "target_net_exposure": 0.1,
                        "reason": "스트레스 테스트 손실 한도 방어를 위한 Net Exposure 축소",
                    },
                },
                "orchestrator_feedback": {
                    "required": True,
                    "reasons": ["stress_test_violation", "portfolio_risk_violation", "structural_risk"],
                    "detail": (
                        "포트폴리오 CVaR 0.0230 > 한도 0.0150. "
                        "스트레스 테스트 손실 12.3% > 한도 10.0%. "
                        "구조적 리스크 플래그(regulatory_action) 발견."
                    ),
                },
            },
            "risk_payload": {
                "portfolio_risk_summary": {
                    "portfolio_cvar_1d": 0.0230,
                    "leverage_ratio": 0.12,
                    "stress_test_loss_pct": 0.123,
                    "herfindahl_index": 1.0,
                },
            },
        },
        "final_report": "",
    }

    result2 = report_writer_node(state_reject)
    print(f"\n{DASH}")
    print("📄 시나리오 2 — IC 메모 (REJECT/FALLBACK)")
    print(DASH)
    print(result2["final_report"])

    # ══════════════════════════════════════════════════════════════════════
    # 검증 Summary
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n\n{SEP}")
    print("✅ 시뮬레이션 검증 요약")
    print(SEP)

    r1 = result1["final_report"]
    r2 = result2["final_report"]

    checks = [
        ("시나리오1: '# ' 헤더 존재 (마크다운)",
         "# " in r1),
        ("시나리오1: 'BUY' 또는 'HOLD' verdict 포함",
         "BUY" in r1 or "HOLD" in r1),
        ("시나리오1: Alpha Thesis 섹션 존재",
         "Alpha Thesis" in r1),
        ("시나리오1: Key Assumptions 섹션 존재",
         "Key Assumptions" in r1 or "Review Triggers" in r1),
        ("시나리오2: 'REJECTED' 또는 'NO TRADE' verdict 포함",
         "REJECTED" in r2 or "NO TRADE" in r2),
        ("시나리오2: 반려 사유 섹션 존재",
         "반려 사유" in r2 or "Rejection" in r2),
        ("시나리오2: 재검토 조건 셕션 존재",
         "재검토 조건" in r2 or "Review Triggers" in r2),
        ("시나리오2: 대안 자산(CASH/SHY/TLT) 언급",
         any(t in r2 for t in ("CASH", "SHY", "TLT"))),
    ]

    all_pass = True
    for label, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_pass = False
        print(f"   {status}  {label}")

    print(f"\n{'✅ 모든 검증 통과!' if all_pass else '❌ 일부 검증 실패 — 위 항목 확인 필요'}")
