"""
agents/macro_agent.py — ② Macro Analyst Agent (v2)
====================================================
v2: 5축 분해 + risk_on_off + key_drivers + what_to_watch + scenario_notes
LLM 없음. Python-only. deterministic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from agents.autonomy_overlay import apply_llm_overlay_macro
from engines.macro_engine import (
    compute_macro_features,
    compute_overlay_guidance,
    compute_macro_axes,
    compute_risk_on_off,
)
from schemas.common import make_evidence, make_risk_flag
from schemas.taxonomy import map_macro_regime_to_canonical


# ── Template helpers ──────────────────────────────────────────────────────────

def _axis_label(axis_name: str, state: str, score: int) -> str:
    arrows = {3: "↑↑", 2: "↑", 1: "↑", 0: "→", -1: "↓", -2: "↓", -3: "↓↓"}
    return f"{axis_name.capitalize()} {arrows.get(score, '→')} ({state})"


def _build_key_drivers(axes: dict, ron: dict, features: dict, indicators: dict) -> list[str]:
    drivers = []
    for name, ax in axes.items():
        score = ax["score"]
        state = ax["state"]
        if abs(score) >= 2:
            sign = "+" if score > 0 else "-"
            drivers.append(_axis_label(name, state, score) + f"  [{sign}{abs(score)}]")

    hy = indicators.get("hy_oas")
    yc = indicators.get("yield_curve_spread")
    pmi = indicators.get("pmi")

    if hy and hy > 400:
        drivers.append(f"Credit spreads elevated (HY OAS {hy:.0f}bp)")
    if yc is not None and yc < -0.10:
        drivers.append(f"Yield curve inverted ({yc:+.2f}%)")
    if pmi and pmi < 50:
        drivers.append(f"PMI in contraction ({pmi:.1f})")
    if ron.get("tail_risk_warning"):
        drivers.append(f"Tail risk: {ron.get('tail_risk_level', 'medium').upper()}")
    return drivers[:6]


def _build_what_to_watch(axes: dict, indicators: dict) -> list[str]:
    items = []
    hy = indicators.get("hy_oas")
    yc = indicators.get("yield_curve_spread")
    pmi = indicators.get("pmi")
    ffr = indicators.get("fed_funds_rate")

    if hy and hy > 400:
        normal_hy = 300
        items.append(f"HY OAS < {normal_hy}bp 복귀 시 크레딧 완화 신호")
    if yc is not None and yc < 0:
        items.append("10Y-2Y 스프레드 플러스 전환 시 레짐 개선 확인")
    if pmi:
        items.append(f"PMI 50 상향 돌파 여부 (현재 {pmi:.1f})")
    if ffr:
        items.append(f"Fed 금리 인하 신호 — 현 FFR {ffr:.2f}%")
    if axes.get("growth", {}).get("score", 0) < -1:
        items.append("GDP/PMI 반등 확인 후 리스크 자산 편입 재고")
    items.append("다음 FOMC / CPI 발표 결과 모니터링")
    return items[:5]


def _build_scenario_notes(regime: str, ron: dict) -> dict:
    tail = ron.get("tail_risk_warning", False)

    scenarios: dict[str, dict] = {
        "goldilocks": {
            "bull": "성장 가속 + 인플레 완화 지속 → 리스크 자산 전반 강세. 주식 비중 확대 적기.",
            "base": "Goldilocks 지속. 이익 성장 + 밸류에이션 재평가. 분할 매수 유효.",
            "bear": "인플레 재점화 또는 연준 매파 전환 시 조정. 금리 민감 섹터 익스포저 축소.",
        },
        "reflation": {
            "bull": "실적 서프라이즈 + 원자재 상승 수혜 — 가치주/금융/에너지 강세.",
            "base": "금리 상승 속도 조절 국면. 듀레이션 축소, 실물 자산 비중 유지.",
            "bear": "인플레 통제 실패 시 긴축 가속 → 성장주 밸류에이션 압박. 현금 비중 확대.",
        },
        "late_cycle": {
            "bull": "연착륙 성공 시 방어 + 퀄리티 주 상대 우위 유지. 하방 헤지 비용 저렴.",
            "base": "점진적 노출 축소. 퀄리티/저변동 팩터 위주, 사이클 익스포저 감소.",
            "bear": "경기 침체 선반영 시 급락. 채권 듀레이션 확대, 현금 20-30% 권고.",
        },
        "stagflation": {
            "bull": "에너지·원자재·실물 헤지로 인플레 수혜 제한적. 리얼 에셋 유지.",
            "base": "방어 섹터(헬스케어·필수소비재)와 현금 혼합. 주식 비중 최소화.",
            "bear": "스태그 심화 시 전 자산 동반 하락 가능. 현금 + 금·단기채 최대 방어.",
        },
        "contraction": {
            "bull": "정책 반전(금리 인하/QE) 확인 직후 반등. 그때까지 현금 대기.",
            "base": "현금·국채·금 위주. 주식 최소. 경기 바닥 확인 후 분할 편입.",
            "bear": "침체 장기화 시 기업 실적 급락, 신용 이벤트 위험. 숏 헤지 필수.",
        },
        "expansion": {
            "bull": "성장 모멘텀 지속 + 유동성 확장 → 사이클·성장주 강세.",
            "base": "정상 확장기. 분산 포트폴리오, 중기 롱 포지션 유지.",
            "bear": "공급 충격 또는 지정학 이벤트로 급격한 레짐 전환 위험. 헤지 비중 일부 확보.",
        },
    }

    note = dict(scenarios.get(regime, scenarios["expansion"]))
    if tail:
        note["bear"] = "⚠️ Tail Risk 活性化 — " + note["bear"]
    return note


_MACRO_EVIDENCE_KINDS = {"macro_headline_context", "macro_release", "press_release_or_ir"}
_PATCHABLE_FIELDS = {
    "key_drivers",
    "what_to_watch",
    "scenario_notes",
    "open_questions",
    "decision_sensitivity",
    "followups",
    "react_trace",
}


def _request_key(req: dict) -> tuple:
    return (
        req.get("desk", ""),
        req.get("kind", ""),
        req.get("ticker", ""),
        req.get("series_id", ""),
        req.get("query", ""),
    )


def _merge_requests(base: list[dict], extra: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen = set()
    for req in (base or []) + (extra or []):
        if not isinstance(req, dict):
            continue
        k = _request_key(req)
        if k in seen:
            continue
        seen.add(k)
        out.append(req)
    return out


def _build_evidence_digest(state: Optional[dict], ticker: str, max_items: int = 5) -> list[dict]:
    store = (state or {}).get("evidence_store", {})
    if not isinstance(store, dict):
        return []
    out: list[dict] = []
    for item in store.values():
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", ""))
        desk = str(item.get("desk", ""))
        if kind and kind not in _MACRO_EVIDENCE_KINDS and desk not in ("macro", ""):
            continue
        item_ticker = str(item.get("ticker", "")).upper()
        if item_ticker and item_ticker != ticker.upper():
            continue
        out.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "published_at": item.get("published_at", ""),
                "trust_tier": item.get("trust_tier", 0.4),
                "kind": kind,
                "resolver_path": item.get("resolver_path", ""),
            }
        )
    out.sort(key=lambda x: (x.get("published_at") or ""), reverse=True)
    return out[:max_items]


def _default_open_questions(ticker: str, focus_areas: list[str], regime: str) -> list[dict]:
    out: list[dict] = []
    for focus in focus_areas[:2]:
        out.append(
            {
                "q": f"{focus}가 {ticker}의 매크로 레짐({regime})을 바꿀 촉매인지?",
                "why": "결론 변경 가능성이 큰 매크로 불확실성 확인",
                "kind": "macro_headline_context",
                "priority": 2,
                "recency_days": 7,
            }
        )
    if not out:
        out.append(
            {
                "q": f"{ticker} 관련 최근 매크로 헤드라인 중 레짐 전환 신호가 있는가?",
                "why": "risk_on/off 판단의 최신 근거 보강",
                "kind": "macro_headline_context",
                "priority": 2,
                "recency_days": 7,
            }
        )
    return out[:5]


def _default_decision_sensitivity(regime: str) -> list[dict]:
    return [
        {
            "if": "HY OAS가 50bp 이상 추가 확대",
            "then_change": "매크로 결론을 한 단계 보수적으로 조정",
            "impact": "high",
        },
        {
            "if": f"현재 레짐({regime})이 다음 주요 지표 발표 후 유지",
            "then_change": "기존 결론 유지",
            "impact": "medium",
        },
    ]


def _default_followups() -> list[dict]:
    return [
        {
            "type": "run_research",
            "detail": "매크로 이벤트/공식 릴리즈 근거를 추가 수집",
            "params": {"kind": "macro_headline_context"},
        },
        {
            "type": "rerun_desk",
            "detail": "새 근거 반영 후 매크로 데스크 재평가",
            "params": {"desk": "macro"},
        },
    ]


def _default_react_trace(has_evidence: bool) -> list[dict]:
    return [
        {"phase": "THOUGHT", "summary": "결론을 바꿀 거시 변수 공백을 점검"},
        {"phase": "ACTION", "summary": "우선순위 질문과 리서치 요청을 구조화"},
        {
            "phase": "OBSERVATION",
            "summary": "증거 반영 여부를 핵심 드라이버에 업데이트" if has_evidence else "신규 증거 대기 상태",
        },
        {"phase": "REFLECTION", "summary": "결론 민감도와 조건부 시나리오를 재확인"},
    ]


def _score_signal(score: float, *, positive: str = "tailwind", negative: str = "headwind") -> str:
    if score >= 2:
        return f"strong_{positive}"
    if score >= 1:
        return positive
    if score <= -2:
        return f"strong_{negative}"
    if score <= -1:
        return negative
    return "neutral"


def _macro_context_text(state: Optional[dict], focus_areas: Optional[list[str]]) -> str:
    parts: list[str] = []
    if isinstance(state, dict):
        parts.append(str(state.get("user_request", "")))
    parts.extend(str(x) for x in (focus_areas or []))
    return " ".join(parts).lower()


def _resolve_macro_universe(ticker: str, state: Optional[dict]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    def _push(raw: Any) -> None:
        t = str(raw or "").strip().upper()
        if not t or t in seen:
            return
        seen.add(t)
        out.append(t)

    _push(ticker)
    if isinstance(state, dict):
        for raw in state.get("universe", []) or []:
            _push(raw)
        directives = state.get("orchestrator_directives", {})
        brief = directives.get("investment_brief", {}) if isinstance(directives, dict) else {}
        for raw in brief.get("target_universe", []) if isinstance(brief, dict) else []:
            _push(raw)
    return out[:8]


def _asset_type_for_macro(ticker: str, state: Optional[dict]) -> str:
    if isinstance(state, dict):
        asset_map = state.get("asset_type_by_ticker", {})
        if isinstance(asset_map, dict):
            at = str(asset_map.get(str(ticker or "").upper(), "")).strip().upper()
            if at:
                return at
    t = str(ticker or "").strip().upper()
    if t in {"TLT", "IEF", "SHY", "BND", "ZROZ", "EDV"}:
        return "BOND"
    if t in {"GLD", "IAU", "GDX", "SGOL", "GC"}:
        return "COMMODITY"
    if t in {"SPY", "QQQ", "VOO", "IVV", "VTI", "DIA", "IWM", "XLE", "XLK", "XLF", "XLP"}:
        return "ETF"
    if t.startswith("^") or t in {"NDX", "SPX"}:
        return "INDEX"
    return "EQUITY"


def _ticker_macro_bucket(ticker: str, state: Optional[dict]) -> str:
    t = str(ticker or "").strip().upper()
    asset_type = _asset_type_for_macro(t, state)
    if t in {"QQQ", "NDX", "^NDX", "XLK", "SOXX", "SMH", "NVDA", "AMD", "MSFT", "AAPL", "META", "AMZN"}:
        return "growth_equity"
    if t in {"TLT", "IEF", "SHY", "BND", "ZROZ", "EDV"} or asset_type == "BOND":
        return "duration"
    if t in {"GLD", "IAU", "GDX", "SGOL", "GC"}:
        return "gold"
    if t in {"XLE", "USO", "BNO", "OIH"}:
        return "energy"
    if asset_type in {"ETF", "INDEX"}:
        return "broad_equity"
    return "single_name_equity"


def _portfolio_score_for_bucket(bucket: str, axes: dict, ron: dict) -> float:
    growth = float(axes.get("growth", {}).get("score", 0))
    inflation = float(axes.get("inflation", {}).get("score", 0))
    rates = float(axes.get("rates", {}).get("score", 0))
    credit = float(axes.get("credit", {}).get("score", 0))
    liquidity = float(axes.get("liquidity", {}).get("score", 0))
    tail_bonus = 1.0 if ron.get("tail_risk_warning") else 0.0
    risk_off_penalty = 1.0 if ron.get("risk_on_off") == "risk_off" else 0.0

    if bucket == "growth_equity":
        return growth + credit + liquidity + rates - 0.75 * max(inflation, 0)
    if bucket == "duration":
        return 1.25 * rates - 0.50 * growth - 0.25 * max(inflation, 0) + 0.75 * tail_bonus
    if bucket == "gold":
        return 0.75 * max(inflation, 0) + 0.50 * rates + 0.75 * tail_bonus - 0.25 * max(growth, 0)
    if bucket == "energy":
        return 0.75 * growth + 1.00 * max(inflation, 0) - 0.75 * risk_off_penalty
    if bucket == "single_name_equity":
        return growth + 0.75 * credit + 0.50 * liquidity + 0.50 * rates - 0.50 * max(inflation, 0)
    return growth + 0.75 * credit + 0.50 * liquidity + 0.50 * rates - 0.50 * max(inflation, 0)


def _stance_from_score(score: float) -> str:
    if score >= 2.0:
        return "overweight"
    if score >= 0.75:
        return "lean_overweight"
    if score > -0.75:
        return "neutral"
    if score > -2.0:
        return "lean_underweight"
    return "underweight"


def _build_transmission_map(
    axes: dict,
    ron: dict,
    indicators: dict,
    focus_areas: Optional[list[str]],
    state: Optional[dict],
) -> dict:
    context_text = _macro_context_text(state, focus_areas)
    inflation_val = indicators.get("core_cpi_yoy")
    if inflation_val is None:
        inflation_val = indicators.get("cpi_yoy")
    if inflation_val is None:
        inflation_val = indicators.get("inflation_expectation")

    transmission = {
        "growth_beta": {
            "score": axes.get("growth", {}).get("score", 0),
            "signal": _score_signal(axes.get("growth", {}).get("score", 0)),
            "current_state": axes.get("growth", {}).get("state", "unknown"),
            "current_value": indicators.get("pmi") if indicators.get("pmi") is not None else indicators.get("gdp_growth"),
            "portfolio_readthrough": "경기민감주·광범위 주식 beta의 상방/하방을 결정",
        },
        "policy_rates": {
            "score": axes.get("rates", {}).get("score", 0),
            "signal": _score_signal(axes.get("rates", {}).get("score", 0), positive="supportive", negative="restrictive"),
            "current_state": axes.get("rates", {}).get("state", "unknown"),
            "current_value": indicators.get("fed_funds_rate"),
            "portfolio_readthrough": "듀레이션 자산과 성장주의 할인율 경로에 직접 영향",
        },
        "credit": {
            "score": axes.get("credit", {}).get("score", 0),
            "signal": _score_signal(axes.get("credit", {}).get("score", 0)),
            "current_state": axes.get("credit", {}).get("state", "unknown"),
            "current_value": indicators.get("hy_oas"),
            "portfolio_readthrough": "신용 스프레드 확대는 지수 beta와 하이베타 자산에 즉각적인 압박",
        },
        "inflation_real_assets": {
            "score": axes.get("inflation", {}).get("score", 0),
            "signal": _score_signal(axes.get("inflation", {}).get("score", 0), positive="inflationary", negative="disinflationary"),
            "current_state": axes.get("inflation", {}).get("state", "unknown"),
            "current_value": inflation_val,
            "portfolio_readthrough": "실질금리·금·에너지·듀레이션 민감 자산의 상대 강약을 바꿈",
        },
        "liquidity": {
            "score": axes.get("liquidity", {}).get("score", 0),
            "signal": _score_signal(axes.get("liquidity", {}).get("score", 0), positive="supportive", negative="draining"),
            "current_state": axes.get("liquidity", {}).get("state", "unknown"),
            "current_value": indicators.get("financial_conditions_index"),
            "portfolio_readthrough": "멀티플 확장/축소와 beta 수용도를 결정",
        },
    }
    sofr_change = indicators.get("sofr_futures_implied_change_6m_bp")
    futures_change = indicators.get("fed_funds_futures_implied_change_6m_bp")
    basis_bp = indicators.get("sofr_ff_6m_basis_bp")
    if basis_bp is None:
        ff_6m = indicators.get("fed_funds_futures_6m_implied_rate")
        sofr_6m = indicators.get("sofr_futures_6m_implied_rate")
        if ff_6m is not None and sofr_6m is not None:
            basis_bp = round((sofr_6m - ff_6m) * 100, 1)

    if sofr_change is not None:
        transmission["rates_pricing"] = {
            "score": sofr_change,
            "signal": "dovish_pricing" if sofr_change <= -10 else ("hawkish_pricing" if sofr_change >= 10 else "balanced_pricing"),
            "current_state": (
                "divergent_pricing"
                if basis_bp is not None and abs(basis_bp) >= 10
                else ("easing_priced" if sofr_change <= -10 else ("higher_rates_priced" if sofr_change >= 10 else "balanced"))
            ),
            "current_value": sofr_change,
            "primary_metric": "sofr_futures_implied_change_6m_bp",
            "secondary_metric": "fed_funds_futures_implied_change_6m_bp" if futures_change is not None else None,
            "secondary_value": futures_change,
            "basis_bp": basis_bp,
            "portfolio_readthrough": "SOFR futures를 우선으로, Fed funds futures를 보조로 사용해 정책 경로와 듀레이션·성장주 할인율 민감도를 보정",
        }
    elif futures_change is not None:
        transmission["rates_pricing"] = {
            "score": futures_change,
            "signal": "dovish_pricing" if futures_change <= -10 else ("hawkish_pricing" if futures_change >= 10 else "balanced_pricing"),
            "current_state": "easing_priced" if futures_change <= -10 else ("higher_rates_priced" if futures_change >= 10 else "balanced"),
            "current_value": futures_change,
            "primary_metric": "fed_funds_futures_implied_change_6m_bp",
            "secondary_metric": None,
            "secondary_value": None,
            "basis_bp": basis_bp,
            "portfolio_readthrough": "실제 Fed funds futures curve 기반 경로로 듀레이션·성장주 민감도를 보정",
        }
    cuts_proxy = indicators.get("cuts_priced_proxy_2y_ffr_bp")
    if cuts_proxy is not None and "rates_pricing" not in transmission:
        transmission["rates_pricing"] = {
            "score": cuts_proxy,
            "signal": "dovish_pricing" if cuts_proxy >= 25 else ("hawkish_pricing" if cuts_proxy <= -25 else "balanced_pricing"),
            "current_state": "cuts_priced" if cuts_proxy >= 25 else ("hikes_priced" if cuts_proxy <= -25 else "balanced"),
            "current_value": cuts_proxy,
            "primary_metric": "cuts_priced_proxy_2y_ffr_bp",
            "secondary_metric": None,
            "secondary_value": None,
            "basis_bp": basis_bp,
            "portfolio_readthrough": "2Y-FFR gap 기반 easing/tightening pricing proxy로 듀레이션·성장주 민감도를 보정",
        }
    vix_level = indicators.get("vix_level")
    if vix_level is None:
        vix_level = indicators.get("vix_index")
    if vix_level is not None:
        transmission["volatility"] = {
            "score": vix_level,
            "signal": "strong_headwind" if vix_level >= 30 else ("headwind" if vix_level >= 20 else ("supportive" if vix_level <= 15 else "neutral")),
            "current_state": "elevated" if vix_level >= 20 else "contained",
            "current_value": vix_level,
            "portfolio_readthrough": "변동성 급등은 broad equity net 노출 축소와 헤지 수요 증가로 연결",
        }
    dollar_index = indicators.get("dollar_index")
    if dollar_index is not None:
        transmission["usd"] = {
            "score": dollar_index,
            "signal": "watch",
            "current_state": "firm_usd" if axes.get("rates", {}).get("score", 0) <= -1 else "stable_usd",
            "current_value": dollar_index,
            "portfolio_readthrough": "달러 강세는 글로벌 위험자산·원자재·실적 환산효과에 영향을 준다",
        }

    if any(tok in context_text for tok in ("oil", "wti", "brent", "energy", "원유", "석유", "호르무즈", "hormuz")):
        transmission["commodity_shock_watch"] = {
            "score": indicators.get("wti_spot") or indicators.get("wti_front_month"),
            "signal": "watch",
            "current_state": "headline_sensitive",
            "current_value": indicators.get("wti_spot") or indicators.get("wti_front_month") or indicators.get("brent_spot") or indicators.get("brent_front_month"),
            "portfolio_readthrough": "원유 쇼크는 인플레 재상승과 성장 둔화를 동시에 자극할 수 있어 XLE·GLD·장기채 판단을 바꿈",
        }
    else:
        transmission["commodity_shock_watch"] = {
            "score": indicators.get("wti_spot") or indicators.get("wti_front_month"),
            "signal": "unknown",
            "current_state": "not_modeled",
            "current_value": indicators.get("wti_spot") or indicators.get("wti_front_month") or indicators.get("brent_spot") or indicators.get("brent_front_month"),
            "portfolio_readthrough": "WTI/Brent 입력이 없어 원자재 충격 경로는 이벤트 기반으로만 감시 중",
        }

    if ron.get("tail_risk_warning"):
        transmission["tail_risk"] = {
            "score": ron.get("risk_score"),
            "signal": "active",
            "current_state": ron.get("tail_risk_level", "medium"),
            "current_value": None,
            "portfolio_readthrough": "지수 익스포저 축소와 방어 헤지(GLD/TLT) 수요가 강화되는 구간",
        }

    return transmission


def _build_portfolio_implications(
    ticker: str,
    axes: dict,
    ron: dict,
    state: Optional[dict],
) -> dict:
    universe = _resolve_macro_universe(ticker, state)
    targets: list[dict] = []
    main_ticker = universe[0] if universe else str(ticker or "").upper()

    for candidate in universe:
        bucket = _ticker_macro_bucket(candidate, state)
        score = round(_portfolio_score_for_bucket(bucket, axes, ron), 2)
        stance = _stance_from_score(score)
        if bucket == "growth_equity":
            rationale = "성장/신용/유동성 우호 여부와 금리 할인율 압력을 함께 반영"
        elif bucket == "duration":
            rationale = "정책금리 경로와 tail-risk 방어 수요를 반영"
        elif bucket == "gold":
            rationale = "인플레/실질금리/리스크오프 방어 수요를 반영"
        elif bucket == "energy":
            rationale = "성장과 인플레 충격의 교차효과를 반영"
        else:
            rationale = "광범위 equity beta와 신용/유동성 환경을 반영"

        targets.append(
            {
                "ticker": candidate,
                "bucket": bucket,
                "stance": stance,
                "macro_fit_score": score,
                "role": "main_exposure" if candidate == main_ticker else "candidate",
                "rationale": rationale,
            }
        )

    hedge_like = [x for x in targets if x["ticker"] != main_ticker and x["bucket"] in {"duration", "gold", "energy"}]
    preferred_hedges = [x["ticker"] for x in sorted(hedge_like, key=lambda x: x["macro_fit_score"], reverse=True)[:2]]

    return {
        "context": {
            "macro_regime": "unknown",
            "risk_on_off": ron.get("risk_on_off", "neutral"),
            "tail_risk_warning": bool(ron.get("tail_risk_warning")),
            "main_ticker": main_ticker,
            "universe_size": len(universe),
        },
        "targets": targets,
        "preferred_hedges": preferred_hedges,
        "main_takeaway": (
            "tail-risk 방어 헤지를 유지하며 beta를 관리"
            if ron.get("tail_risk_warning")
            else "거시 레짐과 금리/신용 조건에 맞춰 main exposure와 hedges를 선별"
        ),
    }


def _build_monitoring_triggers(
    axes: dict,
    indicators: dict,
    focus_areas: Optional[list[str]],
    state: Optional[dict],
) -> list[dict]:
    context_text = _macro_context_text(state, focus_areas)
    inflation_val = indicators.get("core_cpi_yoy")
    if inflation_val is None:
        inflation_val = indicators.get("cpi_yoy")
    if inflation_val is None:
        inflation_val = indicators.get("inflation_expectation")

    triggers: list[dict] = [
        {
            "name": "Credit stress escalation",
            "metric": "hy_oas",
            "current_value": indicators.get("hy_oas"),
            "trigger": "> 500bp",
            "action": "지수 beta를 한 단계 축소하고 방어 헤지(GLD/TLT) 우선순위를 재평가",
            "priority": 1,
        },
        {
            "name": "Growth rollover",
            "metric": "pmi",
            "current_value": indicators.get("pmi"),
            "trigger": "< 48",
            "action": "growth/cyclical 노출을 줄이고 경기방어 또는 듀레이션 방어를 검토",
            "priority": 1,
        },
        {
            "name": "Inflation re-acceleration",
            "metric": "inflation_proxy",
            "current_value": inflation_val,
            "trigger": "> 3.0% and rising",
            "action": "장기채/성장주 비중을 재검토하고 금·에너지 상대 강세 여부를 확인",
            "priority": 1,
        },
        {
            "name": "Policy repricing",
            "metric": (
                "sofr_futures_implied_change_6m_bp"
                if indicators.get("sofr_futures_implied_change_6m_bp") is not None
                else (
                    "fed_funds_futures_implied_change_6m_bp"
                    if indicators.get("fed_funds_futures_implied_change_6m_bp") is not None
                    else "cuts_priced_proxy_2y_ffr_bp"
                )
            ),
            "current_value": (
                indicators.get("sofr_futures_implied_change_6m_bp")
                if indicators.get("sofr_futures_implied_change_6m_bp") is not None
                else (
                    indicators.get("fed_funds_futures_implied_change_6m_bp")
                    if indicators.get("fed_funds_futures_implied_change_6m_bp") is not None
                    else indicators.get("cuts_priced_proxy_2y_ffr_bp")
                )
            ),
            "trigger": (
                "< -10bp or > 10bp"
                if (
                    indicators.get("sofr_futures_implied_change_6m_bp") is not None
                    or indicators.get("fed_funds_futures_implied_change_6m_bp") is not None
                )
                else "< -25bp or > 50bp"
            ),
            "action": "듀레이션·성장주 할인율 가정을 즉시 업데이트",
            "priority": 2,
        },
        {
            "name": "Curve confirmation",
            "metric": "yield_curve_spread",
            "current_value": indicators.get("yield_curve_spread"),
            "trigger": "< -0.20 or > 0.30",
            "action": "침체/재확장 신호로 해석해 macro regime과 risk budget을 다시 점검",
            "priority": 2,
        },
    ]
    vix_level = indicators.get("vix_level")
    if vix_level is None:
        vix_level = indicators.get("vix_index")
    if vix_level is not None:
        triggers.append(
            {
                "name": "Volatility regime shift",
                "metric": "vix_level",
                "current_value": vix_level,
                "trigger": "> 20 / > 30",
                "action": "beta 노출과 tail hedge 수요를 재평가",
                "priority": 1,
            }
        )
    dollar_index = indicators.get("dollar_index")
    if dollar_index is not None:
        triggers.append(
            {
                "name": "Dollar regime shift",
                "metric": "dollar_index",
                "current_value": dollar_index,
                "trigger": "sudden breakout with tighter rates",
                "action": "원자재·금·글로벌 risk appetite 경로를 재평가",
                "priority": 2,
            }
        )
    basis_bp = indicators.get("sofr_ff_6m_basis_bp")
    if basis_bp is None:
        ff_6m = indicators.get("fed_funds_futures_6m_implied_rate")
        sofr_6m = indicators.get("sofr_futures_6m_implied_rate")
        if ff_6m is not None and sofr_6m is not None:
            basis_bp = round((sofr_6m - ff_6m) * 100, 1)
    if basis_bp is not None:
        triggers.append(
            {
                "name": "Rates basis divergence",
                "metric": "sofr_ff_6m_basis_bp",
                "current_value": basis_bp,
                "trigger": "< -10bp or > 10bp",
                "action": "SOFR와 Fed funds pricing 괴리를 확인하고 정책경로 해석을 한 단계 보수적으로 재점검",
                "priority": 2,
            }
        )

    if any(tok in context_text for tok in ("oil", "wti", "brent", "energy", "원유", "석유", "호르무즈", "hormuz")):
        triggers.append(
            {
                "name": "Commodity shock",
                "metric": "WTI/Brent",
                "current_value": indicators.get("wti_spot") or indicators.get("wti_front_month") or indicators.get("brent_spot") or indicators.get("brent_front_month"),
                "trigger": "> 80 or +10% spike / logistics disruption",
                "action": "inflation/energy hedge와 broad equity 감도 재평가",
                "priority": 1,
            }
        )
    if any(tok in context_text for tok in ("fomc", "fed", "금리", "인플레", "cpi", "정책")):
        triggers.append(
            {
                "name": "Macro release surprise",
                "metric": "CPI/FOMC/NFP",
                "current_value": None,
                "trigger": "actual vs priced path mismatch",
                "action": "policy_rates 채널과 portfolio implications를 즉시 다시 계산",
                "priority": 2,
            }
        )

    return triggers[:7]


def _apply_overlay_patch(output: dict, patch: dict) -> None:
    if not isinstance(patch, dict):
        return
    for key in _PATCHABLE_FIELDS:
        if key in patch and patch[key]:
            output[key] = patch[key]
    if patch.get("evidence_requests"):
        output["evidence_requests"] = _merge_requests(
            output.get("evidence_requests", []),
            patch.get("evidence_requests", []),
        )
        output["needs_more_data"] = bool(output.get("evidence_requests"))


def _generate_evidence_requests(
    ticker: str,
    axes: dict,
    ron: dict,
    features: dict,
    indicators: dict,
    focus_areas: Optional[list[str]] = None,
) -> list:
    """매크로 evidence request 생성: regime flip/axes 급변 + driver 불명."""
    reqs = []
    risk_on_off = ron.get("risk_on_off", "neutral")
    if risk_on_off == "risk_off" and not indicators.get("yield_curve_spread"):
        reqs.append(
            {
                "desk": "macro",
                "kind": "macro_headline_context",
                "ticker": ticker,
                "query": f"{ticker} macro headwind driver recession risk",
                "priority": 2,
                "recency_days": 7,
                "max_items": 5,
                "rationale": "risk_off but missing yield curve — need context",
            }
        )
    if ron.get("tail_risk_warning") and not indicators.get("hy_oas"):
        reqs.append(
            {
                "desk": "macro",
                "kind": "macro_headline_context",
                "ticker": ticker,
                "query": "high yield credit spread stress financial conditions",
                "priority": 1,
                "recency_days": 3,
                "max_items": 3,
                "rationale": "tail_risk active but HY OAS missing",
            }
        )
    g_score = axes.get("growth", {}).get("score", 0)
    if abs(g_score) >= 2 and not indicators.get("pmi") and not indicators.get("gdp_growth"):
        reqs.append(
            {
                "desk": "macro",
                "kind": "macro_headline_context",
                "ticker": ticker,
                "query": "US PMI GDP growth contraction latest",
                "priority": 2,
                "recency_days": 7,
                "max_items": 3,
                "rationale": f"growth axis extreme ({g_score}) but PMI/GDP missing",
            }
        )
    prev_regime = indicators.get("prev_macro_regime")
    curr_regime = features.get("macro_regime")
    if prev_regime and curr_regime and prev_regime != curr_regime and not indicators.get("regime_flip_driver"):
        reqs.append(
            {
                "desk": "macro",
                "kind": "macro_headline_context",
                "ticker": ticker,
                "query": f"{ticker} macro regime flip driver",
                "priority": 1,
                "recency_days": 7,
                "max_items": 4,
                "rationale": f"regime flip {prev_regime}->{curr_regime} with unknown driver",
            }
        )

    for focus in (focus_areas or [])[:2]:
        reqs.append(
            {
                "desk": "macro",
                "kind": "macro_headline_context",
                "ticker": ticker,
                "query": f"{ticker} {focus} macro impact",
                "priority": 3,
                "recency_days": 14,
                "max_items": 3,
                "rationale": f"focus area follow-up: {focus}",
            }
        )
    return reqs


# ── Main pipeline ─────────────────────────────────────────────────────────────

def macro_analyst_run(
    ticker: str,
    macro_indicators: dict,
    *,
    run_id: str = "",
    as_of: str = "",
    horizon_days: int = 30,
    source_name: str = "mock",
    focus_areas: Optional[list[str]] = None,
    state: Optional[dict] = None,
) -> dict:
    """
    Macro Analyst v2: engine compute → 5축/risk_on_off → key_drivers/what_to_watch/scenario_notes.
    LLM overlay is optional and patch-only.
    """
    as_of = as_of or datetime.now(timezone.utc).isoformat()
    focus_areas = [str(x).strip() for x in (focus_areas or []) if str(x).strip()]

    # ── Engine calls ─────────────────────────────────────────────────
    features = compute_macro_features(macro_indicators)
    overlay = compute_overlay_guidance(features)
    axes = compute_macro_axes(macro_indicators)
    ron = compute_risk_on_off(axes, macro_indicators)

    regime = features["macro_regime"]
    tail_risk = ron["tail_risk_warning"]
    risk_on_off = ron["risk_on_off"]

    # ── Evidence ──────────────────────────────────────────────────────
    q = 0.3 if source_name == "mock" else 0.7
    evidence = []
    for key in [
        "dgs2",
        "dgs10",
        "yield_curve_spread",
        "hy_oas",
        "inflation_expectation",
        "cpi_yoy",
        "core_cpi_yoy",
        "pmi",
        "fed_funds_rate",
        "sofr_rate",
        "gdp_growth",
        "unemployment_rate",
        "financial_conditions_index",
        "real_10y_yield",
        "cuts_priced_proxy_2y_ffr_bp",
        "fed_funds_futures_front_implied_rate",
        "fed_funds_futures_3m_implied_rate",
        "fed_funds_futures_6m_implied_rate",
        "fed_funds_futures_implied_change_6m_bp",
        "sofr_futures_front_implied_rate",
        "sofr_futures_3m_implied_rate",
        "sofr_futures_6m_implied_rate",
        "sofr_futures_implied_change_6m_bp",
        "sofr_ff_6m_basis_bp",
        "dollar_index",
        "vix_level",
        "vix_index",
        "wti_spot",
        "brent_spot",
        "wti_front_month",
        "brent_front_month",
    ]:
        val = macro_indicators.get(key)
        if val is not None:
            evidence.append(make_evidence(metric=key, value=val, source_name=source_name, quality=q, as_of=as_of))
    ff_6m = macro_indicators.get("fed_funds_futures_6m_implied_rate")
    sofr_6m = macro_indicators.get("sofr_futures_6m_implied_rate")
    if macro_indicators.get("sofr_ff_6m_basis_bp") is None and ff_6m is not None and sofr_6m is not None:
        macro_indicators["sofr_ff_6m_basis_bp"] = round((sofr_6m - ff_6m) * 100, 1)
    if macro_indicators.get("sofr_ff_6m_basis_bp") is not None:
        evidence.append(
            make_evidence(
                metric="sofr_ff_6m_basis_bp",
                value=macro_indicators["sofr_ff_6m_basis_bp"],
                source_name="derived:SOFR-FF",
                source_type="model",
                quality=q,
                as_of=as_of,
                note="SOFR 6m implied minus Fed funds 6m implied, in basis points",
            )
        )
    for bk in ["curve_state", "credit_stress_level", "inflation_state", "growth_state"]:
        evidence.append(
            make_evidence(
                metric=bk,
                value=features[bk],
                source_name="macro_engine",
                source_type="model",
                quality=0.9,
                as_of=as_of,
            )
        )

    # ── Risk flags ────────────────────────────────────────────────────
    risk_flags = []
    if tail_risk:
        risk_flags.append(
            make_risk_flag(
                "macro_tail_risk",
                ron.get("tail_risk_level", "high"),
                "Credit stress or yield curve + FCI",
            )
        )
    if regime in ("contraction", "stagflation"):
        risk_flags.append(make_risk_flag("macro_headwind", "high", f"Macro regime: {regime}"))

    # ── Primary decision ──────────────────────────────────────────────
    inflation_hot = axes.get("inflation", {}).get("score", 0) >= 2
    rates_rising = axes.get("rates", {}).get("score", 0) <= -1

    if risk_on_off == "risk_off" and tail_risk:
        primary_decision = "bearish"
        recommendation = "reject"
        confidence = 0.80
    elif risk_on_off == "risk_off":
        primary_decision = "bearish"
        recommendation = "allow_with_limits"
        confidence = 0.65
    elif risk_on_off == "risk_on" and inflation_hot and rates_rising:
        primary_decision = "neutral"
        recommendation = "allow_with_limits"
        confidence = 0.55
    elif risk_on_off == "risk_on":
        primary_decision = "bullish"
        recommendation = "allow"
        confidence = 0.65
    elif regime == "late_cycle":
        primary_decision = "neutral"
        recommendation = "allow_with_limits"
        confidence = 0.50
    else:
        primary_decision = "neutral"
        recommendation = "allow_with_limits"
        confidence = 0.55

    # ── Data quality ──────────────────────────────────────────────────
    keyed = {k for k in ["yield_curve_spread", "hy_oas", "gdp_growth", "pmi"] if macro_indicators.get(k) is not None}
    data_ok = len(keyed) >= 2
    missing = [k for k in ["yield_curve_spread", "hy_oas", "pmi", "cpi_yoy"] if macro_indicators.get(k) is None]
    if not data_ok:
        confidence = min(confidence, 0.40)

    limitations = []
    if source_name == "mock":
        limitations.append("Mock 데이터 사용 — 실제 매크로 지표와 차이 가능")
    if missing:
        limitations.append(f"누락 지표: {', '.join(missing)}")

    ev_reqs = _generate_evidence_requests(ticker, axes, ron, features, macro_indicators, focus_areas)

    # ── key_drivers / what_to_watch / scenario_notes ──────────────────
    key_drivers = _build_key_drivers(axes, ron, features, macro_indicators)
    what_to_watch = _build_what_to_watch(axes, macro_indicators)
    scenario_notes = _build_scenario_notes(regime, ron)
    evidence_digest = _build_evidence_digest(state, ticker)
    transmission_map = _build_transmission_map(axes, ron, macro_indicators, focus_areas, state)
    portfolio_implications = _build_portfolio_implications(ticker, axes, ron, state)
    portfolio_implications["context"]["macro_regime"] = map_macro_regime_to_canonical(regime)
    monitoring_triggers = _build_monitoring_triggers(axes, macro_indicators, focus_areas, state)
    if evidence_digest:
        title = str(evidence_digest[0].get("title", "external evidence")).strip()[:90]
        kinds = sorted({str(item.get("kind", "")).strip() for item in evidence_digest if item.get("kind")})
        kind_label = ", ".join(kinds[:2]) if kinds else "macro_evidence"
        resolver = str(evidence_digest[0].get("resolver_path", "")).strip() or "unknown"
        key_drivers = (
            key_drivers
            + [
                f"Evidence update: {title}",
                f"Evidence coverage: {len(evidence_digest)} items ({kind_label})",
            ]
        )[:6]
        what_to_watch = (
            what_to_watch
            + [
                "신규 증거와 매크로 축 점수의 불일치 여부 재확인",
                f"최신 근거 경로({resolver})의 후속 발표 추적",
            ]
        )[:5]

    open_questions = _default_open_questions(ticker, focus_areas, regime)
    decision_sensitivity = _default_decision_sensitivity(regime)
    followups = _default_followups()
    react_trace = _default_react_trace(bool(evidence_digest))

    total_fields = 4
    missing_pct = round(len(missing) / total_fields, 3)
    data_quality = {
        "missing_pct": missing_pct,
        "freshness_days": 0.0,
        "warnings": list(missing),
        "missing_fields": missing,
        "is_mock": source_name == "mock",
        "anomaly_flags": ["tail_risk_active"] if tail_risk else [],
        "source_timestamps": {},
    }
    indicator_snapshot = dict(features)
    for key in [
        "dgs2",
        "dgs10",
        "fed_funds_rate",
        "sofr_rate",
        "real_10y_yield",
        "dollar_index",
        "vix_level",
        "vix_index",
        "wti_spot",
        "brent_spot",
        "wti_front_month",
        "brent_front_month",
        "cuts_priced_proxy_2y_ffr_bp",
        "fed_funds_futures_front_implied_rate",
        "fed_funds_futures_3m_implied_rate",
        "fed_funds_futures_6m_implied_rate",
        "fed_funds_futures_implied_change_6m_bp",
        "sofr_futures_front_implied_rate",
        "sofr_futures_3m_implied_rate",
        "sofr_futures_6m_implied_rate",
        "sofr_futures_implied_change_6m_bp",
        "sofr_ff_6m_basis_bp",
        "unemployment_rate",
        "financial_conditions_index",
        "yield_curve_spread",
        "hy_oas",
        "pmi",
        "cpi_yoy",
        "core_cpi_yoy",
        "inflation_expectation",
        "gdp_growth",
    ]:
        value = macro_indicators.get(key)
        if value is not None:
            indicator_snapshot[key] = value

    output = {
        "agent_type": "macro",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of": as_of,
        "ticker": ticker,
        "horizon_days": horizon_days,
        "focus_areas": focus_areas,
        "primary_decision": primary_decision,
        "recommendation": recommendation,
        "confidence": confidence if data_ok else min(confidence, 0.40),
        "signal_strength": abs(ron["risk_score"]) / 100.0,
        "risk_flags": risk_flags,
        "evidence": evidence,
        "data_quality": data_quality,
        "limitations": limitations,
        "data_ok": data_ok,
        "summary": f"리스크: {risk_on_off}. 레짐: {regime}. {overlay.get('equity_overlay_guidance', '')}",
        "status": "ok",
        # Structured outputs
        "macro_axes": axes,
        "risk_on_off": ron,
        "overlay_guidance": overlay,
        "key_drivers": key_drivers,
        "what_to_watch": what_to_watch,
        "scenario_notes": scenario_notes,
        "evidence_digest": evidence_digest,
        "transmission_map": transmission_map,
        "portfolio_implications": portfolio_implications,
        "monitoring_triggers": monitoring_triggers,
        "open_questions": open_questions,
        "decision_sensitivity": decision_sensitivity,
        "followups": followups,
        "react_trace": react_trace,
        # Evidence requests
        "needs_more_data": bool(ev_reqs),
        "evidence_requests": ev_reqs,
        # Backward compat
        "macro_regime_raw": regime,
        "macro_regime": map_macro_regime_to_canonical(regime),
        "tail_risk_warning": tail_risk,
        "indicators": indicator_snapshot,
        "regime": map_macro_regime_to_canonical(regime),
        "gdp_growth": macro_indicators.get("gdp_growth"),
        "interest_rate": macro_indicators.get("fed_funds_rate"),
    }

    patch = apply_llm_overlay_macro(output, state, focus_areas, evidence_digest)
    _apply_overlay_patch(output, patch)

    if not output.get("open_questions"):
        output["open_questions"] = open_questions
    if not output.get("decision_sensitivity"):
        output["decision_sensitivity"] = decision_sensitivity
    if not output.get("followups"):
        output["followups"] = followups

    return output
