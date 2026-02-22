"""
engines/quant_engine.py — Quant 순수 연산 엔진
===============================================
CHANGELOG:
  v1.0 (2026-02-22) — quant_agent.py에서 연산 함수 추출.
                       ADF/Coint/OLS/HAC/HMM/GARCH/Kelly/CVaR 유지.

Iron Rule R3: 데이터 수집 코드 절대 금지. 입력=배열, 출력=JSON.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

# ── 선택적 의존성 ─────────────────────────────────────────────────────────

try:
    from statsmodels.tsa.stattools import adfuller, coint
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    HAS_SM = True
except ImportError:
    HAS_SM = False

try:
    from arch import arch_model  # type: ignore
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore
    HAS_HMM = True
except ImportError:
    HAS_HMM = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 헬퍼
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_FRAC_KELLY = 0.25
_MAX_PORTFOLIO_CVAR = 0.015


def _log_ret(prices: np.ndarray) -> np.ndarray:
    return np.diff(np.log(prices))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1-A. StatArb: 공적분 + ADF + Z-Score + Half-life
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calculate_statarb_metrics(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    pair_names: Tuple[str, str] = ("A", "B"),
) -> dict:
    """공적분 + ADF + Z-Score + OU half-life. 순수함수."""
    r: Dict[str, Any] = {
        "pair": list(pair_names),
        "hedge_ratio": None, "coint_pvalue": None,
        "adf_pvalue": None, "is_stationary": None,
        "current_z_score": None, "half_life_days": None,
        "error": None,
    }
    try:
        if not HAS_SM:
            raise RuntimeError("statsmodels 미설치")
        n = min(len(prices_a), len(prices_b))
        pa, pb = prices_a[-n:], prices_b[-n:]

        score, coint_p, _ = coint(pa, pb)
        r["coint_pvalue"] = round(float(coint_p), 4)

        X = add_constant(pb)
        ols = OLS(pa, X).fit()
        beta = float(ols.params[1])
        r["hedge_ratio"] = round(beta, 4)

        spread = pa - beta * pb
        adf_res = adfuller(spread, autolag="AIC")
        r["adf_pvalue"] = round(float(adf_res[1]), 4)
        r["is_stationary"] = float(adf_res[1]) < 0.05

        window = 20
        roll_mean = np.convolve(spread, np.ones(window) / window, mode="valid")
        roll_std = np.array([np.std(spread[i:i + window]) for i in range(len(spread) - window + 1)])
        roll_std = np.where(roll_std < 1e-8, 1e-8, roll_std)
        z = (spread[window - 1:] - roll_mean) / roll_std
        r["current_z_score"] = round(float(z[-1]), 4)

        S_lag = spread[:-1]
        S_cur = spread[1:]
        ar = OLS(S_cur, add_constant(S_lag)).fit()
        b = float(ar.params[1])
        theta = -np.log(abs(b)) if 0 < abs(b) < 1 else 1e-6
        r["half_life_days"] = round(float(np.log(2) / max(theta, 1e-8)), 2)
    except Exception as exc:
        r["error"] = str(exc)
    return r


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1-B. 팩터 노출도 (OLS + Newey-West HAC)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calculate_factor_exposures(
    prices: np.ndarray,
    market_prices: np.ndarray,
) -> dict:
    """시장 팩터 대비 알파/베타 OLS + Newey-West. 순수함수."""
    r: Dict[str, Any] = {
        "alpha_annualized": None, "beta": None,
        "newey_west_t_stat": None, "p_value": None,
        "r_squared": None, "error": None,
    }
    try:
        if not HAS_SM:
            raise RuntimeError("statsmodels 미설치")
        rs = _log_ret(prices)
        rm = _log_ret(market_prices)
        n = min(len(rs), len(rm))
        rs, rm = rs[-n:], rm[-n:]

        X = add_constant(rm)
        ols = OLS(rs, X).fit()
        hac = ols.get_robustcov_results(cov_type="HAC", maxlags=5)

        r["alpha_annualized"] = round(float(hac.params[0]) * 252 * 100, 4)
        r["beta"] = round(float(hac.params[1]), 4)
        r["newey_west_t_stat"] = round(float(hac.tvalues[0]), 4)
        r["p_value"] = round(float(hac.pvalues[0]), 4)
        r["r_squared"] = round(float(hac.rsquared), 4)
    except Exception as exc:
        r["error"] = str(exc)
    return r


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1-C. 시장 국면 (HMM) + 조건부 변동성 (GARCH)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_regime_and_volatility(prices: np.ndarray) -> dict:
    """3-State HMM 국면 + GARCH(1,1) 변동성 예측. 순수함수."""
    r: Dict[str, Any] = {
        "model_used": "GaussianHMM_3_States",
        "state_probabilities": None,
        "volatility_forecast": {"model": "GARCH(1,1)", "t_plus_1_volatility": None},
        "error": None,
    }
    try:
        rets = _log_ret(prices)

        if not HAS_HMM:
            raise RuntimeError("hmmlearn 미설치")
        X_hmm = rets.reshape(-1, 1)
        model = GaussianHMM(n_components=3, covariance_type="full", n_iter=200, random_state=42, verbose=False)
        model.fit(X_hmm)

        variances = model.covars_.flatten()
        order = np.argsort(variances)
        _, posteriors = model.score_samples(X_hmm)
        last_post = posteriors[-1]
        r["state_probabilities"] = {
            "regime_0_calm": round(float(last_post[order[0]]), 4),
            "regime_1_mean_revert": round(float(last_post[order[1]]), 4),
            "regime_2_high_vol": round(float(last_post[order[2]]), 4),
        }

        if HAS_ARCH:
            am = arch_model(rets * 100, vol="Garch", p=1, q=1, dist="Normal", mean="Constant")
            res = am.fit(disp="off", show_warning=False)
            fc = res.forecast(horizon=1)
            var_1d = float(fc.variance.values[-1, 0])
            r["volatility_forecast"]["t_plus_1_volatility"] = round(np.sqrt(var_1d) * np.sqrt(252) / 100, 4)
        else:
            r["error"] = "arch 미설치 — GARCH 생략"
    except Exception as exc:
        r["error"] = str(exc)
    return r


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1-D. CVaR + Kelly
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calculate_kelly_and_cvar(prices: np.ndarray) -> dict:
    """99% CVaR(ES) + Kelly 사이징. 순수함수."""
    r: Dict[str, Any] = {
        "max_portfolio_cvar_limit": _MAX_PORTFOLIO_CVAR,
        "asset_cvar_99_daily": None,
        "kelly_optimization": {"full_kelly_fraction": None, "fractional_multiplier": _FRAC_KELLY},
        "error": None,
    }
    try:
        rets = _log_ret(prices)
        if len(rets) < 50:
            raise ValueError("수익률 데이터 부족 (50일 이상 필요)")

        var_th = np.percentile(rets, 1)
        tail = rets[rets <= var_th]
        r["asset_cvar_99_daily"] = round(float(abs(np.mean(tail))) if len(tail) > 0 else float(abs(var_th)), 6)

        mu = float(np.mean(rets))
        sig2 = float(np.var(rets))
        kelly = max(mu / sig2, 0.0) if sig2 > 1e-12 else 0.0
        r["kelly_optimization"]["full_kelly_fraction"] = round(kelly, 4)
    except Exception as exc:
        r["error"] = str(exc)
    return r


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 통합 페이로드 조립
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_quant_payload(
    ticker: str,
    prices: np.ndarray,
    pair_prices: np.ndarray,
    market_prices: np.ndarray,
    pair_ticker: str = "MSFT",
) -> dict:
    """4개 통계 모듈 결과를 통합 JSON으로 조립. 순수함수."""
    sa = calculate_statarb_metrics(prices, pair_prices, pair_names=(ticker, pair_ticker))
    fe = calculate_factor_exposures(prices, market_prices)
    rv = detect_regime_and_volatility(prices)
    kc = calculate_kelly_and_cvar(prices)

    payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "market_regime_context": {
            "model_used": rv.get("model_used"),
            "state_probabilities": rv.get("state_probabilities") or {
                "regime_0_calm": None, "regime_1_mean_revert": None, "regime_2_high_vol": None,
            },
            "volatility_forecast": rv.get("volatility_forecast", {}),
        },
        "alpha_signals": {
            "statistical_arbitrage": {
                "adf_test": {"p_value": sa.get("adf_pvalue"), "is_stationary": sa.get("is_stationary")},
                "execution": {
                    "hedge_ratio": sa.get("hedge_ratio"),
                    "current_z_score": sa.get("current_z_score"),
                    "half_life_days": sa.get("half_life_days"),
                },
            },
            "factor_exposures": {
                "newey_west_t_stat": fe.get("newey_west_t_stat"),
                "p_value": fe.get("p_value"),
            },
        },
        "portfolio_risk_parameters": {
            "max_portfolio_cvar_limit": kc.get("max_portfolio_cvar_limit"),
            "asset_cvar_99_daily": kc.get("asset_cvar_99_daily"),
            "kelly_optimization": kc.get("kelly_optimization", {}),
        },
        "_diagnostics": {
            "statarb_error": sa.get("error"),
            "factor_error": fe.get("error"),
            "regime_error": rv.get("error"),
            "kelly_error": kc.get("error"),
        },
    }
    any_fatal = any(v is not None for v in [sa.get("error"), fe.get("error"), kc.get("error")])
    payload["_data_ok"] = not any_fatal
    return payload


def mock_quant_decision(payload: dict) -> dict:
    """4단계 CoT를 Python으로 구현한 규칙 기반 Mock. 순수함수."""
    cot = []
    alloc = 0.0

    regime = (payload.get("market_regime_context") or {}).get("state_probabilities") or {}
    sa = ((payload.get("alpha_signals") or {}).get("statistical_arbitrage") or {})
    fe = ((payload.get("alpha_signals") or {}).get("factor_exposures") or {})
    risk = payload.get("portfolio_risk_parameters") or {}
    kelly_opt = risk.get("kelly_optimization") or {}

    p_hv = regime.get("regime_2_high_vol") or 0.0
    defense = p_hv > 0.50
    cot.append(f"[Step1] regime_2_high_vol={p_hv:.0%} {'> 50% → 방어' if defense else '≤ 50% → 정상'}.")

    adf_p = ((sa.get("adf_test") or {}).get("p_value"))
    adf_ok = adf_p is not None and adf_p < 0.05
    nw_t = fe.get("newey_west_t_stat")
    fe_p = fe.get("p_value")
    factor_ok = nw_t is not None and fe_p is not None and abs(nw_t) >= 1.96 and fe_p < 0.05

    cot.append(f"[Step2] ADF p={adf_p} {'유의' if adf_ok else '비유의'}. Factor {'유의' if factor_ok else '비유의'}.")

    if not adf_ok and not factor_ok:
        return {"cot_reasoning": " ".join(cot) + " → HOLD.", "decision": "HOLD", "final_allocation_pct": 0.0}

    z = ((sa.get("execution") or {}).get("current_z_score")) or 0.0
    if z > 2.0:
        decision = "SHORT"
    elif z < -2.0:
        decision = "LONG"
    elif abs(z) < 0.5:
        return {"cot_reasoning": " ".join(cot) + f" [Step3] Z={z:.2f} → CLEAR.", "decision": "CLEAR", "final_allocation_pct": 0.0}
    else:
        return {"cot_reasoning": " ".join(cot) + f" [Step3] Z={z:.2f} → HOLD.", "decision": "HOLD", "final_allocation_pct": 0.0}

    cot.append(f"[Step3] Z-Score={z:.2f} → {decision}.")

    fk = kelly_opt.get("full_kelly_fraction") or 0.0
    fm = kelly_opt.get("fractional_multiplier") or _FRAC_KELLY
    alloc = fk * fm
    cvar = risk.get("asset_cvar_99_daily") or 0.0
    limit = risk.get("max_portfolio_cvar_limit") or _MAX_PORTFOLIO_CVAR

    if cvar > 0 and alloc * cvar > limit:
        alloc = limit / cvar

    if defense:
        alloc *= 0.5

    alloc = round(max(alloc, 0.0), 4)
    cot.append(f"[Step4] final_alloc={alloc}.")

    return {"cot_reasoning": " ".join(cot), "decision": decision, "final_allocation_pct": alloc}
