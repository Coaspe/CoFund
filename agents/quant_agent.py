"""
quant_agent.py — ⑤ Quant Analyst Agent (v2)
=============================================
르네상스 테크놀로지 수준의 StatArb / 팩터 리서처 에이전트.

설계 원칙:
  Python Layer : 공적분, HMM, GARCH, CVaR, Kelly 등 모든 통계 연산 담당
  LLM   Layer  : 통계 요약 JSON 만 받아 4단계 CoT 결정 트리로 포지션 결정

의존 패키지:
  pip install yfinance statsmodels scipy arch hmmlearn \
              langchain-openai langgraph pydantic

실행:
  OPENAI_API_KEY=sk-... python quant_agent.py      # 실제 LLM
  python quant_agent.py                             # Mock 모드
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timedelta
from typing import Any, Literal, Optional, Tuple

import numpy as np

# ─── 선택적 임포트 ──────────────────────────────────────────────────────────
try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

try:
    from statsmodels.tsa.stattools import adfuller, coint
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    HAS_SM = True
except ImportError:
    HAS_SM = False

try:
    from arch import arch_model          # type: ignore
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore
    HAS_HMM = True
except ImportError:
    HAS_HMM = False

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = object  # type: ignore

try:
    from langchain_core.messages import SystemMessage, HumanMessage
    HAS_LC = True
except ImportError:
    HAS_LC = False

from schemas.common import InvestmentState
from llm.router import get_llm, get_agent_config

warnings.filterwarnings("ignore")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 0. 공통 헬퍼
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_LOOKBACK = 504          # 약 2년 영업일
_MARKET = "SPY"


def _fetch(ticker: str, days: int = _LOOKBACK) -> np.ndarray:
    """수정 종가(Adj Close) 1-D float64 배열로 반환. 50행 미만이면 ValueError."""
    if not HAS_YF:
        raise RuntimeError("yfinance 미설치 — Mock 데이터를 사용하세요.")
    end = datetime.today()
    start = end - timedelta(days=int(days * 1.5))
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
    if df.empty or len(df) < 50:
        raise ValueError(f"[{ticker}] 데이터 부족 ({len(df)}행)")
    return df["Close"].dropna().values.flatten().astype(np.float64)[-days:]


def _log_ret(prices: np.ndarray) -> np.ndarray:
    return np.diff(np.log(prices))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 1-A. StatArb: 공적분(Cointegration) + ADF + Z-Score + Half-life
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calculate_statarb_metrics(
    pair_tickers: Tuple[str, str],
    prices_a: Optional[np.ndarray] = None,
    prices_b: Optional[np.ndarray] = None,
) -> dict:
    """
    두 종목의 공적분 관계를 검증하고 스프레드의 평균회귀 특성을 분석합니다.

    파이프라인:
      1. Engle-Granger 공적분 검정 → 헤지 비율(β) 및 p-value
      2. 스프레드 = P_a − β·P_b 에 대해 ADF 검정 → 잔차 정상성 확인
      3. 롤링 Z-Score (window=20) 및 OU 반감기(Half-life) 산출

    Args:
        pair_tickers: (종목A, 종목B) 코드 튜플
        prices_a:     종목A 가격 배열 (None 이면 yfinance 수집)
        prices_b:     종목B 가격 배열 (None 이면 yfinance 수집)

    Returns:
        {
            "pair":          [str, str],
            "hedge_ratio":   float,       # OLS β
            "coint_pvalue":  float,       # Engle-Granger 공적분 p-value
            "adf_pvalue":    float,       # 스프레드 ADF p-value
            "is_stationary": bool,        # adf_pvalue < 0.05
            "current_z_score": float,     # 현재 스프레드 Z-Score
            "half_life_days":  float,     # OU 반감기 (영업일)
            "error":         str | None
        }
    """
    r: dict[str, Any] = {
        "pair": list(pair_tickers),
        "hedge_ratio": None, "coint_pvalue": None,
        "adf_pvalue": None, "is_stationary": None,
        "current_z_score": None, "half_life_days": None,
        "error": None,
    }
    try:
        if not HAS_SM:
            raise RuntimeError("statsmodels 미설치")

        pa = prices_a if prices_a is not None else _fetch(pair_tickers[0])
        pb = prices_b if prices_b is not None else _fetch(pair_tickers[1])

        n = min(len(pa), len(pb))
        pa, pb = pa[-n:], pb[-n:]

        # 1. Engle-Granger 공적분 검정
        score, coint_p, _ = coint(pa, pb)
        r["coint_pvalue"] = round(float(coint_p), 4)

        # 헤지 비율: OLS  P_a = α + β·P_b + ε
        X = add_constant(pb)
        ols = OLS(pa, X).fit()
        beta = float(ols.params[1])
        r["hedge_ratio"] = round(beta, 4)

        # 2. 스프레드 & ADF
        spread = pa - beta * pb
        adf_res = adfuller(spread, autolag="AIC")
        adf_p = float(adf_res[1])
        r["adf_pvalue"] = round(adf_p, 4)
        r["is_stationary"] = adf_p < 0.05

        # 3. Z-Score (롤링 20일)
        window = 20
        roll_mean = np.convolve(spread, np.ones(window) / window, mode="valid")
        roll_std = np.array([
            np.std(spread[i:i + window]) for i in range(len(spread) - window + 1)
        ])
        roll_std = np.where(roll_std < 1e-8, 1e-8, roll_std)
        z = (spread[window - 1:] - roll_mean) / roll_std
        r["current_z_score"] = round(float(z[-1]), 4)

        # 4. OU 반감기: AR(1) OLS  S_t = a + b·S_{t-1}
        S_lag = spread[:-1]
        S_cur = spread[1:]
        X2 = add_constant(S_lag)
        ar = OLS(S_cur, X2).fit()
        b = float(ar.params[1])
        theta = -np.log(abs(b)) if 0 < abs(b) < 1 else 1e-6
        hl = np.log(2) / max(theta, 1e-8)
        r["half_life_days"] = round(float(hl), 2)

    except Exception as exc:
        r["error"] = str(exc)
    return r


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 1-B.  팩터 노출도 (OLS + Newey-West HAC)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calculate_factor_exposures(
    ticker: str,
    prices: Optional[np.ndarray] = None,
    market_prices: Optional[np.ndarray] = None,
) -> dict:
    """
    시장 팩터(SPY) 대비 알파/베타를 OLS 회귀로 추정합니다.
    Newey-West(HAC) 강건 표준오차로 이분산·자기상관을 통제합니다.

    Args:
        ticker:        종목 코드
        prices:        종목 가격 (None → yfinance)
        market_prices: 시장 가격 (None → SPY)

    Returns:
        {
            "alpha_annualized": float,   # 연환산 알파 (%)
            "beta":             float,
            "newey_west_t_stat":float,   # 알파의 NW t-stat
            "p_value":          float,   # 알파의 NW p-value
            "r_squared":        float,
            "error":            str | None
        }
    """
    r: dict[str, Any] = {
        "alpha_annualized": None, "beta": None,
        "newey_west_t_stat": None, "p_value": None,
        "r_squared": None, "error": None,
    }
    try:
        if not HAS_SM:
            raise RuntimeError("statsmodels 미설치")

        p = prices if prices is not None else _fetch(ticker)
        m = market_prices if market_prices is not None else _fetch(_MARKET)

        rs = _log_ret(p)
        rm = _log_ret(m)
        n = min(len(rs), len(rm))
        rs, rm = rs[-n:], rm[-n:]

        X = add_constant(rm)
        ols = OLS(rs, X).fit()
        hac = ols.get_robustcov_results(cov_type="HAC", maxlags=5)

        alpha_d = float(hac.params[0])
        r["alpha_annualized"] = round(alpha_d * 252 * 100, 4)
        r["beta"] = round(float(hac.params[1]), 4)
        r["newey_west_t_stat"] = round(float(hac.tvalues[0]), 4)
        r["p_value"] = round(float(hac.pvalues[0]), 4)
        r["r_squared"] = round(float(hac.rsquared), 4)

    except Exception as exc:
        r["error"] = str(exc)
    return r


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 1-C.  시장 국면(HMM) + 조건부 변동성(GARCH)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_regime_and_volatility(
    ticker: str,
    prices: Optional[np.ndarray] = None,
) -> dict:
    """
    시장 국면(3-State Gaussian HMM) 및 GARCH(1,1) 변동성 예측을 수행합니다.

    Regime 정의 (분산 기준 자동 분류):
      regime_0_calm        — 가장 낮은 분산
      regime_1_mean_revert — 중간 분산
      regime_2_high_vol    — 가장 높은 분산 (위기 국면)

    Args:
        ticker: 종목 코드
        prices: 가격 배열 (None → yfinance)

    Returns:
        {
            "model_used": "GaussianHMM_3_States",
            "state_probabilities": {
                "regime_0_calm":        float,
                "regime_1_mean_revert": float,
                "regime_2_high_vol":    float
            },
            "volatility_forecast": {
                "model": "GARCH(1,1)",
                "t_plus_1_volatility": float   # 연환산 변동성
            },
            "error": str | None
        }
    """
    r: dict[str, Any] = {
        "model_used": "GaussianHMM_3_States",
        "state_probabilities": None,
        "volatility_forecast": {"model": "GARCH(1,1)", "t_plus_1_volatility": None},
        "error": None,
    }
    try:
        p = prices if prices is not None else _fetch(ticker)
        rets = _log_ret(p)

        # ── HMM 3-State ────────────────────────────────────────────────────
        if not HAS_HMM:
            raise RuntimeError("hmmlearn 미설치")

        X_hmm = rets.reshape(-1, 1)
        model = GaussianHMM(
            n_components=3, covariance_type="full",
            n_iter=200, random_state=42, verbose=False,
        )
        model.fit(X_hmm)

        # 분산 기준 정렬: calm(0) < mean_revert(1) < high_vol(2)
        variances = model.covars_.flatten()
        order = np.argsort(variances)
        label_map = {int(order[i]): i for i in range(3)}

        # 마지막 시점의 상태 사후 확률
        _, posteriors = model.score_samples(X_hmm)
        last_post = posteriors[-1]   # shape (3,)
        probs = {
            "regime_0_calm": round(float(last_post[order[0]]), 4),
            "regime_1_mean_revert": round(float(last_post[order[1]]), 4),
            "regime_2_high_vol": round(float(last_post[order[2]]), 4),
        }
        r["state_probabilities"] = probs

        # ── GARCH(1,1) ─────────────────────────────────────────────────────
        if not HAS_ARCH:
            r["error"] = "arch 라이브러리 미설치 — GARCH 생략"
        else:
            am = arch_model(rets * 100, vol="Garch", p=1, q=1, dist="Normal", mean="Constant")
            res = am.fit(disp="off", show_warning=False)
            fc = res.forecast(horizon=1)
            var_1d = float(fc.variance.values[-1, 0])  # %^2
            vol_ann = np.sqrt(var_1d) * np.sqrt(252) / 100  # 연환산 비율
            r["volatility_forecast"]["t_plus_1_volatility"] = round(vol_ann, 4)

    except Exception as exc:
        r["error"] = str(exc)
    return r


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 1-D.  CVaR + Kelly
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_FRAC_KELLY = 0.25
_MAX_PORTFOLIO_CVAR = 0.015   # 포트폴리오 전체 CVaR 상한


def calculate_kelly_and_cvar(
    ticker: str,
    prices: Optional[np.ndarray] = None,
) -> dict:
    """
    과거 수익률 기반 99% CVaR(Expected Shortfall) 및 Kelly 사이징을 산출합니다.

    Args:
        ticker: 종목 코드
        prices: 가격 배열 (None → yfinance)

    Returns:
        {
            "max_portfolio_cvar_limit": float,      # 포트폴리오 CVaR 상한
            "asset_cvar_99_daily":      float,      # 해당 자산 99% 일간 CVaR
            "kelly_optimization": {
                "full_kelly_fraction":    float,
                "fractional_multiplier":  float     # 0.25
            },
            "error": str | None
        }
    """
    r: dict[str, Any] = {
        "max_portfolio_cvar_limit": _MAX_PORTFOLIO_CVAR,
        "asset_cvar_99_daily": None,
        "kelly_optimization": {
            "full_kelly_fraction": None,
            "fractional_multiplier": _FRAC_KELLY,
        },
        "error": None,
    }
    try:
        p = prices if prices is not None else _fetch(ticker)
        rets = _log_ret(p)

        if len(rets) < 50:
            raise ValueError("수익률 데이터 부족 (50일 이상 필요)")

        # 99% CVaR (Historical Simulation)
        var_threshold = np.percentile(rets, 1)
        tail = rets[rets <= var_threshold]
        cvar = float(abs(np.mean(tail))) if len(tail) > 0 else float(abs(var_threshold))
        r["asset_cvar_99_daily"] = round(cvar, 6)

        # Full Kelly: f* = μ / σ²
        mu = float(np.mean(rets))
        sig2 = float(np.var(rets))
        kelly = mu / sig2 if sig2 > 1e-12 else 0.0
        kelly = max(kelly, 0.0)       # 음수 Kelly → 0
        r["kelly_optimization"]["full_kelly_fraction"] = round(kelly, 4)

    except Exception as exc:
        r["error"] = str(exc)
    return r


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 2.  generate_quant_payload — 통합 JSON 래퍼
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_quant_payload(
    ticker: str,
    pair_ticker: str = "MSFT",
    prices: Optional[np.ndarray] = None,
    pair_prices: Optional[np.ndarray] = None,
    market_prices: Optional[np.ndarray] = None,
) -> dict:
    """
    4개 통계 모듈 결과를 모아 LLM에 전달할 단일 JSON을 반환합니다.

    Args:
        ticker:       주 분석 종목 코드
        pair_ticker:  StatArb 쌍 종목 (기본 "MSFT")
        prices:       주 종목 가격 배열 (None → yfinance)
        pair_prices:  쌍 종목 가격 배열 (None → yfinance)
        market_prices: 시장 가격 배열 (None → SPY)

    Returns:
        요구사항에 명시된 JSON 스키마
    """
    print(f"   [퀀트 도구] generate_quant_payload 시작: {ticker}")

    # 가격을 한 번만 수집
    p = prices
    pp = pair_prices
    mp = market_prices

    if p is None and HAS_YF:
        try:
            print(f"   [퀀트 도구] yfinance → {ticker}")
            p = _fetch(ticker)
        except Exception as e:
            print(f"   ⚠️ {e}")

    if pp is None and HAS_YF:
        try:
            print(f"   [퀀트 도구] yfinance → {pair_ticker}")
            pp = _fetch(pair_ticker)
        except Exception as e:
            print(f"   ⚠️ {e}")

    if mp is None and HAS_YF:
        try:
            print(f"   [퀀트 도구] yfinance → {_MARKET}")
            mp = _fetch(_MARKET)
        except Exception as e:
            print(f"   ⚠️ {e}")

    print("   [퀀트 도구] 1/4  StatArb (Engle-Granger + ADF + Z-Score)...")
    sa = calculate_statarb_metrics((ticker, pair_ticker), prices_a=p, prices_b=pp)

    print("   [퀀트 도구] 2/4  Factor Exposures (OLS + Newey-West HAC)...")
    fe = calculate_factor_exposures(ticker, prices=p, market_prices=mp)

    print("   [퀀트 도구] 3/4  Regime & Volatility (HMM + GARCH)...")
    rv = detect_regime_and_volatility(ticker, prices=p)

    print("   [퀀트 도구] 4/4  CVaR & Kelly...")
    kc = calculate_kelly_and_cvar(ticker, prices=p)

    # 엄격한 출력 스키마 매핑
    payload = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "market_regime_context": {
            "model_used": rv.get("model_used", "GaussianHMM_3_States"),
            "state_probabilities": rv.get("state_probabilities") or {
                "regime_0_calm": None,
                "regime_1_mean_revert": None,
                "regime_2_high_vol": None,
            },
            "volatility_forecast": rv.get("volatility_forecast", {}),
        },
        "alpha_signals": {
            "statistical_arbitrage": {
                "adf_test": {
                    "p_value": sa.get("adf_pvalue"),
                    "is_stationary": sa.get("is_stationary"),
                },
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
    any_fatal = any(
        v is not None for v in [sa.get("error"), fe.get("error"), kc.get("error")]
    )
    # regime 오류는 치명적이지 않음 (HMM/GARCH 수렴 실패는 흔함)
    payload["_data_ok"] = not any_fatal
    print(f"   [퀀트 도구] 페이로드 생성 완료 (data_ok={payload['_data_ok']})")
    return payload


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 3-A.  LLM 출력 스키마 (Pydantic)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if HAS_PYDANTIC:
    class QuantDecision(BaseModel):
        """Quant Analyst LLM 출력 스키마 — with_structured_output 으로 강제."""
        cot_reasoning: str = Field(
            description="1~4단계 CoT 판단 과정 서술"
        )
        decision: Literal["LONG", "SHORT", "HOLD", "CLEAR"] = Field(
            description="LONG(매수), SHORT(매도), HOLD(대기), CLEAR(기존 포지션 청산)"
        )
        final_allocation_pct: float = Field(
            ge=0.0, le=1.0,
            description="포트폴리오 대비 최종 비중 (0~1). HOLD/CLEAR는 0.0"
        )
else:
    QuantDecision = dict  # type: ignore[assignment,misc]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 3-B.  System Prompt
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUANT_SYSTEM_PROMPT = """\
당신은 르네상스 테크놀로지 출신의 수석 퀀트 애널리스트입니다.
파이썬 도구가 연산하여 제공한 JSON 요약본만 100% 신뢰하며,
절대 스스로 수학 연산을 시도하지 마십시오.

다음 4단계 Chain-of-Thought(CoT) 결정 트리를 순서대로 평가하여 포지션을 결정하십시오:

[Step 1] 거시 국면 평가
  - market_regime_context.state_probabilities.regime_2_high_vol > 0.50 이면
    → 방어 모드: 비중 축소 또는 HOLD.

[Step 2] 통계적 유의성 검증
  - StatArb: alpha_signals.statistical_arbitrage.adf_test.p_value < 0.05 → 승인.
  - 팩터:   alpha_signals.factor_exposures.newey_west_t_stat 절대값 >= 1.96
            AND p_value < 0.05 일 때만 알파 인정.
  - 둘 다 미충족 시 → HOLD.

[Step 3] 진입/청산 타이밍
  - Z-Score > +2.0  → SHORT
  - Z-Score < -2.0  → LONG
  - |Z-Score| < 0.5 → CLEAR (기존 포지션 청산)
  - 그 외            → HOLD

[Step 4] 사이징 및 리스크 제약
  A) 1차 비중 = full_kelly_fraction × fractional_multiplier
  B) 1차 비중 × asset_cvar_99_daily > max_portfolio_cvar_limit 이면
     → 비중 = max_portfolio_cvar_limit / asset_cvar_99_daily 로 축소.
  C) Step 1에서 방어 모드가 발동되었으면 비중에 × 0.5 추가 적용.

최종 출력은 반드시 아래 JSON 스키마만 사용하라:
{
  "cot_reasoning": "<1~4단계 판단 서술>",
  "decision": "LONG | SHORT | HOLD | CLEAR",
  "final_allocation_pct": <float 0.0 ~ 1.0>
}"""


def _build_human_msg(payload: dict) -> str:
    return (
        f"아래는 {payload.get('timestamp', '')} 기준 퀀트 분석 JSON입니다.\n"
        "위 4단계 CoT를 거쳐 포지션을 결정하세요.\n\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 3-C.  LLM 호출 (실제 / Mock)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _call_llm(payload: dict) -> dict:
    """
    Quant decision pipeline (Invariant A: Python 의사결정 우선):
      1. _mock_decision() → Python 규칙 기반 의사결정 (ALWAYS, source of truth)
      2. (Optional) LLM이 enabled=True이면, payload 기반 "설명만" 생성
      3. LLM은 결정값(decision/final_allocation_pct)을 절대 변경하지 않음
    """
    # Step 1: Python deterministic decision (ALWAYS)
    decision = _mock_decision(payload)

    # Step 2: Optional LLM explanation enrichment
    config = get_agent_config("quant")
    if not config.get("enabled", False):
        return decision

    llm = get_llm("quant")
    if llm is None or not HAS_LC:
        return decision

    try:
        prompt = (
            "Below is a quant analysis payload and a Python-computed decision. "
            "Generate a 2-3 sentence Korean explanation of WHY this decision makes sense. "
            "DO NOT change decision, final_allocation_pct, or any numerical values.\n\n"
            f"Payload (summary): Z-score={payload.get('alpha_signals', {}).get('statistical_arbitrage', {}).get('execution', {}).get('current_z_score')}, "
            f"CVaR={payload.get('portfolio_risk_parameters', {}).get('asset_cvar_99_daily')}\n\n"
            f"Decision: {json.dumps(decision, ensure_ascii=False)}\n\n"
            "Return ONLY the explanation text, nothing else."
        )
        msgs = [HumanMessage(content=prompt)]
        raw = llm.invoke(msgs)
        explanation = raw.content.strip()
        if explanation:
            decision["cot_reasoning"] = decision.get("cot_reasoning", "") + f" [LLM] {explanation}"
    except Exception as exc:
        print(f"   [LLM] ⚠️ 설명 생성 실패 (decision 동일): {exc}")

    return decision


def _mock_decision(payload: dict) -> dict:
    """4단계 CoT를 Python으로 구현한 규칙 기반 Mock."""
    cot = []
    alloc = 0.0

    regime = (payload.get("market_regime_context") or {}).get("state_probabilities") or {}
    sa = ((payload.get("alpha_signals") or {}).get("statistical_arbitrage") or {})
    fe = ((payload.get("alpha_signals") or {}).get("factor_exposures") or {})
    risk = payload.get("portfolio_risk_parameters") or {}
    kelly_opt = risk.get("kelly_optimization") or {}

    # Step 1: 거시 국면
    p_hv = regime.get("regime_2_high_vol") or 0.0
    defense = p_hv > 0.50
    if defense:
        cot.append(f"[Step1] regime_2_high_vol={p_hv:.0%} > 50% → 방어 모드 발동.")
    else:
        cot.append(f"[Step1] regime_2_high_vol={p_hv:.0%} ≤ 50% → 정상 모드.")

    # Step 2: 유의성
    adf_p = ((sa.get("adf_test") or {}).get("p_value"))
    adf_ok = (adf_p is not None) and (adf_p < 0.05)
    nw_t = fe.get("newey_west_t_stat")
    fe_p = fe.get("p_value")
    factor_ok = (nw_t is not None and fe_p is not None
                 and abs(nw_t) >= 1.96 and fe_p < 0.05)
    if adf_ok:
        cot.append(f"[Step2] ADF p={adf_p:.4f} < 0.05 → StatArb 유의.")
    else:
        cot.append(f"[Step2] ADF p={adf_p} — StatArb 비유의.")
    if factor_ok:
        cot.append(f"[Step2] NW t-stat={nw_t:.2f}, p={fe_p:.4f} → 팩터 알파 유의.")
    else:
        cot.append(f"[Step2] 팩터 알파 비유의 (t={nw_t}, p={fe_p}).")

    if not adf_ok and not factor_ok:
        return {
            "cot_reasoning": " ".join(cot) + " → 유의한 알파 없음, HOLD.",
            "decision": "HOLD",
            "final_allocation_pct": 0.0
        }

    # Step 3: 타이밍
    z = ((sa.get("execution") or {}).get("current_z_score")) or 0.0
    if z > 2.0:
        decision = "SHORT"
        cot.append(f"[Step3] Z-Score={z:.2f} > +2.0 → SHORT.")
    elif z < -2.0:
        decision = "LONG"
        cot.append(f"[Step3] Z-Score={z:.2f} < -2.0 → LONG.")
    elif abs(z) < 0.5:
        return {
            "cot_reasoning": " ".join(cot) + f" [Step3] Z-Score={z:.2f} 0 근접 → CLEAR.",
            "decision": "CLEAR",
            "final_allocation_pct": 0.0
        }
    else:
        return {
            "cot_reasoning": " ".join(cot) + f" [Step3] Z-Score={z:.2f} 구간 외 → HOLD.",
            "decision": "HOLD",
            "final_allocation_pct": 0.0
        }

    # Step 4: 사이징
    fk = kelly_opt.get("full_kelly_fraction") or 0.0
    fm = kelly_opt.get("fractional_multiplier") or _FRAC_KELLY
    alloc = fk * fm
    cvar = risk.get("asset_cvar_99_daily") or 0.0
    limit = risk.get("max_portfolio_cvar_limit") or _MAX_PORTFOLIO_CVAR

    cot.append(f"[Step4] 1차 비중 = {fk:.4f}×{fm} = {alloc:.4f}.")

    if cvar > 0 and alloc * cvar > limit:
        alloc_new = limit / cvar
        cot.append(f"[Step4] CVaR 한도 초과 ({alloc:.4f}×{cvar:.4f}={alloc*cvar:.6f} > {limit}) → 비중 축소 {alloc_new:.4f}.")
        alloc = alloc_new

    if defense:
        alloc *= 0.5
        cot.append(f"[Step4] 방어 모드 0.5 적용 → {alloc:.4f}.")

    alloc = round(max(alloc, 0.0), 4)

    return {
        "cot_reasoning": " ".join(cot),
        "decision": decision,
        "final_allocation_pct": alloc,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 3-D.  LangGraph 노드 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def quant_analyst_node(
    state: InvestmentState,
    *,
    _prices: Optional[np.ndarray] = None,
    _pair_prices: Optional[np.ndarray] = None,
    _market_prices: Optional[np.ndarray] = None,
) -> dict:
    """
    ⑤ Quant Analyst LangGraph 노드.

    실행 흐름:
      1. generate_quant_payload()로 통계 페이로드 생성
      2. LLM에 4단계 CoT 프롬프트 + 페이로드 주입
      3. technical_analysis 필드에 결과 저장

    Args:
        state:          InvestmentState
        _prices:        주 종목 가격 (테스트 주입)
        _pair_prices:   쌍 종목 가격 (테스트 주입)
        _market_prices: 시장 가격 (테스트 주입)

    Returns:
        {"technical_analysis": dict}
    """
    ticker = state.get("target_ticker", "AAPL")
    iteration = state.get("iteration_count", 1)

    print(f"\n⑤ QUANT ANALYST  (iteration #{iteration})")
    print(f"   [대상 종목] {ticker}")

    payload = generate_quant_payload(
        ticker, prices=_prices, pair_prices=_pair_prices, market_prices=_market_prices,
    )

    print("   [LLM] 4단계 CoT 의사결정 요청 중...")
    decision = _call_llm(payload)

    result = {
        "quant_payload": payload,
        "llm_decision": decision,
        # flat 필드: Risk Manager가 직접 읽을 수 있도록 상위 노출
        "decision": decision.get("decision"),
        "final_allocation_pct": decision.get("final_allocation_pct"),
        "z_score": (
            (payload.get("alpha_signals") or {})
            .get("statistical_arbitrage", {})
            .get("execution", {})
            .get("current_z_score")
        ),
        "regime_2_high_vol": (
            (payload.get("market_regime_context") or {})
            .get("state_probabilities", {})
            .get("regime_2_high_vol")
        ),
        "asset_cvar_99_daily": (
            (payload.get("portfolio_risk_parameters") or {})
            .get("asset_cvar_99_daily")
        ),
    }

    print(f"   [결과] {decision.get('decision')} | 비중 {decision.get('final_allocation_pct')}")
    print(f"   [CoT]  {(decision.get('cot_reasoning') or '')[:150]}...")

    return {"technical_analysis": result}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# __main__ — Mock 시뮬레이션
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _synth(n: int = 504, mu: float = 150.0, theta: float = 0.05,
           sigma: float = 1.5, seed: int = 42) -> np.ndarray:
    """OU 프로세스로 합성 가격 생성."""
    rng = np.random.default_rng(seed)
    px = [mu]
    for _ in range(n - 1):
        px.append(max(px[-1] + theta * (mu - px[-1]) + rng.normal(0, sigma), 1.0))
    return np.array(px)


if __name__ == "__main__":
    SEP = "=" * 65

    print(SEP)
    print("⑤ QUANT ANALYST v2 — Mock 시뮬레이션")
    print(SEP)

    TICKER = "AAPL"
    PAIR   = "MSFT"
    mock_a   = _synth(504, mu=150, theta=0.05, sigma=1.5, seed=42)
    mock_b   = _synth(504, mu=145, theta=0.04, sigma=1.4, seed=99)
    mock_mkt = _synth(504, mu=400, theta=0.02, sigma=3.0, seed=7)

    print(f"\n[설정] {TICKER}/{PAIR}  |  데이터 {len(mock_a)}pt (합성)")
    print(f"[설정] statsmodels:{HAS_SM}  arch:{HAS_ARCH}  hmmlearn:{HAS_HMM}  LangChain:{HAS_LC}")

    # ── Part 1: 개별 도구 ──────────────────────────────────────────────────
    for title, fn, kwargs in [
        ("Part 1-A: StatArb (Engle-Granger + ADF + Z-Score)",
         calculate_statarb_metrics,
         {"pair_tickers": (TICKER, PAIR), "prices_a": mock_a, "prices_b": mock_b}),
        ("Part 1-B: Factor Exposures (OLS + Newey-West HAC)",
         calculate_factor_exposures,
         {"ticker": TICKER, "prices": mock_a, "market_prices": mock_mkt}),
        ("Part 1-C: Regime & Volatility (HMM + GARCH)",
         detect_regime_and_volatility,
         {"ticker": TICKER, "prices": mock_a}),
        ("Part 1-D: CVaR & Kelly",
         calculate_kelly_and_cvar,
         {"ticker": TICKER, "prices": mock_a}),
    ]:
        print(f"\n{'─' * 65}\n{title}\n{'─' * 65}")
        print(json.dumps(fn(**kwargs), indent=2, ensure_ascii=False))

    # ── Part 2: 통합 페이로드 ──────────────────────────────────────────────
    print(f"\n{'─' * 65}\nPart 2: generate_quant_payload (통합 JSON)\n{'─' * 65}")
    payload = generate_quant_payload(
        TICKER, pair_ticker=PAIR,
        prices=mock_a, pair_prices=mock_b, market_prices=mock_mkt,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    # ── Part 3: LLM 의사결정 ──────────────────────────────────────────────
    print(f"\n{'─' * 65}\nPart 3: LLM 4단계 CoT 의사결정\n{'─' * 65}")
    dec = _call_llm(payload)

    print(f"\n{SEP}\n✅ 최종 QuantDecision JSON\n{SEP}")
    print(json.dumps(dec, indent=2, ensure_ascii=False))

    # ── LangGraph 노드 통합 테스트 ────────────────────────────────────────
    print(f"\n{SEP}\nLangGraph Node 통합 테스트\n{SEP}")
    mock_state: InvestmentState = {
        "user_request": "AAPL 6개월 투자 분석",
        "target_ticker": TICKER,
        "analysis_tasks": ["technical_analysis"],
        "macro_analysis": {}, "fundamental_analysis": {},
        "sentiment_analysis": {}, "technical_analysis": {},
        "risk_assessment": {}, "final_report": "",
        "iteration_count": 1,
    }
    out = quant_analyst_node(
        mock_state, _prices=mock_a, _pair_prices=mock_b, _market_prices=mock_mkt,
    )
    print("\n[노드 반환값 요약]")
    flat = {k: v for k, v in out["technical_analysis"].items()
            if k not in ("quant_payload", "llm_decision")}
    print(json.dumps(flat, indent=2, ensure_ascii=False))
