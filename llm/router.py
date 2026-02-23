"""
llm/router.py — Multi-Provider LLM Router
==========================================
에이전트별로 서로 다른 LLM Provider/Model을 라우팅합니다.

Provider 우선순위 (무료 쿼터 중심):
  - Groq    : orchestrator, macro, fundamental, sentiment, quant(opt), risk_manager
  - Gemini  : report_writer (fallback → Groq → Mock)

환경변수:
  GROQ_API_KEY     — Groq 호출에 필수
  GOOGLE_API_KEY   — Gemini 호출에 필수 (또는 GEMINI_API_KEY)
  에이전트별 모델 오버라이드: ORCH_MODEL, RISK_MODEL, REPORT_MODEL 등

토큰/비용 절감:
  - temperature=0 (구조화 노드)
  - max_tokens 기본 낮게 제한
  - in-memory 캐시로 동일 입력 반복 호출 방지
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Optional

# ── 에이전트별 기본 LLM 설정 ──────────────────────────────────────────────

AGENT_LLM_CONFIG: dict[str, dict[str, Any]] = {
    "orchestrator": {
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
        "temperature": 0,
        "max_tokens": 1200,
        "structured": True,
        "enabled": True,
        "env_model_key": "ORCH_MODEL",
    },
    "macro": {
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
        "temperature": 0,
        "max_tokens": 700,
        "structured": True,
        "enabled": True,
        "env_model_key": "MACRO_MODEL",
    },
    "fundamental": {
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
        "temperature": 0,
        "max_tokens": 900,
        "structured": True,
        "enabled": True,
        "env_model_key": "FUNDA_MODEL",
    },
    "sentiment": {
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
        "temperature": 0,
        "max_tokens": 700,
        "structured": True,
        "enabled": True,
        "env_model_key": "SENTI_MODEL",
    },
    "quant": {
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
        "temperature": 0,
        "max_tokens": 400,
        "structured": False,
        "enabled": False,  # 기본 OFF — Python 규칙 기반 의사결정 우선
        "env_model_key": "QUANT_MODEL",
    },
    "risk_manager": {
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
        "temperature": 0,
        "max_tokens": 1200,
        "structured": True,
        "enabled": True,
        "env_model_key": "RISK_MODEL",
    },
    "report_writer": {
        "provider": "gemini",
        "model": "gemini-2.5-flash-lite",
        "temperature": 0.3,
        "max_tokens": 3000,
        "structured": False,
        "enabled": True,
        "env_model_key": "REPORT_MODEL",
        "fallback_model": "gemini-2.5-flash",
    },
}


# ── In-memory 캐시 (iteration 피드백 루프 최적화) ─────────────────────────

_response_cache: dict[str, Any] = {}


def _cache_key(agent_name: str, content: str) -> str:
    """에이전트명 + 입력 콘텐츠 해시로 캐시 키 생성."""
    h = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"{agent_name}:{h}"


def clear_cache() -> None:
    """캐시 전체 초기화 (새 run 시작 시 호출)."""
    _response_cache.clear()


# ── Provider별 API 키 확인 ────────────────────────────────────────────────

def _get_groq_key() -> str:
    return os.environ.get("GROQ_API_KEY", "")


def _get_gemini_key() -> str:
    return os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")


# ── LLM 인스턴스 생성 ────────────────────────────────────────────────────

def _create_groq_llm(config: dict):
    """Groq ChatModel 생성. langchain-groq 필요."""
    key = _get_groq_key()
    if not key:
        return None
    try:
        from langchain_groq import ChatGroq  # type: ignore
        model = os.environ.get(config.get("env_model_key", ""), "") or config["model"]
        return ChatGroq(
            model=model,
            temperature=config.get("temperature", 0),
            max_tokens=config.get("max_tokens", 1000),
            api_key=key,
        )
    except ImportError:
        print("   [LLM Router] ⚠️ langchain-groq 미설치")
        return None
    except Exception as exc:
        print(f"   [LLM Router] ⚠️ Groq 생성 실패: {exc}")
        return None


def _create_gemini_llm(config: dict):
    """Gemini ChatModel 생성. langchain-google-genai 필요."""
    key = _get_gemini_key()
    if not key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        model = os.environ.get(config.get("env_model_key", ""), "") or config["model"]
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=config.get("temperature", 0.3),
            max_output_tokens=config.get("max_tokens", 3000),
            google_api_key=key,
        )
    except ImportError:
        print("   [LLM Router] ⚠️ langchain-google-genai 미설치")
        return None
    except Exception as exc:
        print(f"   [LLM Router] ⚠️ Gemini 생성 실패: {exc}")
        # Fallback: Gemini 실패 → Groq로 시도
        if _get_groq_key():
            print("   [LLM Router] Gemini → Groq fallback")
            fallback_config = dict(config)
            fallback_config["provider"] = "groq"
            fallback_config["model"] = "llama-3.1-8b-instant"
            return _create_groq_llm(fallback_config)
        return None


_PROVIDER_FACTORY = {
    "groq": _create_groq_llm,
    "gemini": _create_gemini_llm,
}


# ── 공개 API ──────────────────────────────────────────────────────────────

def get_llm(agent_name: str):
    """
    에이전트별 LLM ChatModel을 반환합니다.

    Returns:
        BaseChatModel 또는 None (API 키 없거나 disabled)
    """
    config = AGENT_LLM_CONFIG.get(agent_name)
    if config is None:
        print(f"   [LLM Router] ⚠️ 알 수 없는 에이전트: {agent_name}")
        return None

    if not config.get("enabled", True):
        return None

    provider = config.get("provider", "groq")
    factory = _PROVIDER_FACTORY.get(provider)
    if factory is None:
        print(f"   [LLM Router] ⚠️ 알 수 없는 provider: {provider}")
        return None

    llm = factory(config)

    # Gemini-specific fallback chain: Gemini → Groq → None
    if llm is None and provider == "gemini" and _get_groq_key():
        print("   [LLM Router] Gemini 키 없음 → Groq fallback")
        fallback_config = dict(config)
        fallback_config["model"] = "llama-3.1-8b-instant"
        llm = _create_groq_llm(fallback_config)

    if llm is None:
        print(f"   [LLM Router] {agent_name}: API 키 없음 → Mock fallback")

    return llm


def get_llm_with_cache(agent_name: str, cache_content: str):
    """
    캐시 키를 확인하고 캐시 히트면 (llm, cached_response) 반환,
    미스면 (llm, None) 반환.

    Usage:
        llm, cached = get_llm_with_cache("orchestrator", input_text)
        if cached is not None:
            return cached
        if llm is None:
            return mock_fallback()
        response = llm.invoke(msgs)
        set_cache("orchestrator", input_text, response)
    """
    key = _cache_key(agent_name, cache_content)
    cached = _response_cache.get(key)
    if cached is not None:
        print(f"   [LLM Router] {agent_name}: 캐시 히트")
        return None, cached

    llm = get_llm(agent_name)
    return llm, None


def set_cache(agent_name: str, cache_content: str, response: Any) -> None:
    """LLM 응답을 캐시에 저장."""
    key = _cache_key(agent_name, cache_content)
    _response_cache[key] = response


def get_agent_config(agent_name: str) -> dict:
    """에이전트의 LLM 설정을 반환."""
    return dict(AGENT_LLM_CONFIG.get(agent_name, {}))
