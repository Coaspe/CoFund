"""
llm/router.py — Multi-Provider LLM Router (bounded concurrency)
================================================================
전역 환경변수로 기본 LLM provider/model을 제어합니다.

핵심 정책:
  - Global default: LLM_PROVIDER + LLM_MODEL_NAME
  - Current default fallback: cerebras + gpt-oss-120b
  - Cerebras rate-limit fallback: qwen-3-235B-A22B-2507
  - Agent-level override: <AGENT>_MODEL (기존 동작 유지)
  - Global concurrent LLM invokes: 기본 1개 (초과 시 대기 + 콘솔 로그)
  - Rate limit 감지 시 콘솔 로그 + fallback 시도
  - pytest 실행 중에는 기본적으로 캐시를 비활성화하여 실호출을 강제
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from api_usage_stats import record_api_request

# Auto-load .env from project root (for python -c / direct module calls)
try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass


# ── Runtime flags ─────────────────────────────────────────────────────────────

def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _is_pytest_running() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST")) or _env_flag("PYTEST_RUNNING", False)


def _llm_trace_enabled() -> bool:
    return _env_flag("LLM_TRACE", False)


def _llm_trace(agent_name: str, message: str) -> None:
    if _llm_trace_enabled():
        print(f"   [LLM TRACE] {agent_name}: {message}", flush=True)


def _force_real_llm_in_tests() -> bool:
    # 테스트에서는 기본적으로 실호출 강제 (캐시 비활성)
    return _env_flag("LLM_FORCE_REAL_IN_TESTS", default=_is_pytest_running())


def force_real_llm_in_tests() -> bool:
    """Public accessor for test-time real LLM policy."""
    return _force_real_llm_in_tests()


def _safe_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _safe_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _normalize_provider_name(provider: str) -> str:
    p = str(provider or "").strip().lower()
    aliases = {
        "z.ai": "zai",
        "zhipu": "zai",
        "cerabras": "cerebras",  # common typo support
    }
    return aliases.get(p, p)


def _global_llm_provider() -> str:
    raw = os.environ.get("LLM_PROVIDER", "").strip()
    if not raw:
        return ""
    return _normalize_provider_name(raw)


def _global_llm_model() -> str:
    return (
        os.environ.get("LLM_MODEL_NAME", "").strip()
        or os.environ.get("LLM_MODEL", "").strip()
    )


def _llm_min_request_interval_sec() -> float:
    """
    Global minimum interval between any LLM API requests.
    Priority: LLM_MIN_REQUEST_INTERVAL_SEC > ZAI_MIN_REQUEST_INTERVAL_SEC > default(2.0s)
    """
    raw = os.environ.get("LLM_MIN_REQUEST_INTERVAL_SEC")
    if raw is None:
        raw = os.environ.get("ZAI_MIN_REQUEST_INTERVAL_SEC")
    if raw is None:
        return 2.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 2.0


_LLM_MAX_CONCURRENCY = max(1, _safe_int_env("LLM_MAX_CONCURRENCY", 1))
_LLM_CONCURRENCY_SEM = threading.BoundedSemaphore(_LLM_MAX_CONCURRENCY)
_ZAI_RL_RETRY_DELAY_SEC = 10
_ZAI_RL_RETRY_MAX = 1
_ZAI_REQUEST_TIMEOUT_SEC = max(5.0, _safe_float_env("ZAI_REQUEST_TIMEOUT_SEC", 180.0))
_ZAI_MAX_TOKENS = max(1, _safe_int_env("ZAI_MAX_TOKENS", 5000))
_LLM_REQUEST_TIMEOUT_SEC = max(5.0, _safe_float_env("LLM_REQUEST_TIMEOUT_SEC", 180.0))
_LLM_TRACE_PREVIEW_CHARS = max(80, _safe_int_env("LLM_TRACE_PREVIEW_CHARS", 260))
_LLM_MIN_REQUEST_INTERVAL_SEC = max(0.0, _llm_min_request_interval_sec())
_LLM_REQUEST_GUARD_LOCK = threading.Lock()
_LLM_LAST_REQUEST_TS = 0.0
_CEREBRAS_RL_FALLBACK_MODEL = "qwen-3-235B-A22B-2507"


# ── Agent config ──────────────────────────────────────────────────────────────

AGENT_LLM_CONFIG: dict[str, dict[str, Any]] = {
    "orchestrator": {
        "provider": "cerebras",
        "model": "gpt-oss-120b",
        "temperature": 0,
        "max_tokens": 5000,
        "structured": True,
        "enabled": True,
        "env_model_key": "ORCH_MODEL",
        "fallback_chain": [
            {"provider": "groq", "model": "llama-3.3-70b-versatile"},
            {"provider": "gemini", "model": "gemini-2.5-flash"},
        ],
    },
    "macro": {
        "provider": "cerebras",
        "model": "gpt-oss-120b",
        "temperature": 0,
        "max_tokens": 5000,
        "structured": True,
        "enabled": True,
        "env_model_key": "MACRO_MODEL",
        "fallback_chain": [
            {"provider": "groq", "model": "llama-3.1-8b-instant"},
            {"provider": "gemini", "model": "gemini-2.5-flash"},
        ],
    },
    "fundamental": {
        "provider": "cerebras",
        "model": "gpt-oss-120b",
        "temperature": 0,
        "max_tokens": 5000,
        "structured": True,
        "enabled": True,
        "env_model_key": "FUNDA_MODEL",
        "fallback_chain": [
            {"provider": "groq", "model": "llama-3.1-8b-instant"},
            {"provider": "gemini", "model": "gemini-2.5-flash"},
        ],
    },
    "sentiment": {
        "provider": "cerebras",
        "model": "gpt-oss-120b",
        "temperature": 0,
        "max_tokens": 5000,
        "structured": True,
        "enabled": True,
        "env_model_key": "SENTI_MODEL",
        "fallback_chain": [
            {"provider": "groq", "model": "llama-3.1-8b-instant"},
            {"provider": "gemini", "model": "gemini-2.5-flash"},
        ],
    },
    "quant": {
        "provider": "cerebras",
        "model": "gpt-oss-120b",
        "temperature": 0,
        "max_tokens": 5000,
        "structured": False,
        "enabled": False,  # 기본 OFF — Python 규칙 기반 의사결정 우선
        "env_model_key": "QUANT_MODEL",
        "fallback_chain": [
            {"provider": "groq", "model": "llama-3.1-8b-instant"},
        ],
    },
    "risk_manager": {
        "provider": "cerebras",
        "model": "gpt-oss-120b",
        "temperature": 0,
        "max_tokens": 5000,
        "structured": True,
        "enabled": True,
        "env_model_key": "RISK_MODEL",
        "fallback_chain": [
            {"provider": "groq", "model": "llama-3.3-70b-versatile"},
            {"provider": "gemini", "model": "gemini-2.5-flash"},
        ],
    },
    "report_writer": {
        "provider": "cerebras",
        "model": "gpt-oss-120b",
        "temperature": 0.3,
        "max_tokens": 5000,
        "structured": False,
        "enabled": True,
        "env_model_key": "REPORT_MODEL",
        "fallback_chain": [
            {"provider": "gemini", "model": "gemini-2.5-flash-lite"},
            {"provider": "groq", "model": "llama-3.1-8b-instant"},
        ],
    },
}


# ── Response cache ────────────────────────────────────────────────────────────

_response_cache: dict[str, Any] = {}


def _cache_key(agent_name: str, content: str) -> str:
    h = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"{agent_name}:{h}"


def clear_cache() -> None:
    _response_cache.clear()


# ── Provider keys ─────────────────────────────────────────────────────────────

def _get_zai_key() -> str:
    return (
        os.environ.get("ZAI_API_KEY", "")
        or os.environ.get("ZHIPU_API_KEY", "")
        or os.environ.get("BIGMODEL_API_KEY", "")
    )


def _get_cerebras_key() -> str:
    return (
        os.environ.get("CEREBRAS_API_KEY", "")
        or os.environ.get("CERABRAS_API_KEY", "")  # common typo support
    )


def _get_groq_key() -> str:
    return os.environ.get("GROQ_API_KEY", "")


def _get_gemini_key() -> str:
    return os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")


# ── Provider constructors ─────────────────────────────────────────────────────

def _resolved_model(config: dict[str, Any]) -> str:
    forced = str(config.get("_forced_model", "")).strip()
    if forced:
        return forced
    env_key = str(config.get("env_model_key", "")).strip()
    if env_key:
        override = os.environ.get(env_key, "").strip()
        if override:
            return override
    global_override = _global_llm_model()
    if global_override:
        return global_override
    return str(config.get("model", "")).strip()


def _create_cerebras_llm(config: dict[str, Any]):
    key = _get_cerebras_key()
    if not key:
        return None
    try:
        from langchain_openai import ChatOpenAI  # type: ignore

        model = _resolved_model(config) or "gpt-oss-120b"
        base_url = os.environ.get("CEREBRAS_BASE_URL", "").strip() or "https://api.cerebras.ai/v1"
        return ChatOpenAI(
            model=model,
            temperature=config.get("temperature", 0),
            max_tokens=config.get("max_tokens", 5000),
            api_key=key,
            base_url=base_url,
            request_timeout=_LLM_REQUEST_TIMEOUT_SEC,
        )
    except ImportError:
        print("   [LLM Router] ⚠️ langchain-openai 미설치", flush=True)
        return None
    except Exception as exc:
        print(f"   [LLM Router] ⚠️ Cerebras 생성 실패: {exc}", flush=True)
        return None


def _create_zai_llm(config: dict[str, Any]):
    key = _get_zai_key()
    if not key:
        return None
    try:
        from langchain_openai import ChatOpenAI  # type: ignore

        model = _resolved_model(config) or "glm-4.7-flash"
        base_url = os.environ.get("ZAI_BASE_URL", "").strip() or "https://api.z.ai/api/paas/v4"
        return ChatOpenAI(
            model=model,
            temperature=config.get("temperature", 0),
            max_tokens=_ZAI_MAX_TOKENS,
            api_key=key,
            base_url=base_url,
            request_timeout=_ZAI_REQUEST_TIMEOUT_SEC,
        )
    except ImportError:
        print("   [LLM Router] ⚠️ langchain-openai 미설치", flush=True)
        return None
    except Exception as exc:
        print(f"   [LLM Router] ⚠️ Z.ai 생성 실패: {exc}", flush=True)
        return None


def _create_groq_llm(config: dict[str, Any]):
    key = _get_groq_key()
    if not key:
        return None
    try:
        from langchain_groq import ChatGroq  # type: ignore

        model = _resolved_model(config)
        return ChatGroq(
            model=model,
            temperature=config.get("temperature", 0),
            max_tokens=config.get("max_tokens", 1000),
            api_key=key,
            request_timeout=_LLM_REQUEST_TIMEOUT_SEC,
        )
    except ImportError:
        print("   [LLM Router] ⚠️ langchain-groq 미설치", flush=True)
        return None
    except Exception as exc:
        print(f"   [LLM Router] ⚠️ Groq 생성 실패: {exc}", flush=True)
        return None


def _create_gemini_llm(config: dict[str, Any]):
    key = _get_gemini_key()
    if not key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

        model = _resolved_model(config)
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=config.get("temperature", 0.3),
            max_output_tokens=config.get("max_tokens", 3000),
            google_api_key=key,
            timeout=_LLM_REQUEST_TIMEOUT_SEC,
        )
    except ImportError:
        print("   [LLM Router] ⚠️ langchain-google-genai 미설치", flush=True)
        return None
    except Exception as exc:
        print(f"   [LLM Router] ⚠️ Gemini 생성 실패: {exc}", flush=True)
        return None


_PROVIDER_FACTORY = {
    "cerebras": _create_cerebras_llm,
    "zai": _create_zai_llm,
    "groq": _create_groq_llm,
    "gemini": _create_gemini_llm,
}


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    keys = (
        "429",
        "rate limit",
        "too many requests",
        "quota",
        "resource exhausted",
        "tokens per minute",
        "tpm",
        "rpm",
    )
    return any(k in text for k in keys)


def _provider_label(config: dict[str, Any]) -> str:
    provider = _normalize_provider_name(str(config.get("provider", "unknown")))
    model = _resolved_model(config) or str(config.get("model", ""))
    return f"{provider}:{model}"


def _apply_llm_request_interval_guard(agent_name: str, label: str) -> None:
    global _LLM_LAST_REQUEST_TS
    if _LLM_MIN_REQUEST_INTERVAL_SEC <= 0:
        return

    wait_sec = 0.0
    with _LLM_REQUEST_GUARD_LOCK:
        now = time.monotonic()
        elapsed = now - _LLM_LAST_REQUEST_TS
        if elapsed < _LLM_MIN_REQUEST_INTERVAL_SEC:
            wait_sec = _LLM_MIN_REQUEST_INTERVAL_SEC - elapsed
        _LLM_LAST_REQUEST_TS = now + wait_sec

    if wait_sec > 0:
        print(
            f"   [LLM Router] {agent_name}: LLM 호출 간격 보호 대기 {wait_sec:.2f}초 ({label})",
            flush=True,
        )
        time.sleep(wait_sec)


def _to_trace_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text") or item.get("input_text") or item.get("content") or item.get("type")
                if text is not None:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(value)


def _preview_text(text: str, max_len: int | None = None) -> str:
    cap = int(max_len or _LLM_TRACE_PREVIEW_CHARS)
    s = " ".join(str(text).split())
    if len(s) <= cap:
        return s
    return s[: cap - 3].rstrip() + "..."


def _invoke_payload_trace_lines(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[str]:
    kw_keys = sorted(kwargs.keys())
    if not args:
        return [f"payload args=0 kwargs={kw_keys}"]

    first = args[0]
    if isinstance(first, list):
        lines = [f"payload messages={len(first)} kwargs={kw_keys}"]
        for i, msg in enumerate(first[:4]):
            role = str(getattr(msg, "type", msg.__class__.__name__)).lower()
            content = _to_trace_text(getattr(msg, "content", msg))
            lines.append(
                f"msg[{i}] role={role} chars={len(content)} preview={_preview_text(content)}"
            )
        if len(first) > 4:
            lines.append(f"msg[...]=+{len(first)-4} more")
        return lines

    content = _to_trace_text(first)
    return [
        f"payload arg0_type={type(first).__name__} kwargs={kw_keys}",
        f"payload chars={len(content)} preview={_preview_text(content)}",
    ]


def _invoke_response_trace_line(out: Any) -> str:
    content = _to_trace_text(getattr(out, "content", out))
    return f"response chars={len(content)} preview={_preview_text(content)}"


class _BoundedLLMProxy:
    """Global concurrency bound + fallback invoke wrapper."""

    def __init__(self, agent_name: str, backends: list[tuple[str, Any]]):
        self._agent_name = agent_name
        self._backends = list(backends)

    def _acquire_slot(self) -> None:
        if _LLM_CONCURRENCY_SEM.acquire(blocking=False):
            return
        msg = (
            f"   [LLM Router] {self._agent_name}: 동시 호출 한도({_LLM_MAX_CONCURRENCY}) "
            f"초과로 대기 중..."
        )
        print(msg, flush=True)
        if _is_pytest_running():
            print(
                f"   [LLM Router][TEST] {self._agent_name}: 대기 로그 기록",
                flush=True,
            )
        _LLM_CONCURRENCY_SEM.acquire()

    def _release_slot(self) -> None:
        try:
            _LLM_CONCURRENCY_SEM.release()
        except ValueError:
            pass

    def _invoke_with_fallback(self, method_name: str, *args, **kwargs):
        self._acquire_slot()
        last_exc: Exception | None = None
        all_rate_limited = True
        try:
            for idx, (label, backend) in enumerate(self._backends):
                method = getattr(backend, method_name, None)
                if method is None:
                    all_rate_limited = False
                    continue
                zai_rl_retry = 0
                while True:
                    _apply_llm_request_interval_guard(self._agent_name, label)
                    _llm_trace(self._agent_name, f"{method_name} 시도 -> {label}")
                    if _llm_trace_enabled():
                        for line in _invoke_payload_trace_lines(args, kwargs):
                            _llm_trace(self._agent_name, line)
                    print(
                        f"   [LLM Router] {self._agent_name}: API 호출 ({label})",
                        flush=True,
                    )
                    try:
                        out = method(*args, **kwargs)
                        record_api_request(label, success=True, category="llm")
                        _llm_trace(self._agent_name, f"{method_name} 성공 <- {label}")
                        if _llm_trace_enabled():
                            _llm_trace(self._agent_name, _invoke_response_trace_line(out))
                        return out
                    except Exception as exc:
                        record_api_request(label, success=False, category="llm")
                        last_exc = exc
                        is_rl = _is_rate_limit_error(exc)
                        if is_rl:
                            print(
                                f"   [LLM Router] {self._agent_name}: rate limit 감지 ({label})",
                                flush=True,
                            )
                            if label.startswith("zai:") and zai_rl_retry < _ZAI_RL_RETRY_MAX:
                                zai_rl_retry += 1
                                print(
                                    f"   [LLM Router] {self._agent_name}: "
                                    f"GLM 재시도 대기 {_ZAI_RL_RETRY_DELAY_SEC}초 "
                                    f"({zai_rl_retry}/{_ZAI_RL_RETRY_MAX})",
                                    flush=True,
                                )
                                time.sleep(_ZAI_RL_RETRY_DELAY_SEC)
                                continue
                        else:
                            all_rate_limited = False
                            print(
                                f"   [LLM Router] {self._agent_name}: 호출 실패 ({label}) - {exc}",
                                flush=True,
                            )
                        if idx < len(self._backends) - 1:
                            nxt = self._backends[idx + 1][0]
                            print(
                                f"   [LLM Router] {self._agent_name}: fallback 시도 → {nxt}",
                                flush=True,
                            )
                        break
            if all_rate_limited and last_exc is not None:
                print("   [LLM Router] ⚠️ 모든 LLM API가 rate limit 상태입니다.", flush=True)
            if last_exc is not None:
                raise last_exc
            raise RuntimeError(f"{self._agent_name}: invoke 가능한 LLM backend가 없습니다.")
        finally:
            self._release_slot()

    def invoke(self, *args, **kwargs):
        return self._invoke_with_fallback("invoke", *args, **kwargs)

    def with_structured_output(self, *args, **kwargs):
        structured_backends: list[tuple[str, Any]] = []
        errors = []
        for label, backend in self._backends:
            try:
                structured_backends.append((label, backend.with_structured_output(*args, **kwargs)))
            except Exception as exc:
                errors.append(f"{label}:{exc}")
        if not structured_backends:
            msg = "; ".join(errors) if errors else "unknown"
            raise RuntimeError(
                f"{self._agent_name}: structured output backend 생성 실패 ({msg})"
            )
        return _BoundedLLMProxy(self._agent_name, structured_backends)

    def __getattr__(self, name: str):
        # invoke/with_structured_output 외 속성은 첫 backend로 위임.
        return getattr(self._backends[0][1], name)


def _build_provider_chain(config: dict[str, Any]) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    primary = dict(config)
    primary.pop("fallback_chain", None)
    provider_override = _global_llm_provider()
    if provider_override:
        primary["provider"] = provider_override
    chain.append(primary)
    primary_provider = _normalize_provider_name(str(primary.get("provider", "")))
    primary_model = _resolved_model(primary).lower()
    if primary_provider == "cerebras" and primary_model == "gpt-oss-120b":
        # Cerebras gpt-oss-120b가 rate limit일 때 사용할 동일 provider 모델 fallback
        rl_fallback = dict(primary)
        rl_fallback["model"] = _CEREBRAS_RL_FALLBACK_MODEL
        rl_fallback["_forced_model"] = _CEREBRAS_RL_FALLBACK_MODEL
        chain.append(rl_fallback)
    # TEMP: fallback chain 비활성화 (GLM 단일 경로 강제)
    # for fb in config.get("fallback_chain", []) or []:
    #     if not isinstance(fb, dict):
    #         continue
    #     merged = dict(config)
    #     merged.update(fb)
    #     merged.pop("fallback_chain", None)
    #     chain.append(merged)
    return chain


# ── Public API ────────────────────────────────────────────────────────────────

def get_llm(agent_name: str):
    """
    에이전트별 LLM ChatModel proxy를 반환.
    - Primary: AGENT_LLM_CONFIG + (optional) global env override
    - Fallback: AGENT_LLM_CONFIG[fallback_chain]
    """
    config = AGENT_LLM_CONFIG.get(agent_name)
    if config is None:
        print(f"   [LLM Router] ⚠️ 알 수 없는 에이전트: {agent_name}", flush=True)
        return None

    if not config.get("enabled", True):
        return None

    backends: list[tuple[str, Any]] = []
    for candidate in _build_provider_chain(config):
        provider = _normalize_provider_name(str(candidate.get("provider", "zai")))
        factory = _PROVIDER_FACTORY.get(provider)
        if factory is None:
            print(f"   [LLM Router] ⚠️ 알 수 없는 provider: {provider}", flush=True)
            continue
        llm = factory(candidate)
        if llm is None:
            continue
        backends.append((_provider_label(candidate), llm))

    if not backends:
        msg = f"   [LLM Router] {agent_name}: 사용 가능한 LLM provider가 없습니다."
        print(msg, flush=True)
        if _force_real_llm_in_tests():
            raise RuntimeError(f"{agent_name}: 실호출 테스트 모드에서 사용 가능한 LLM provider 없음")
        print(f"{msg} → Mock fallback", flush=True)
        return None

    _llm_trace(agent_name, "provider chain = " + " -> ".join(label for label, _ in backends))

    return _BoundedLLMProxy(agent_name, backends)


def get_llm_with_cache(agent_name: str, cache_content: str):
    """
    캐시 조회 후 (llm, cached_response) 반환.
    테스트 실호출 모드에서는 캐시를 사용하지 않는다.
    """
    if _force_real_llm_in_tests():
        llm = get_llm(agent_name)
        return llm, None

    key = _cache_key(agent_name, cache_content)
    cached = _response_cache.get(key)
    if cached is not None:
        print(f"   [LLM Router] {agent_name}: 캐시 히트", flush=True)
        return None, cached

    llm = get_llm(agent_name)
    return llm, None


def set_cache(agent_name: str, cache_content: str, response: Any) -> None:
    if _force_real_llm_in_tests():
        return
    key = _cache_key(agent_name, cache_content)
    _response_cache[key] = response


def get_agent_config(agent_name: str) -> dict[str, Any]:
    return dict(AGENT_LLM_CONFIG.get(agent_name, {}))
