from __future__ import annotations


def test_force_real_llm_default_under_pytest():
    from llm.router import force_real_llm_in_tests

    assert force_real_llm_in_tests() is True


def test_orchestrator_plan_cache_bypassed_in_test_mode(monkeypatch):
    from agents import orchestrator_agent as orch

    user_request = "AAPL 매수해도 돼?"
    cache_key = orch._plan_cache_key(user_request, 0, None)
    old_value = orch._PLAN_CACHE.get(cache_key)
    orch._PLAN_CACHE[cache_key] = {"from": "cache"}

    monkeypatch.setattr(orch, "HAS_LC", False)
    monkeypatch.setattr(orch, "force_real_llm_in_tests", lambda: True)

    result = orch._call_llm(user_request, 0)
    assert result != {"from": "cache"}
    if old_value is None:
        orch._PLAN_CACHE.pop(cache_key, None)
    else:
        orch._PLAN_CACHE[cache_key] = old_value


def test_waiting_log_is_printed_in_pytest(capfd):
    import threading
    import time

    from llm import router

    class _BlockingBackend:
        def __init__(self, entered: threading.Event, release: threading.Event):
            self._entered = entered
            self._release = release

        def invoke(self, *args, **kwargs):
            self._entered.set()
            self._release.wait(timeout=1.0)
            return {"ok": True}

    old_sem = router._LLM_CONCURRENCY_SEM
    old_max = router._LLM_MAX_CONCURRENCY
    router._LLM_CONCURRENCY_SEM = threading.BoundedSemaphore(1)
    router._LLM_MAX_CONCURRENCY = 1
    try:
        entered = threading.Event()
        release = threading.Event()
        backend = _BlockingBackend(entered, release)
        proxy = router._BoundedLLMProxy("orchestrator", [("zai:glm-4.7-flash", backend)])

        t1 = threading.Thread(target=lambda: proxy.invoke("first"), daemon=True)
        t1.start()
        assert entered.wait(timeout=1.0)

        t2 = threading.Thread(target=lambda: proxy.invoke("second"), daemon=True)
        t2.start()
        time.sleep(0.05)

        captured = capfd.readouterr()
        text = (captured.out or "") + "\n" + (captured.err or "")
        assert "동시 호출 한도(1) 초과로 대기 중" in text
        assert "[LLM Router][TEST] orchestrator: 대기 로그 기록" in text

        release.set()
        t1.join(timeout=1.0)
        t2.join(timeout=1.0)
    finally:
        router._LLM_CONCURRENCY_SEM = old_sem
        router._LLM_MAX_CONCURRENCY = old_max


def test_orchestrator_raw_json_retry_after_structured_failure(monkeypatch):
    from agents import orchestrator_agent as orch

    class _RawResp:
        content = """```json
{
  "current_iteration": 1,
  "action_type": "scale_down",
  "investment_brief": {
    "rationale": "raw retry path",
    "target_universe": ["AAPL"]
  },
  "desk_tasks": {
    "macro": {"horizon_days": 30, "focus_areas": ["x"]},
    "fundamental": {"horizon_days": 30, "focus_areas": ["x"]},
    "sentiment": {"horizon_days": 30, "focus_areas": ["x"]},
    "quant": {"horizon_days": 30, "risk_budget": "Moderate", "focus_areas": ["x"]}
  }
}
```"""

    class _Structured:
        def invoke(self, msgs):
            raise ValueError("Invalid JSON from fenced output")

    class _FakeLLM:
        def with_structured_output(self, schema):
            return _Structured()

        def invoke(self, msgs):
            return _RawResp()

    monkeypatch.setattr(orch, "HAS_LC", True)
    monkeypatch.setattr(orch, "HAS_PYDANTIC", True)
    monkeypatch.setattr(orch, "force_real_llm_in_tests", lambda: True)
    monkeypatch.setattr(orch, "get_llm_with_cache", lambda agent, content: (_FakeLLM(), None))
    monkeypatch.setattr(orch, "set_cache", lambda *args, **kwargs: None)

    result = orch._call_llm("AAPL 매수해도 돼?", 1)
    assert result.get("action_type") == "scale_down"
    assert result.get("investment_brief", {}).get("target_universe") == ["AAPL"]
    assert "macro" in result.get("desk_tasks", {})
    assert "quant" in result.get("desk_tasks", {})


def test_orchestrator_trace_logs_when_enabled(monkeypatch, capfd):
    from agents import orchestrator_agent as orch

    monkeypatch.setenv("ORCH_TRACE", "1")
    monkeypatch.setattr(orch, "HAS_LC", False)
    monkeypatch.setattr(orch, "force_real_llm_in_tests", lambda: False)

    old_cache = dict(orch._PLAN_CACHE)
    try:
        orch._PLAN_CACHE.clear()
        orch._call_llm("미국 증시 전망", 1)
        out = capfd.readouterr().out
        assert "[ORCH TRACE] start iteration=1" in out
        assert "[ORCH TRACE] user_request=미국 증시 전망" in out
        assert "[ORCH TRACE] rules_plan intent=" in out
    finally:
        orch._PLAN_CACHE.clear()
        orch._PLAN_CACHE.update(old_cache)


def test_llm_router_trace_logs_provider_attempt(monkeypatch, capfd):
    from llm import router

    monkeypatch.setenv("LLM_TRACE", "1")

    class _Backend:
        def invoke(self, *args, **kwargs):
            return {"ok": True}

    proxy = router._BoundedLLMProxy("orchestrator", [("zai:glm-4.7-flash", _Backend())])
    proxy.invoke("ping")
    out = capfd.readouterr().out
    assert "[LLM TRACE] orchestrator: invoke 시도 -> zai:glm-4.7-flash" in out
    assert "[LLM TRACE] orchestrator: invoke 성공 <- zai:glm-4.7-flash" in out


def test_llm_router_logs_provider_label_on_api_call(capfd):
    from llm import router

    class _Backend:
        def invoke(self, *args, **kwargs):
            return {"ok": True}

    proxy = router._BoundedLLMProxy("orchestrator", [("zai:glm-4.7-flash", _Backend())])
    proxy.invoke("ping")
    out = capfd.readouterr().out
    assert "[LLM Router] orchestrator: API 호출 (zai:glm-4.7-flash)" in out


def test_global_llm_env_override_zai_glm(monkeypatch):
    from llm import router

    monkeypatch.setenv("LLM_PROVIDER", "ZAI")
    monkeypatch.setenv("LLM_MODEL_NAME", "glm-4.7-flash")
    cfg = dict(router.AGENT_LLM_CONFIG["orchestrator"])
    chain = router._build_provider_chain(cfg)
    assert chain[0]["provider"] == "zai"
    assert router._resolved_model(chain[0]) == "glm-4.7-flash"


def test_global_llm_env_override_cerabras_typo_supported(monkeypatch):
    from llm import router

    monkeypatch.setenv("LLM_PROVIDER", "cerabras")
    monkeypatch.setenv("LLM_MODEL_NAME", "gpt-oss-120b")
    cfg = dict(router.AGENT_LLM_CONFIG["orchestrator"])
    chain = router._build_provider_chain(cfg)
    assert chain[0]["provider"] == "cerebras"
    assert router._resolved_model(chain[0]) == "gpt-oss-120b"
