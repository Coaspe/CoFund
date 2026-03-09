import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from visualization.webapp import create_app


def _write_demo_run(
    runs_dir: Path,
    run_id: str,
    *,
    request: str,
    ticker: str,
    complete: bool = True,
    created_at: str = "2026-03-08T09:00:00+00:00",
) -> None:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(
        json.dumps({"run_id": run_id, "mode": "mock", "created_at": created_at}, ensure_ascii=False),
        encoding="utf-8",
    )
    if complete:
        (run_dir / "final_state.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "mode": "mock",
                    "user_request": request,
                    "target_ticker": ticker,
                    "analysis_tasks": ["macro", "sentiment"],
                    "risk_assessment": {"grade": "Low", "summary": "All risk gates passed."},
                    "technical_analysis": {"decision": "LONG", "recommendation": "allow"},
                    "final_report": f"# Final Memo — {ticker}\n\n{ticker} looks actionable.",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    events = [
        {"ts": "2026-03-08T10:00:00+00:00", "run_id": run_id, "node_name": "orchestrator", "iteration": 0, "phase": "enter", "inputs_summary": {"ticker": ticker}, "outputs_summary": None, "errors": []},
        {"ts": "2026-03-08T10:00:01+00:00", "run_id": run_id, "node_name": "orchestrator", "iteration": 0, "phase": "exit", "inputs_summary": None, "outputs_summary": {"action": "initial_delegation"}, "errors": []},
        {"ts": "2026-03-08T10:00:02+00:00", "run_id": run_id, "node_name": "macro", "iteration": 1, "phase": "enter", "inputs_summary": {"ticker": ticker}, "outputs_summary": None, "errors": []},
        {"ts": "2026-03-08T10:00:03+00:00", "run_id": run_id, "node_name": "macro", "iteration": 1, "phase": "exit", "inputs_summary": None, "outputs_summary": {"regime": "expansion"}, "errors": []},
    ]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )


def test_agent_empire_webapp_routes(tmp_path: Path, monkeypatch):
    runs_dir = tmp_path / "runs"
    _write_demo_run(
        runs_dir,
        "run-a",
        request="NVDA 이벤트 리플레이",
        ticker="NVDA",
        created_at="2026-03-08T09:00:00+00:00",
    )
    _write_demo_run(
        runs_dir,
        "run-b",
        request="AAPL 리스크 리플레이",
        ticker="AAPL",
        complete=False,
        created_at="2026-03-08T10:00:00+00:00",
    )

    launches: list[dict] = []

    def fake_launch_run(*, question: str, mode: str, seed: int, runs_dir: Path, portfolio_context: dict | None = None) -> dict:
        launches.append(
            {
                "question": question,
                "mode": mode,
                "seed": seed,
                "runs_dir": str(runs_dir),
                "portfolio_context": portfolio_context or {},
            }
        )
        return {"pid": 43210, "log_path": "/tmp/fake-launch.log", "command": ["python", "investment_team.py"]}

    monkeypatch.setattr(
        "visualization.webapp._preview_launch_requirements",
        lambda **kwargs: {
            "question_understanding": {"intent": "single_name", "question_type": "single_name_analysis", "primary_tickers": ["SPY"]},
            "portfolio_intake": {"holdings": [], "missing_fields": []},
            "target_ticker": "SPY",
            "universe": ["SPY"],
            "needs_clarification": False,
            "clarification": {"required": False, "fields": [], "message": "", "target_ticker": "SPY", "currency": "USD"},
        },
    )

    app = create_app(runs_dir=runs_dir, launch_run=fake_launch_run)
    client = TestClient(app)

    response = client.get("/api/runs")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["runs"]) == 2
    assert payload["runs"][0]["run_id"] == "run-b"
    assert payload["runs"][1]["run_id"] == "run-a"
    by_id = {item["run_id"]: item for item in payload["runs"]}
    assert by_id["run-a"]["status"] == "complete"
    assert by_id["run-b"]["status"] == "running"

    response = client.get("/api/runs/run-a")
    assert response.status_code == 200
    model = response.json()
    assert model["run_id"] == "run-a"
    assert model["nodes"]
    assert model["status"] == "complete"

    response = client.get("/api/runs/run-b")
    assert response.status_code == 200
    model = response.json()
    assert model["run_id"] == "run-b"
    assert model["status"] == "running"

    response = client.get("/api/runs/run-a/result")
    assert response.status_code == 200
    result = response.json()
    assert result["available"] is True
    assert "Final Memo" in result["content"]
    assert result["recommendation"] == "allow"

    response = client.get("/runs/run-a")
    assert response.status_code == 200
    assert "Agent Empire Replay" in response.text
    assert "run-a" in response.text

    response = client.get("/")
    assert response.status_code == 200
    assert "FastAPI Control Room" in response.text
    assert "Recent Runs" in response.text
    assert "Launch Run" in response.text
    assert "Final Result" in response.text
    assert '.join("\\n")' in response.text

    response = client.post(
        "/api/launch",
        json={
            "question": "SPY를 분석해줘",
            "mode": "mock",
            "seed": 7,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "accepted"
    assert payload["pid"] == 43210
    assert launches == [
        {
            "question": "SPY를 분석해줘",
            "mode": "mock",
            "seed": 7,
            "runs_dir": str(runs_dir),
            "portfolio_context": {},
        }
    ]

    with client.stream("GET", "/api/live/runs?once=1") as response:
        assert response.status_code == 200
        body = "".join(chunk for chunk in response.iter_text())
    assert "event: snapshot" in body
    assert "\"run-b\"" in body

    with client.stream("GET", "/api/live/runs/run-b?once=1") as response:
        assert response.status_code == 200
        body = "".join(chunk for chunk in response.iter_text())
    assert "event: snapshot" in body
    assert "\"status\": \"running\"" in body


def test_agent_empire_webapp_pending_rate_limit_launch_control(tmp_path: Path, monkeypatch):
    runs_dir = tmp_path / "runs"
    launch_log = tmp_path / "live-launch.log"
    launch_log.write_text("", encoding="utf-8")
    launch_calls: list[dict] = []

    def fake_launch_run(*, question: str, mode: str, seed: int, runs_dir: Path, portfolio_context: dict | None = None) -> dict:
        launch_calls.append(
            {
                "question": question,
                "mode": mode,
                "seed": seed,
                "runs_dir": str(runs_dir),
                "portfolio_context": portfolio_context or {},
            }
        )
        return {
            "pid": 98765,
            "log_path": str(launch_log),
            "command": ["python", "investment_team.py"],
        }

    monkeypatch.setattr(
        "visualization.webapp._preview_launch_requirements",
        lambda **kwargs: {
            "question_understanding": {"intent": "position_review", "question_type": "single_position_review", "primary_tickers": ["NVDA"]},
            "portfolio_intake": {"holdings": [{"ticker": "NVDA", "shares": 10, "avg_cost": 120, "currency": "USD"}], "missing_fields": []},
            "target_ticker": "NVDA",
            "universe": ["NVDA"],
            "needs_clarification": False,
            "clarification": {"required": False, "fields": [], "message": "", "target_ticker": "NVDA", "currency": "USD"},
        },
    )

    app = create_app(runs_dir=runs_dir, launch_run=fake_launch_run)
    client = TestClient(app)

    response = client.post(
        "/api/launch",
        json={
            "question": "NVDA 포지션을 지금 줄여야 할까?",
            "mode": "live",
            "seed": 11,
            "portfolio_context": {
                "holdings": [{"ticker": "NVDA", "shares": 10, "avg_cost": 120, "currency": "USD"}],
            },
        },
    )
    assert response.status_code == 200
    payload = response.json()
    launch_id = payload["launch_id"]

    launch_log.write_text(
        "\n".join(
            [
                "Run ID: 11111111-2222-3333-4444-555555555555",
                "[LLM Router] orchestrator: rate limit 감지 (gpt-oss)",
                "429 Too Many Requests: tokens per minute exceeded",
            ]
        ),
        encoding="utf-8",
    )

    response = client.get("/api/launches")
    assert response.status_code == 200
    launches = response.json()["launches"]
    assert launches[0]["launch_id"] == launch_id
    assert launches[0]["status"] == "pending_rate_limit"
    assert launches[0]["run_id"] == "11111111-2222-3333-4444-555555555555"
    assert launches[0]["resume_strategy"] in {"signal", "relaunch"}
    assert "rate limit" in launches[0]["pending_reason"].lower()
    assert launches[0]["rate_limit_api_type"] == "llm"
    assert launches[0]["rate_limit_api_source"] == "gpt-oss"
    assert launches[0]["rate_limit_api_scope"] == "orchestrator"

    with client.stream("GET", "/api/live/runs?once=1") as response:
        assert response.status_code == 200
        body = "".join(chunk for chunk in response.iter_text())
    assert "\"launches\"" in body
    assert "\"pending_rate_limit\"" in body

    response = client.post(f"/api/launches/{launch_id}/continue")
    assert response.status_code == 200
    resume_payload = response.json()
    assert resume_payload["status"] in {"running", "accepted"}
    assert len(launch_calls) == 2

    response = client.get("/")
    assert response.status_code == 200
    assert ".modal-shell[hidden]" in response.text
    assert "Rate Limit Pending" in response.text
    assert "계속 진행할까요?" in response.text


def test_agent_empire_webapp_position_review_launch_prepare(tmp_path: Path, monkeypatch):
    runs_dir = tmp_path / "runs"

    def fake_preview(question: str, *, portfolio_context: dict | None = None, mode: str = "mock", seed: int | None = 42) -> dict:
        holdings = (portfolio_context or {}).get("holdings") or []
        if holdings:
            return {
                "question_understanding": {"intent": "position_review", "question_type": "single_position_review", "primary_tickers": ["NVDA"]},
                "portfolio_intake": {"holdings": holdings, "missing_fields": []},
                "target_ticker": "NVDA",
                "universe": ["NVDA"],
                "needs_clarification": False,
                "clarification": {"required": False, "fields": [], "message": "", "target_ticker": "NVDA", "currency": "USD"},
            }
        return {
            "question_understanding": {"intent": "position_review", "question_type": "single_position_review", "primary_tickers": ["NVDA"]},
            "portfolio_intake": {"holdings": [], "missing_fields": ["shares", "avg_cost"]},
            "target_ticker": "NVDA",
            "universe": ["NVDA"],
            "needs_clarification": True,
            "clarification": {"required": True, "fields": ["shares", "avg_cost"], "message": "need holdings", "target_ticker": "NVDA", "currency": "USD"},
        }

    monkeypatch.setattr("visualization.webapp._preview_launch_requirements", fake_preview)

    launches: list[dict] = []

    def fake_launch_run(*, question: str, mode: str, seed: int, runs_dir: Path, portfolio_context: dict | None = None) -> dict:
        launches.append({"question": question, "portfolio_context": portfolio_context or {}})
        return {"pid": 1234, "log_path": "/tmp/launch.log", "command": ["python", "investment_team.py"]}

    app = create_app(runs_dir=runs_dir, launch_run=fake_launch_run)
    client = TestClient(app)

    response = client.post("/api/launch/prepare", json={"question": "NVDA 언제 팔까?", "mode": "mock", "seed": 7})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "needs_clarification"
    assert payload["clarification"]["fields"] == ["shares", "avg_cost"]

    response = client.post("/api/launch", json={"question": "NVDA 언제 팔까?", "mode": "mock", "seed": 7})
    assert response.status_code == 409
    assert response.json()["detail"]["status"] == "needs_clarification"

    response = client.post(
        "/api/launch",
        json={
            "question": "NVDA 언제 팔까?",
            "mode": "mock",
            "seed": 7,
            "portfolio_context": {
                "holdings": [{"ticker": "NVDA", "shares": 10, "avg_cost": 120, "currency": "USD"}],
            },
        },
    )
    assert response.status_code == 200
    assert launches == [
        {
            "question": "NVDA 언제 팔까?",
            "portfolio_context": {
                "holdings": [{"ticker": "NVDA", "shares": 10, "avg_cost": 120, "currency": "USD"}],
            },
        }
    ]
