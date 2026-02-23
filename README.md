# 🚀 7-Agent AI Investment Team (CoFund)

A high-performance, 7-agent AI investment system designed for multi-dimensional stock analysis and risk-managed portfolio allocation.

## 🏛️ Architecture

The system utilizes a modular architecture with clear separation between data retrieval, numerical computation (engines), and narrative interpretation (agents).

### 1. Desk Agents (Analysts)
- **Macro Agent**: Analyzes global economic indicators and regime shifts.
- **Fundamental Agent**: Evaluates company financials, SEC filings, and structural risks.
- **Sentiment Agent**: Processes news sentiment and tilt factors.
- **Quant Agent**: Executes statistical models, Z-score analysis, and CVaR calculations.

### 2. Control Agents
- **Orchestrator (CIO/PM)**: Delegated tasks to desk agents and reconciles feedback.
- **Risk Manager**: Enforces a 5-Gate safety check (Gate 1 to 5) to filter high-risk positions.
- **Report Writer**: Generates IC Memos and "Red Team" analysis for final investment decisions.

### 3. Core Infrastructure
- **Data Hub**: Centralized data provider interface (FRED, NewsAPI, FMP, AlphaVantage).
- **Engines**: Pure Python logic for deterministic computations, isolated from LLM interpretation.
- **Telemetry**: Full auditability with event logging (`events.jsonl`) and state tracking.

## ⚖️ Iron Rules (Enforced)

- **R0 (No Trading)**: Absolutely no broker API or order placement code.
- **R1 (Python-LLM Separation)**: Engines compute numbers; LLMs only interpret narratives.
- **R2 (No Evidence, No Trade)**: Every decision must be backed by a verifiable `evidence[]` array.
- **R3 (Quant Isolation)**: Quant engines perform pure computation without direct data fetching.
- **R4 (Risk Gate Order)**: Risk checks must strictly follow Gates 1 through 5.
- **R5 (Sentiment Cap)**: Sentiment tilt factors are capped between [0.7, 1.3].
- **R6 (Disagreement = Risk)**: High disagreement between agents is flagged as model risk.
- **R7 (Auditability)**: Every run must generate a unique `run_id` and full trace documentation.

## 🛠️ Setup

1.  **Clone the repository**:
    ```bash
    git clone git@github.com:Coaspe/CoFund.git
    cd ai-investment-team
    ```

2.  **Environment Setup**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env  # Add your API keys here
    ```

## 🚀 Usage

Run the investment pipeline in mock mode:
```bash
python investment_team.py --mode mock --seed 42
```

Run in live mode (requires API keys):
```bash
python investment_team.py --mode live
```

## 📂 Project Structure

- `agents/`: LLM-based decision logic for each desk.
- `engines/`: Deterministic numerical computation engines.
- `data_providers/`: API connectors and data normalization.
- `schemas/`: Pydantic models for type-safe state and evidence.
- `llm/`: Multi-provider LLM router (Groq, Gemini).
- `risk/`: Risk gate engine — `risk/engine.py` (fixed 1→5 order) + `risk/gates/` (Gate 1–5 files).
- `portfolio/`: Deterministic multi-ticker portfolio allocator.
- `storage/`: Point-in-Time snapshot store — `pit_store.py`.
- `validators/`: LLM output fact-check — `factcheck.py`.
- `backtest/`: PIT-based reproducible backtest runner.
- `runs/`: Output directory per `run_id` (raw/features/decisions/llm_io/final_report/config).
- `tests/`: 58 tests covering all engines, agents, gates, and invariants.

## 🧪 Running Tests

```bash
./.venv/bin/python -m pytest tests/ -v
```

Key test files:
| File | Purpose |
|------|---------|
| `test_gate_order.py` | Risk Gates run strictly 1→2→3→4→5 |
| `test_gate_parity.py` | New engine matches old compute_risk_decision (decision-level) |
| `test_report_factcheck.py` | T3: LLM report/narrative fact-check + fallback |
| `test_pipeline_reproducibility.py` | T4: same PIT → identical positions_final + gate_trace |
| `test_quant_engine_isolation.py` | T1: quant_engine has no HTTP imports |

## 📊 Backtest

```bash
# Mock mode (deterministic, no API keys needed)
./.venv/bin/python backtest/runner.py \
  --start 2024-01-01 --end 2024-06-30 \
  --universe AAPL MSFT --mode mock --seed 42
```

Output files written to `runs/backtest_{id}/`:
- `backtest_results.csv` — per-period returns, turnover, drawdown
- `backtest_summary.json` — CAGR, Sharpe, MaxDD, config_hash
- `config/config_hash.txt` — reproducibility key

## ✅ Definition of Done (DoD)

| DoD | Status |
|-----|--------|
| DoD1: All agent outputs Pydantic-validated | ✅ `schemas/common.py` BaseAnalystOutput |
| DoD2: `runs/{run_id}/raw/features/decisions/llm_io/final_report/config` | ✅ `storage/pit_store.py` |
| DoD3: Risk gate trace always 1→2→3→4→5 | ✅ `risk/engine.py` GATES fixed array |
| DoD4: Report/narrative fact-check + retry/fallback | ✅ `validators/factcheck.py` |
| DoD5: Backtest runner reproducible PIT results | ✅ `backtest/runner.py` |
| DoD6: T1–T4 pytest all PASS | ✅ 58/58 tests green |
