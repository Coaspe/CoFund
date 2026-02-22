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
- `runs/`: Output directory for audit logs and final states.
- `tests/`: Comprehensive test suite for engines and agents.
