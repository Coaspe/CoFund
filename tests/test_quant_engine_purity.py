"""
tests/test_quant_engine_purity.py — Quant Engine 순수성 검증
=============================================================
Iron Rule R3: quant_engine.py는 yfinance/외부 데이터 수집 코드 금지.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_no_data_imports():
    """quant_engine.py가 yfinance, requests, ccxt 등 데이터 수집 라이브러리를 import하지 않음."""
    engine_file = Path(__file__).resolve().parent.parent / "engines" / "quant_engine.py"
    source = engine_file.read_text()

    banned = ["import yfinance", "import ccxt", "import fredapi", "import requests",
              "from yfinance", "from ccxt", "from fredapi", "from requests"]
    for b in banned:
        assert b not in source, f"quant_engine.py MUST NOT contain '{b}' (R3)"


def test_payload_with_injected_arrays():
    """합성 배열을 주입하여 payload 정상 생성 확인."""
    from engines.quant_engine import generate_quant_payload

    rng = np.random.default_rng(42)
    prices = np.cumsum(rng.normal(0, 1, 300)) + 150
    pair = np.cumsum(rng.normal(0, 1, 300)) + 140
    market = np.cumsum(rng.normal(0, 1, 300)) + 4000

    payload = generate_quant_payload("TEST", prices, pair, market)

    assert "alpha_signals" in payload
    assert "portfolio_risk_parameters" in payload
    assert "market_regime_context" in payload
    assert payload.get("_data_ok") is not None


def test_mock_decision_pure():
    """mock_quant_decision이 외부 호출 없이 결정을 반환."""
    from engines.quant_engine import generate_quant_payload, mock_quant_decision

    rng = np.random.default_rng(99)
    prices = np.cumsum(rng.normal(0, 1, 300)) + 150
    pair = np.cumsum(rng.normal(0, 1, 300)) + 140
    market = np.cumsum(rng.normal(0, 1, 300)) + 4000

    payload = generate_quant_payload("TEST", prices, pair, market)
    decision = mock_quant_decision(payload)

    assert "decision" in decision
    assert decision["decision"] in ("LONG", "SHORT", "HOLD", "CLEAR")
    assert "final_allocation_pct" in decision
    assert isinstance(decision["final_allocation_pct"], float)


if __name__ == "__main__":
    test_no_data_imports()
    print("✅ test_no_data_imports PASSED")
    test_payload_with_injected_arrays()
    print("✅ test_payload_with_injected_arrays PASSED")
    test_mock_decision_pure()
    print("✅ test_mock_decision_pure PASSED")
