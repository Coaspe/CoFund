"""
tests/test_sentiment_tilt_cap.py — tilt_factor ∈ [0.7, 1.3] 강제 검증
=======================================================================
Iron Rule R5: Sentiment는 sizing tilt만. 하드캡 [0.7, 1.3].
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines.sentiment_engine import compute_sentiment_features
from agents.sentiment_agent import sentiment_analyst_run


def test_tilt_always_in_range():
    """다양한 극단 입력에서도 tilt_factor가 [0.7, 1.3] 범위."""
    extreme_cases = [
        {"put_call_ratio": 2.0, "pcr_percentile_90d": 99, "vix_level": 50, "short_interest_pct": 30, "news_sentiment_score": -0.9},
        {"put_call_ratio": 0.3, "pcr_percentile_90d": 1, "vix_level": 10, "short_interest_pct": 2, "news_sentiment_score": 0.9},
        {"put_call_ratio": 0.8, "pcr_percentile_90d": 50, "vix_level": 20, "short_interest_pct": 15, "news_sentiment_score": 0.0},
        {"vix_level": 80, "news_sentiment_score": -1.0, "upcoming_events": ["FOMC Meeting (1d)", "NFP (2d)"]},
        {},  # empty
    ]
    for i, case in enumerate(extreme_cases):
        features = compute_sentiment_features(case)
        tilt = features["base_tilt_factor"]
        assert 0.7 <= tilt <= 1.3, f"Case {i}: tilt={tilt} OUT OF RANGE [0.7, 1.3]! Input={case}"
        print(f"   Case {i}: tilt={tilt} ✅")


def test_agent_never_rejects():
    """Sentiment agent의 recommendation은 절대 'reject'가 아니어야 함."""
    indicators = {"put_call_ratio": 2.0, "vix_level": 60, "news_sentiment_score": -0.9}
    output = sentiment_analyst_run("TEST", indicators)
    assert output["recommendation"] != "reject", "Sentiment MUST NOT reject (R5)"
    assert output["recommendation"] in ("allow", "allow_with_limits")
    assert 0.7 <= output["tilt_factor"] <= 1.3


def test_vol_crisis_caps_long_tilt():
    """VIX crisis에서 long tilt는 0.9 이하로 제한."""
    indicators = {
        "put_call_ratio": 1.5,  # fear → would push tilt up
        "pcr_percentile_90d": 95,
        "vix_level": 45,  # crisis
        "short_interest_pct": 25,  # short crowded → would push tilt up more
        "news_sentiment_score": -0.8,
    }
    features = compute_sentiment_features(indicators)
    tilt = features["base_tilt_factor"]
    assert tilt <= 0.9, f"Vol crisis should cap tilt ≤ 0.9, got {tilt}"


if __name__ == "__main__":
    test_tilt_always_in_range()
    print("✅ test_tilt_always_in_range PASSED")
    test_agent_never_rejects()
    print("✅ test_agent_never_rejects PASSED")
    test_vol_crisis_caps_long_tilt()
    print("✅ test_vol_crisis_caps_long_tilt PASSED")
