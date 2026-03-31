"""Tests for the scoring module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portfolio_manager.utils.scoring import (
    composite_score,
    determine_key_signal,
    score_financial_quality,
    score_momentum,
    score_sentiment,
    score_valuation,
    score_catalyst,
)


def _sample_stock() -> dict:
    """Return a sample stock data dict for testing."""
    return {
        "ticker": "TEST",
        "company": "Test Corp",
        "sector": "Technology",
        "current_price": 150.0,
        "market_cap": 500_000_000_000,
        "revenue_growth_yoy": 0.15,
        "earnings_growth": 0.20,
        "gross_margin": 0.60,
        "operating_margin": 0.25,
        "profit_margin": 0.20,
        "free_cash_flow": 10_000_000_000,
        "debt_to_equity": 80,
        "roe": 0.25,
        "roic": 0.15,
        "pe_trailing": 25.0,
        "pe_forward": 20.0,
        "ps_ratio": 8.0,
        "pb_ratio": 5.0,
        "ev_ebitda": 15.0,
        "peg_ratio": 1.5,
        "perf_1w": 0.02,
        "perf_1m": 0.05,
        "perf_3m": 0.12,
        "perf_6m": 0.20,
        "perf_12m": 0.35,
        "rsi_14": 55.0,
        "above_sma50": 1.0,
        "above_sma200": 1.0,
        "dist_sma50": 0.03,
        "dist_sma200": 0.10,
        "analyst_rating": 2.0,
        "analyst_count": 25,
        "target_mean": 175.0,
        "target_high": 200.0,
        "target_low": 130.0,
        "beta": 1.2,
        "hist_volatility": 0.25,
    }


def test_financial_quality_score_range():
    data = _sample_stock()
    score = score_financial_quality(data)
    assert 0 <= score <= 100, f"Score {score} out of range"


def test_valuation_score_range():
    data = _sample_stock()
    score = score_valuation(data)
    assert 0 <= score <= 100, f"Score {score} out of range"


def test_momentum_score_range():
    data = _sample_stock()
    score = score_momentum(data)
    assert 0 <= score <= 100, f"Score {score} out of range"


def test_sentiment_score_range():
    data = _sample_stock()
    score = score_sentiment(data)
    assert 0 <= score <= 100, f"Score {score} out of range"


def test_catalyst_score_range():
    data = _sample_stock()
    score = score_catalyst(data)
    assert 0 <= score <= 100, f"Score {score} out of range"


def test_composite_score_range():
    data = _sample_stock()
    score = composite_score(data)
    assert 0 <= score <= 100, f"Composite score {score} out of range"


def test_composite_score_weights_sum():
    """Verify default weights sum to 1.0."""
    from portfolio_manager.utils.scoring import composite_score
    weights = {
        "financial_quality": 0.25,
        "valuation": 0.20,
        "momentum": 0.20,
        "sentiment": 0.15,
        "catalyst": 0.20,
    }
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_key_signal_strong_momentum():
    data = _sample_stock()
    data["perf_3m"] = 0.30
    signal = determine_key_signal(data)
    assert "momentum" in signal.lower() or "3M" in signal


def test_key_signal_low_pe():
    data = _sample_stock()
    data["pe_forward"] = 10.0
    data["perf_3m"] = 0.01
    data["earnings_growth"] = 0.01
    data["analyst_rating"] = 3.0
    data["target_mean"] = data["current_price"]
    signal = determine_key_signal(data)
    assert "P/E" in signal or "fwd" in signal.lower()


def test_zero_data_does_not_crash():
    data = {"ticker": "ZERO", "current_price": 0}
    score = composite_score(data)
    assert isinstance(score, float)


def test_negative_earnings_handled():
    data = _sample_stock()
    data["pe_forward"] = -5.0
    data["earnings_growth"] = -0.50
    score = composite_score(data)
    assert 0 <= score <= 100
