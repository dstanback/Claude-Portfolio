"""Tests for portfolio construction logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Mock external dependencies that may not be installed in test env
for mod in ["yfinance", "anthropic", "schedule", "dotenv"]:
    if mod not in sys.modules:
        sys.modules[mod] = type(sys)(mod)

from config.settings import PortfolioConfig

# Import directly to avoid stages/__init__.py transitive imports
from portfolio_manager.stages.construction import PortfolioConstructionStage


def _mock_candidates(n: int = 20) -> list[dict]:
    """Generate mock candidates with scenario data."""
    sectors = ["Technology", "Healthcare", "Financials", "Energy", "Industrials"]
    candidates = []
    for i in range(n):
        ticker = f"STOCK{i}"
        sector = sectors[i % len(sectors)]
        expected_return = 0.15 - (i * 0.005)  # Decreasing returns
        candidates.append({
            "ticker": ticker,
            "company": f"Company {i}",
            "sector": sector,
            "current_price": 100.0 + i,
            "composite_score": 90 - i,
            "hist_volatility": 0.20 + (i * 0.01),
            "beta": 1.0 + (i * 0.05),
            "scenario": {
                "metrics": {
                    "expected_return_12m": expected_return,
                    "risk_reward": 2.5 - (i * 0.1),
                    "alpha_vs_spy": expected_return - 0.10,
                },
            },
            "adversarial": {
                "conviction_score": 50 - i,
            },
        })
    return candidates


def test_selects_max_positions():
    config = PortfolioConfig(max_positions=15)
    stage = PortfolioConstructionStage(config=config, portfolio_value=100_000)
    portfolio = stage.run(_mock_candidates(20))
    assert len(portfolio["positions"]) <= 15


def test_weights_sum_to_one():
    stage = PortfolioConstructionStage(portfolio_value=100_000)
    portfolio = stage.run(_mock_candidates(20))
    total_weight = sum(p["weight"] for p in portfolio["positions"])
    assert abs(total_weight - 1.0) < 0.01, f"Weights sum to {total_weight}"


def test_allocations_sum_to_portfolio_value():
    value = 100_000.0
    stage = PortfolioConstructionStage(portfolio_value=value)
    portfolio = stage.run(_mock_candidates(20))
    total_alloc = sum(p["allocation"] for p in portfolio["positions"])
    assert abs(total_alloc - value) < 1.0, f"Allocations sum to {total_alloc}"


def test_min_position_weight():
    config = PortfolioConfig(min_position_pct=2.0)
    stage = PortfolioConstructionStage(config=config, portfolio_value=100_000)
    portfolio = stage.run(_mock_candidates(20))
    for p in portfolio["positions"]:
        assert p["weight"] >= 0.019, f"{p['ticker']} weight {p['weight']:.3f} below minimum"


def test_max_position_weight():
    config = PortfolioConfig(max_position_pct=12.0)
    stage = PortfolioConstructionStage(config=config, portfolio_value=100_000)
    portfolio = stage.run(_mock_candidates(20))
    for p in portfolio["positions"]:
        assert p["weight"] <= 0.125, f"{p['ticker']} weight {p['weight']:.3f} above maximum"


def test_sector_diversification():
    config = PortfolioConfig(max_sector_pct=35.0)
    stage = PortfolioConstructionStage(config=config, portfolio_value=100_000)
    portfolio = stage.run(_mock_candidates(20))
    sector_weights: dict[str, float] = {}
    for p in portfolio["positions"]:
        sector_weights[p["sector"]] = sector_weights.get(p["sector"], 0) + p["weight"]
    for sector, weight in sector_weights.items():
        assert weight <= 0.36, f"Sector {sector} weight {weight:.3f} exceeds 35%"


def test_positive_expected_return():
    stage = PortfolioConstructionStage(portfolio_value=100_000)
    portfolio = stage.run(_mock_candidates(20))
    for p in portfolio["positions"]:
        assert p["expected_return"] > 0, f"{p['ticker']} has non-positive expected return"


def test_portfolio_metrics_present():
    stage = PortfolioConstructionStage(portfolio_value=100_000)
    portfolio = stage.run(_mock_candidates(20))
    metrics = portfolio["metrics"]
    assert "expected_return" in metrics
    assert "sharpe" in metrics
    assert "beta" in metrics
    assert "num_sectors" in metrics
    assert metrics["num_sectors"] >= 4


def test_empty_candidates():
    stage = PortfolioConstructionStage(portfolio_value=100_000)
    portfolio = stage.run([])
    assert portfolio["positions"] == []
