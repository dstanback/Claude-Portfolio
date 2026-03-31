"""Portfolio management system configuration."""

from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"


@dataclass(frozen=True)
class ScreeningConfig:
    """Stage 1: Screening parameters."""
    top_n: int = 50
    weights: dict[str, float] = field(default_factory=lambda: {
        "financial_quality": 0.25,
        "valuation": 0.20,
        "momentum": 0.20,
        "sentiment": 0.15,
        "catalyst": 0.20,
    })


@dataclass(frozen=True)
class AdversarialConfig:
    """Stage 2: Adversarial research parameters."""
    bull_agents: int = 15
    bear_agents: int = 15
    min_conviction_score: int = 20
    lookback_days: int = 7


@dataclass(frozen=True)
class ScenarioConfig:
    """Stage 3: Scenario modeling parameters."""
    bull_prob_range: tuple[float, float] = (0.20, 0.35)
    base_prob_range: tuple[float, float] = (0.40, 0.55)
    bear_prob_range: tuple[float, float] = (0.15, 0.30)
    horizons_months: tuple[int, ...] = (1, 3, 6, 12)


@dataclass(frozen=True)
class PortfolioConfig:
    """Stage 4: Portfolio construction parameters."""
    max_positions: int = 15
    min_position_pct: float = 2.0
    max_position_pct: float = 12.0
    max_sector_pct: float = 35.0
    min_sectors: int = 4
    benchmark: str = "SPY"


@dataclass(frozen=True)
class RebalanceConfig:
    """Stage 5: Rebalancing parameters."""
    min_conviction_threshold: int = 10
    guidance_cut_threshold: float = -0.10
    rebalance_hour: int = 17
    rebalance_minute: int = 0


@dataclass(frozen=True)
class PipelineConfig:
    """Master configuration for the full pipeline."""
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    adversarial: AdversarialConfig = field(default_factory=AdversarialConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    rebalance: RebalanceConfig = field(default_factory=RebalanceConfig)
    portfolio_value: float = 100_000.00
    model: str = "claude-opus-4-6"
    max_concurrent_agents: int = 10
