"""Main pipeline orchestrator — runs all 5 stages in sequence."""

from __future__ import annotations

import logging
from datetime import datetime

from config.settings import PipelineConfig
from portfolio_manager.stages.screening import ScreeningStage
from portfolio_manager.stages.adversarial import AdversarialResearchStage
from portfolio_manager.stages.scenario import ScenarioModelingStage
from portfolio_manager.stages.construction import PortfolioConstructionStage
from portfolio_manager.stages.rebalancing import RebalancingStage

logger = logging.getLogger(__name__)


class PortfolioPipeline:
    """Orchestrates the full 5-stage portfolio management pipeline.

    Stages:
        1. Screening — Narrow Russell 1000 to top 50 candidates
        2. Adversarial Research — Bull/bear debate for each candidate
        3. Scenario Modeling — Probability-weighted return models
        4. Portfolio Construction — Build optimal 15-position portfolio
        5. Rebalancing — Ongoing monitoring and trade recommendations
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.portfolio: dict | None = None
        self.bench_candidates: list[dict] = []

    def run_full_pipeline(self) -> dict:
        """Execute the complete pipeline from screening to portfolio construction.

        Returns:
            The constructed portfolio dict.
        """
        start = datetime.now()
        logger.info("=" * 70)
        logger.info("PORTFOLIO PIPELINE — Full Run Starting at %s", start.isoformat())
        logger.info("=" * 70)

        # Stage 1: Screening
        screening = ScreeningStage(config=self.config.screening)
        candidates = screening.run()
        logger.info("Stage 1 produced %d candidates", len(candidates))

        # Stage 2: Adversarial Research
        adversarial = AdversarialResearchStage(
            config=self.config.adversarial,
            model=self.config.model,
        )
        vetted = adversarial.run(candidates)
        logger.info("Stage 2 passed %d candidates", len(vetted))

        # Keep eliminated candidates as bench
        vetted_tickers = {c["ticker"] for c in vetted}
        self.bench_candidates = [c for c in candidates if c["ticker"] not in vetted_tickers]

        # Stage 3: Scenario Modeling
        scenario = ScenarioModelingStage(
            config=self.config.scenario,
            model=self.config.model,
        )
        modeled = scenario.run(vetted)
        logger.info("Stage 3 modeled %d candidates", len(modeled))

        # Stage 4: Portfolio Construction
        construction = PortfolioConstructionStage(
            config=self.config.portfolio,
            portfolio_value=self.config.portfolio_value,
        )
        self.portfolio = construction.run(modeled)

        elapsed = (datetime.now() - start).total_seconds()
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETE — Elapsed: %.1f seconds", elapsed)
        logger.info("=" * 70)

        return self.portfolio

    def run_daily_rebalance(self) -> list[dict]:
        """Run daily rebalance check on the current portfolio.

        Must be called after run_full_pipeline() or after loading a portfolio.

        Returns:
            List of trade recommendations.
        """
        if self.portfolio is None:
            raise RuntimeError("No portfolio loaded. Run run_full_pipeline() first.")

        # Re-screen for fresh bench candidates
        screening = ScreeningStage(config=self.config.screening)
        fresh_candidates = screening.run()

        # Filter out current holdings from bench
        held_tickers = {p["ticker"] for p in self.portfolio.get("positions", [])}
        bench = [c for c in fresh_candidates if c["ticker"] not in held_tickers]
        self.bench_candidates = bench

        # Run rebalance check
        rebalancer = RebalancingStage(
            config=self.config.rebalance,
            portfolio_config=self.config.portfolio,
        )
        return rebalancer.run_daily_check(self.portfolio, bench)

    def run_weekly_report(self) -> str:
        """Generate the weekly portfolio status report.

        Returns:
            Formatted report string.
        """
        if self.portfolio is None:
            raise RuntimeError("No portfolio loaded. Run run_full_pipeline() first.")

        self.portfolio["bench_candidates"] = self.bench_candidates

        rebalancer = RebalancingStage(
            config=self.config.rebalance,
            portfolio_config=self.config.portfolio,
        )
        return rebalancer.generate_weekly_report(self.portfolio)

    def run_screening_only(self) -> list[dict]:
        """Run only Stage 1 screening (useful for quick bench updates).

        Returns:
            List of top 50 candidates.
        """
        screening = ScreeningStage(config=self.config.screening)
        return screening.run()
