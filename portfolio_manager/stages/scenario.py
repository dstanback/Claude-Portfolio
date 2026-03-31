"""Stage 3: Scenario Modeling — Probability-weighted return models."""

from __future__ import annotations

import logging
from datetime import datetime

from config.settings import ScenarioConfig
from portfolio_manager.utils.claude_client import run_scenario_analysis
from portfolio_manager.utils.reporting import format_scenario_report, save_report, save_json

logger = logging.getLogger(__name__)

# Assumed annual SPY expected return for alpha calculation
SPY_EXPECTED_ANNUAL_RETURN = 0.10


class ScenarioModelingStage:
    """Stage 3: Build probability-weighted scenario models for each candidate."""

    def __init__(self, config: ScenarioConfig | None = None, model: str = "claude-opus-4-6"):
        self.config = config or ScenarioConfig()
        self.model = model

    def run(self, candidates: list[dict]) -> list[dict]:
        """Generate scenario models for all candidates.

        Args:
            candidates: List of stock data dicts that passed Stage 2.

        Returns:
            Candidates enriched with scenario models and derived metrics.
        """
        logger.info("Stage 3: Scenario Modeling — Modeling %d candidates", len(candidates))
        results = []

        for i, stock in enumerate(candidates, 1):
            ticker = stock["ticker"]
            company = stock.get("company", ticker)
            current_price = stock.get("current_price", 0)
            logger.info("[%d/%d] Scenario model: %s", i, len(candidates), ticker)

            scenario = run_scenario_analysis(
                ticker, company, stock,
                stock.get("adversarial", {}),
                model=self.model,
            )

            # Validate and normalize probabilities
            scenario = self._validate_scenarios(scenario, current_price)

            # Calculate derived metrics
            metrics = self._calculate_metrics(scenario, current_price)
            scenario["metrics"] = metrics
            stock["scenario"] = scenario

            # Log report
            report = format_scenario_report(ticker, company, current_price, scenario)
            logger.info("\n%s", report)
            results.append(stock)

        # Save results
        save_report(
            self._build_summary(results),
            "stage3_scenarios",
        )
        save_json(
            [{"ticker": r["ticker"], "scenario": r["scenario"]} for r in results],
            "scenario_models",
        )

        logger.info("Stage 3 complete: %d scenario models generated", len(results))
        return results

    def _validate_scenarios(self, scenario: dict, current_price: float) -> dict:
        """Ensure scenario probabilities sum to 1 and targets are reasonable."""
        scenarios = scenario.get("scenarios", {})

        # Ensure all three scenarios exist
        for name in ["bull", "base", "bear"]:
            if name not in scenarios:
                scenarios[name] = {
                    "probability": 1 / 3,
                    "targets": {
                        "1m": current_price,
                        "3m": current_price,
                        "6m": current_price,
                        "12m": current_price,
                    },
                    "assumptions": f"Default {name} scenario",
                }

        # Normalize probabilities to sum to 1.0
        total_prob = sum(s.get("probability", 0) for s in scenarios.values())
        if total_prob > 0:
            for s in scenarios.values():
                s["probability"] = s.get("probability", 0) / total_prob
        else:
            for s in scenarios.values():
                s["probability"] = 1 / 3

        # Clamp probabilities to config ranges
        bull_prob = scenarios["bull"]["probability"]
        base_prob = scenarios["base"]["probability"]
        bear_prob = scenarios["bear"]["probability"]

        bull_prob = max(self.config.bull_prob_range[0],
                       min(self.config.bull_prob_range[1], bull_prob))
        base_prob = max(self.config.base_prob_range[0],
                       min(self.config.base_prob_range[1], base_prob))
        bear_prob = max(self.config.bear_prob_range[0],
                       min(self.config.bear_prob_range[1], bear_prob))

        # Re-normalize after clamping
        total = bull_prob + base_prob + bear_prob
        scenarios["bull"]["probability"] = bull_prob / total
        scenarios["base"]["probability"] = base_prob / total
        scenarios["bear"]["probability"] = bear_prob / total

        scenario["scenarios"] = scenarios
        return scenario

    def _calculate_metrics(self, scenario: dict, current_price: float) -> dict:
        """Calculate probability-weighted expected returns and risk metrics."""
        scenarios = scenario.get("scenarios", {})

        if current_price <= 0:
            return {
                "expected_return_12m": 0,
                "expected_downside_12m": 0,
                "risk_reward": 0,
                "alpha_vs_spy": 0,
            }

        # Probability-weighted expected return at 12M
        expected_price_12m = sum(
            s["probability"] * s["targets"].get("12m", current_price)
            for s in scenarios.values()
        )
        expected_return_12m = (expected_price_12m - current_price) / current_price

        # Expected downside (bear case weighted)
        bear = scenarios.get("bear", {})
        bear_target = bear.get("targets", {}).get("12m", current_price)
        expected_downside_12m = (bear_target - current_price) / current_price * bear.get(
            "probability", 0.25
        )

        # Risk/reward ratio
        bull = scenarios.get("bull", {})
        bull_target = bull.get("targets", {}).get("12m", current_price)
        expected_upside_12m = (bull_target - current_price) / current_price * bull.get(
            "probability", 0.25
        )
        risk_reward = (
            abs(expected_upside_12m / expected_downside_12m)
            if expected_downside_12m != 0
            else 10.0
        )

        # Alpha vs SPY
        alpha_vs_spy = expected_return_12m - SPY_EXPECTED_ANNUAL_RETURN

        return {
            "expected_return_12m": expected_return_12m,
            "expected_downside_12m": expected_downside_12m,
            "risk_reward": round(risk_reward, 2),
            "alpha_vs_spy": alpha_vs_spy,
            "expected_price_12m": expected_price_12m,
        }

    def _build_summary(self, results: list[dict]) -> str:
        """Build a summary of all scenario models."""
        lines = [
            f"SCENARIO MODELING SUMMARY — {datetime.now().strftime('%Y-%m-%d')}",
            f"Models generated: {len(results)}",
            "",
            "Ranked by expected 12M return:",
        ]
        ranked = sorted(
            results,
            key=lambda x: x.get("scenario", {}).get("metrics", {}).get("expected_return_12m", 0),
            reverse=True,
        )
        for stock in ranked:
            m = stock.get("scenario", {}).get("metrics", {})
            lines.append(
                f"  {stock['ticker']:6s} | "
                f"E[R] 12M: {m.get('expected_return_12m', 0):+.1%} | "
                f"R/R: {m.get('risk_reward', 0):.2f}x | "
                f"Alpha: {m.get('alpha_vs_spy', 0):+.1%}"
            )
        return "\n".join(lines)
