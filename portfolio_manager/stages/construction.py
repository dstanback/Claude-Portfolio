"""Stage 4: Portfolio Construction — Build optimal 15-position portfolio."""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from config.settings import PortfolioConfig
from portfolio_manager.utils.reporting import format_portfolio_report, save_report, save_json

logger = logging.getLogger(__name__)


class PortfolioConstructionStage:
    """Stage 4: Select and weight 15 positions with sector/risk constraints."""

    def __init__(
        self,
        config: PortfolioConfig | None = None,
        portfolio_value: float = 100_000.0,
    ):
        self.config = config or PortfolioConfig()
        self.portfolio_value = portfolio_value

    def run(self, candidates: list[dict]) -> dict:
        """Construct the optimal portfolio from scenario-modeled candidates.

        Args:
            candidates: List of stock data dicts with scenario models from Stage 3.

        Returns:
            Portfolio dict with positions, allocations, and aggregate metrics.
        """
        logger.info(
            "Stage 4: Portfolio Construction — Selecting from %d candidates",
            len(candidates),
        )

        # Step 1: Rank candidates by composite optimization score
        ranked = self._rank_candidates(candidates)

        # Step 2: Select positions respecting constraints
        positions = self._select_positions(ranked)

        # Step 3: Calculate weights
        positions = self._calculate_weights(positions)

        # Step 4: Calculate dollar allocations
        for p in positions:
            p["allocation"] = round(p["weight"] * self.portfolio_value, 2)

        # Step 5: Calculate portfolio-level metrics
        metrics = self._calculate_portfolio_metrics(positions)

        portfolio = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_value": self.portfolio_value,
            "positions": positions,
            "metrics": metrics,
        }

        # Generate and save report
        report = format_portfolio_report(portfolio["date"], portfolio)
        logger.info("\n%s", report)
        save_report(report, "stage4_construction")
        save_json(portfolio, "portfolio")

        logger.info(
            "Stage 4 complete: %d positions, expected return %+.1f%%",
            len(positions),
            metrics.get("expected_return", 0) * 100,
        )
        return portfolio

    def _rank_candidates(self, candidates: list[dict]) -> list[dict]:
        """Rank candidates by a combined optimization score."""
        for c in candidates:
            scenario = c.get("scenario", {})
            metrics = scenario.get("metrics", {})
            adversarial = c.get("adversarial", {})

            # Multi-factor ranking score
            expected_return = metrics.get("expected_return_12m", 0)
            risk_reward = metrics.get("risk_reward", 0)
            conviction = adversarial.get("conviction_score", 0)

            # Normalize and combine
            c["_opt_score"] = (
                expected_return * 40  # Return contribution
                + min(risk_reward, 5) * 8  # Risk/reward (capped)
                + conviction * 0.2  # Conviction contribution
            )

        candidates.sort(key=lambda x: x["_opt_score"], reverse=True)
        return candidates

    def _select_positions(self, ranked: list[dict]) -> list[dict]:
        """Select positions while respecting sector diversification constraints."""
        selected: list[dict] = []
        sector_counts: dict[str, int] = {}
        sectors_represented: set[str] = set()

        for candidate in ranked:
            if len(selected) >= self.config.max_positions:
                break

            sector = candidate.get("sector", "Unknown")
            scenario = candidate.get("scenario", {})
            metrics = scenario.get("metrics", {})

            # Must have positive expected return
            if metrics.get("expected_return_12m", 0) <= 0:
                continue

            # Track sector count (will enforce weight limits in weighting step)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            sectors_represented.add(sector)

            selected.append({
                "ticker": candidate["ticker"],
                "company": candidate.get("company", ""),
                "sector": sector,
                "expected_return": metrics.get("expected_return_12m", 0),
                "risk_reward": metrics.get("risk_reward", 0),
                "conviction": candidate.get("adversarial", {}).get("conviction_score", 0),
                "volatility": candidate.get("hist_volatility", 0.25),
                "beta": candidate.get("beta", 1.0),
                "composite_score": candidate.get("composite_score", 0),
                "_opt_score": candidate.get("_opt_score", 0),
            })

        # Ensure minimum sector diversity
        if len(sectors_represented) < self.config.min_sectors:
            logger.warning(
                "Only %d sectors represented (min: %d). Portfolio may be under-diversified.",
                len(sectors_represented),
                self.config.min_sectors,
            )

        return selected

    def _calculate_weights(self, positions: list[dict]) -> list[dict]:
        """Calculate position weights using conviction and vol-adjusted sizing."""
        if not positions:
            return positions

        # Raw weight proportional to optimization score, inversely proportional to volatility
        raw_weights = []
        for p in positions:
            vol = max(p.get("volatility", 0.25), 0.10)  # Floor at 10%
            conviction_weight = max(p.get("_opt_score", 1), 0.1)
            raw = conviction_weight / vol
            raw_weights.append(raw)

        total_raw = sum(raw_weights)
        if total_raw <= 0:
            # Equal weight fallback
            equal = 1.0 / len(positions)
            for p in positions:
                p["weight"] = equal
            return positions

        # Normalize to sum to 1.0
        for i, p in enumerate(positions):
            p["weight"] = raw_weights[i] / total_raw

        # Enforce min/max position constraints
        positions = self._enforce_position_limits(positions)

        # Enforce sector limits
        positions = self._enforce_sector_limits(positions)

        return positions

    def _enforce_position_limits(self, positions: list[dict]) -> list[dict]:
        """Clamp weights to min/max position sizes and renormalize."""
        min_w = self.config.min_position_pct / 100
        max_w = self.config.max_position_pct / 100

        for _ in range(10):  # Iterative clamping
            clamped = False
            for p in positions:
                if p["weight"] < min_w:
                    p["weight"] = min_w
                    clamped = True
                elif p["weight"] > max_w:
                    p["weight"] = max_w
                    clamped = True
            if not clamped:
                break

            # Renormalize
            total = sum(p["weight"] for p in positions)
            if total > 0:
                for p in positions:
                    p["weight"] = p["weight"] / total

        return positions

    def _enforce_sector_limits(self, positions: list[dict]) -> list[dict]:
        """Ensure no sector exceeds max allocation."""
        max_sector = self.config.max_sector_pct / 100

        for _ in range(5):  # Iterative adjustment
            sector_weights: dict[str, float] = {}
            for p in positions:
                sector = p["sector"]
                sector_weights[sector] = sector_weights.get(sector, 0) + p["weight"]

            over_limit = {s: w for s, w in sector_weights.items() if w > max_sector}
            if not over_limit:
                break

            # Scale down positions in over-limit sectors
            for sector, total_weight in over_limit.items():
                scale_factor = max_sector / total_weight
                for p in positions:
                    if p["sector"] == sector:
                        p["weight"] *= scale_factor

            # Redistribute freed weight to other positions
            current_total = sum(p["weight"] for p in positions)
            if current_total > 0 and current_total < 1.0:
                deficit = 1.0 - current_total
                non_capped = [
                    p for p in positions
                    if sector_weights.get(p["sector"], 0) <= max_sector
                ]
                if non_capped:
                    bonus = deficit / len(non_capped)
                    for p in non_capped:
                        p["weight"] += bonus

        # Final normalization
        total = sum(p["weight"] for p in positions)
        if total > 0:
            for p in positions:
                p["weight"] = p["weight"] / total

        return positions

    def _calculate_portfolio_metrics(self, positions: list[dict]) -> dict:
        """Calculate aggregate portfolio statistics."""
        if not positions:
            return {
                "expected_return": 0,
                "max_drawdown": 0,
                "sharpe": 0,
                "beta": 0,
                "num_sectors": 0,
            }

        # Weighted expected return
        exp_return = sum(p["weight"] * p.get("expected_return", 0) for p in positions)

        # Weighted beta
        weighted_beta = sum(p["weight"] * p.get("beta", 1.0) for p in positions)

        # Weighted volatility for Sharpe estimate
        weighted_vol = sum(p["weight"] * p.get("volatility", 0.25) for p in positions)

        # Sharpe estimate (using risk-free rate of ~5%)
        risk_free = 0.05
        sharpe = (exp_return - risk_free) / weighted_vol if weighted_vol > 0 else 0

        # Estimated max drawdown (heuristic: ~2x weighted vol)
        max_drawdown = -weighted_vol * 2

        # Sector count
        sectors = set(p["sector"] for p in positions)

        return {
            "expected_return": exp_return,
            "max_drawdown": max_drawdown,
            "sharpe": round(sharpe, 2),
            "beta": round(weighted_beta, 2),
            "num_sectors": len(sectors),
            "weighted_volatility": weighted_vol,
        }
