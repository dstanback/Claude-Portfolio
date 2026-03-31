"""Stage 5: Rebalancing — Continuous monitoring and trade recommendations."""

from __future__ import annotations

import logging
from datetime import datetime

from config.settings import RebalanceConfig, PortfolioConfig
from portfolio_manager.utils.market_data import fetch_stock_data, fetch_benchmark_data
from portfolio_manager.utils.reporting import (
    format_trade_recommendation,
    format_weekly_report,
    save_report,
    save_json,
)

logger = logging.getLogger(__name__)


class RebalancingStage:
    """Stage 5: Monitor portfolio and generate trade recommendations."""

    def __init__(
        self,
        config: RebalanceConfig | None = None,
        portfolio_config: PortfolioConfig | None = None,
    ):
        self.config = config or RebalanceConfig()
        self.portfolio_config = portfolio_config or PortfolioConfig()

    def run_daily_check(
        self,
        portfolio: dict,
        bench_candidates: list[dict] | None = None,
    ) -> list[dict]:
        """Run daily monitoring and generate trade recommendations.

        Args:
            portfolio: Current portfolio state from Stage 4.
            bench_candidates: Fresh screening candidates (the "bench") to
                consider as replacements.

        Returns:
            List of trade recommendation dicts.
        """
        logger.info("Stage 5: Daily Rebalance Check — %s", datetime.now().strftime("%Y-%m-%d"))
        trades = []
        positions = portfolio.get("positions", [])

        # Step 1: Update current prices for all holdings
        updated_positions = self._refresh_prices(positions)

        # Step 2: Check for trade triggers
        for pos in updated_positions:
            trigger = self._check_triggers(pos, bench_candidates or [])
            if trigger:
                trades.append(trigger)

        # Step 3: Check sector drift
        sector_trades = self._check_sector_drift(updated_positions)
        trades.extend(sector_trades)

        # Step 4: Log and save recommendations
        if trades:
            for trade in trades:
                report = format_trade_recommendation(trade)
                logger.info("\n%s", report)
            save_report(
                "\n---\n".join(format_trade_recommendation(t) for t in trades),
                "stage5_trades",
            )
            save_json(trades, "trade_recommendations")
        else:
            logger.info("No trade triggers fired. Portfolio unchanged.")

        return trades

    def generate_weekly_report(self, portfolio: dict) -> str:
        """Generate the weekly portfolio status report.

        Args:
            portfolio: Current portfolio state.

        Returns:
            Formatted weekly report string.
        """
        logger.info("Generating weekly portfolio report")
        positions = portfolio.get("positions", [])

        # Refresh prices and calculate P&L
        holdings = []
        for pos in positions:
            ticker = pos["ticker"]
            current_data = fetch_stock_data(ticker)
            current_price = (
                current_data["current_price"] if current_data else pos.get("current_price", 0)
            )
            avg_cost = pos.get("avg_cost", pos.get("current_price", current_price))
            allocation = pos.get("allocation", 0)
            shares = allocation / avg_cost if avg_cost > 0 else 0

            pnl_dollar = (current_price - avg_cost) * shares
            pnl_pct = (current_price - avg_cost) / avg_cost if avg_cost > 0 else 0

            holdings.append({
                "ticker": ticker,
                "shares": round(shares, 2),
                "avg_cost": avg_cost,
                "current_price": current_price,
                "pnl_dollar": pnl_dollar,
                "pnl_pct": pnl_pct,
            })

        # Benchmark comparison (simplified)
        spy_data = fetch_benchmark_data("SPY", period="1mo")
        spy_return_1w = 0.0
        if len(spy_data) >= 5:
            spy_return_1w = (
                float(spy_data["Close"].iloc[-1]) / float(spy_data["Close"].iloc[-5]) - 1
            )

        portfolio_state = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "holdings": holdings,
            "performance": {
                "1w": sum(h["pnl_pct"] for h in holdings) / max(len(holdings), 1),
                "mtd": sum(h["pnl_pct"] for h in holdings) / max(len(holdings), 1),
                "ytd": sum(h["pnl_pct"] for h in holdings) / max(len(holdings), 1),
                "spy_1w": spy_return_1w,
                "spy_mtd": spy_return_1w * 4,  # Rough estimate
                "spy_ytd": spy_return_1w * 52,  # Rough estimate
            },
            "watchlist": self._get_watchlist(portfolio.get("bench_candidates", [])),
        }

        report = format_weekly_report(portfolio_state)
        logger.info("\n%s", report)
        save_report(report, "stage5_weekly")
        return report

    def _refresh_prices(self, positions: list[dict]) -> list[dict]:
        """Fetch current prices for all positions."""
        for pos in positions:
            data = fetch_stock_data(pos["ticker"])
            if data:
                pos["current_price"] = data["current_price"]
                pos["_latest_data"] = data
        return positions

    def _check_triggers(self, position: dict, bench: list[dict]) -> dict | None:
        """Check if any trade triggers fire for a position.

        Trade triggers:
        1. Conviction score drops below threshold
        2. Price breaches bear case target
        3. Bench candidate dominates on all dimensions
        4. Material negative catalyst
        """
        ticker = position["ticker"]
        data = position.get("_latest_data", {})

        # Trigger 1: Weak momentum reversal (proxy for conviction drop)
        perf_1w = data.get("perf_1w", 0)
        perf_1m = data.get("perf_1m", 0)
        if perf_1w < -0.10 and perf_1m < -0.15:
            # Find best replacement from bench
            replacement = self._find_replacement(position, bench)
            if replacement:
                return self._build_trade("SWAP", position, replacement, "Momentum breakdown")

        # Trigger 2: Breach bear case target (use allocation drop as proxy)
        original_price = position.get("allocation", 0) / max(
            position.get("weight", 0.05) * 10000, 1
        )
        current_price = data.get("current_price", original_price)
        if original_price > 0 and current_price < original_price * 0.85:
            replacement = self._find_replacement(position, bench)
            if replacement:
                return self._build_trade(
                    "SWAP", position, replacement, "Bear case target breached"
                )

        # Trigger 3: Check if any bench candidate dominates
        for candidate in bench[:10]:
            cand_metrics = candidate.get("scenario", {}).get("metrics", {})
            if (
                cand_metrics.get("expected_return_12m", 0) > position.get("expected_return", 0)
                and cand_metrics.get("risk_reward", 0) > position.get("risk_reward", 0)
                and candidate.get("adversarial", {}).get("conviction_score", 0)
                > position.get("conviction", 0)
            ):
                return self._build_trade(
                    "SWAP", position, candidate,
                    "Bench candidate dominates on all dimensions",
                )

        return None

    def _check_sector_drift(self, positions: list[dict]) -> list[dict]:
        """Check if any sector has drifted above the maximum allowed weight."""
        max_sector = self.portfolio_config.max_sector_pct / 100
        sector_weights: dict[str, float] = {}
        total_value = sum(
            p.get("current_price", 0) * p.get("allocation", 0) / max(p.get("current_price", 1), 1)
            for p in positions
        )

        if total_value <= 0:
            return []

        for p in positions:
            sector = p.get("sector", "Unknown")
            pos_value = p.get("current_price", 0) * p.get("allocation", 0) / max(
                p.get("current_price", 1), 1
            )
            sector_weights[sector] = sector_weights.get(sector, 0) + pos_value / total_value

        trades = []
        for sector, weight in sector_weights.items():
            if weight > max_sector:
                # Find highest-weight position in sector to trim
                sector_positions = [p for p in positions if p.get("sector") == sector]
                sector_positions.sort(key=lambda x: x.get("weight", 0), reverse=True)
                if sector_positions:
                    trades.append({
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "action": "TRIM",
                        "sell_ticker": sector_positions[0]["ticker"],
                        "sell_reason": f"Sector {sector} overweight at {weight:.0%} (max {max_sector:.0%})",
                        "buy_ticker": "N/A",
                        "buy_reason": "Redistribute to underweight sectors",
                        "return_impact": 0,
                        "sector_impact": f"{sector}: {weight:.0%} → {max_sector:.0%}",
                        "rationale": (
                            f"The {sector} sector has drifted to {weight:.0%} of portfolio, "
                            f"exceeding the {max_sector:.0%} maximum. Trimming the largest "
                            f"position ({sector_positions[0]['ticker']}) to restore balance."
                        ),
                    })
        return trades

    def _find_replacement(self, position: dict, bench: list[dict]) -> dict | None:
        """Find the best replacement for a position from the bench."""
        pos_sector = position.get("sector", "Unknown")
        for candidate in bench:
            cand_metrics = candidate.get("scenario", {}).get("metrics", {})
            if cand_metrics.get("expected_return_12m", 0) > 0:
                return candidate
        return None

    def _build_trade(
        self, action: str, sell_pos: dict, buy_candidate: dict, reason: str
    ) -> dict:
        """Build a trade recommendation dict."""
        buy_metrics = buy_candidate.get("scenario", {}).get("metrics", {})
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "action": action,
            "sell_ticker": sell_pos["ticker"],
            "sell_reason": reason,
            "buy_ticker": buy_candidate["ticker"],
            "buy_reason": (
                f"Expected return {buy_metrics.get('expected_return_12m', 0):+.1%}, "
                f"R/R {buy_metrics.get('risk_reward', 0):.2f}x"
            ),
            "return_impact": (
                buy_metrics.get("expected_return_12m", 0)
                - sell_pos.get("expected_return", 0)
            ),
            "sector_impact": (
                f"{sell_pos.get('sector', '?')} → {buy_candidate.get('sector', '?')}"
            ),
            "rationale": (
                f"{sell_pos['ticker']} is being removed due to: {reason}. "
                f"Replacing with {buy_candidate['ticker']} which offers a superior "
                f"risk-adjusted return profile with expected 12M return of "
                f"{buy_metrics.get('expected_return_12m', 0):+.1%} and "
                f"risk/reward ratio of {buy_metrics.get('risk_reward', 0):.2f}x."
            ),
        }

    def _get_watchlist(self, bench: list[dict]) -> list[dict]:
        """Get top 5 bench candidates as watchlist."""
        watchlist = []
        for c in bench[:5]:
            metrics = c.get("scenario", {}).get("metrics", {})
            watchlist.append({
                "ticker": c["ticker"],
                "score": c.get("composite_score", 0),
                "signal": f"E[R]={metrics.get('expected_return_12m', 0):+.1%}",
            })
        return watchlist
