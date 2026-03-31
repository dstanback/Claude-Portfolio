"""Stage 2: Adversarial Research — Bull/Bear debate for each candidate."""

from __future__ import annotations

import logging
from datetime import datetime

from config.settings import AdversarialConfig
from portfolio_manager.utils.claude_client import run_bull_bear_analysis
from portfolio_manager.utils.reporting import format_adversarial_report, save_report, save_json

logger = logging.getLogger(__name__)


class AdversarialResearchStage:
    """Stage 2: Structured bull/bear adversarial analysis using Claude agents."""

    def __init__(self, config: AdversarialConfig | None = None, model: str = "claude-opus-4-6"):
        self.config = config or AdversarialConfig()
        self.model = model

    def run(self, candidates: list[dict]) -> list[dict]:
        """Run adversarial research on all candidates.

        Args:
            candidates: List of stock data dicts from Stage 1.

        Returns:
            List of candidates that pass the conviction threshold, enriched
            with adversarial research results.
        """
        logger.info("Stage 2: Adversarial Research — Analyzing %d candidates", len(candidates))
        results = []

        for i, stock in enumerate(candidates, 1):
            ticker = stock["ticker"]
            company = stock.get("company", ticker)
            logger.info(
                "[%d/%d] Adversarial analysis: %s (%s)",
                i, len(candidates), ticker, company,
            )

            result = self._analyze_stock(ticker, company, stock)
            stock["adversarial"] = result

            # Log the report
            report = format_adversarial_report(ticker, company, result)
            logger.info("\n%s", report)

            if result["verdict"] == "ADVANCE":
                results.append(stock)
                logger.info("%s ADVANCED (conviction: %+d)", ticker, result["conviction_score"])
            else:
                logger.info("%s ELIMINATED (conviction: %+d)", ticker, result["conviction_score"])

        # Save results
        save_report(
            self._build_summary(results, len(candidates)),
            "stage2_adversarial",
        )
        save_json(
            [{"ticker": r["ticker"], **r["adversarial"]} for r in results],
            "adversarial_results",
        )

        logger.info(
            "Stage 2 complete: %d/%d candidates advanced",
            len(results), len(candidates),
        )
        return results

    def _analyze_stock(self, ticker: str, company: str, stock_data: dict) -> dict:
        """Run bull and bear analysis for a single stock."""
        # Prepare a clean subset of data for the API call
        analysis_data = {
            k: stock_data[k]
            for k in [
                "ticker", "company", "sector", "current_price", "market_cap",
                "revenue_growth_yoy", "earnings_growth", "gross_margin",
                "operating_margin", "profit_margin", "pe_forward", "ps_ratio",
                "ev_ebitda", "perf_1m", "perf_3m", "perf_6m", "rsi_14",
                "analyst_rating", "target_mean", "composite_score",
            ]
            if k in stock_data
        }

        # Run bull and bear analyses
        bull_result = run_bull_bear_analysis(
            ticker, company, analysis_data, "bull",
            agent_count=self.config.bull_agents,
            model=self.model,
        )
        bear_result = run_bull_bear_analysis(
            ticker, company, analysis_data, "bear",
            agent_count=self.config.bear_agents,
            model=self.model,
        )

        # Calculate conviction score: bull confidence minus bear confidence,
        # normalized to -100 to +100
        conviction = bull_result.confidence - bear_result.confidence
        verdict = "ADVANCE" if conviction >= self.config.min_conviction_score else "ELIMINATE"

        return {
            "bull": {
                "arguments": bull_result.arguments,
                "confidence": bull_result.confidence,
                "sources": bull_result.sources,
                "agent_count": self.config.bull_agents,
            },
            "bear": {
                "arguments": bear_result.arguments,
                "confidence": bear_result.confidence,
                "sources": bear_result.sources,
                "agent_count": self.config.bear_agents,
            },
            "conviction_score": conviction,
            "verdict": verdict,
        }

    def _build_summary(self, advanced: list[dict], total: int) -> str:
        """Build a summary report of all adversarial results."""
        lines = [
            f"ADVERSARIAL RESEARCH SUMMARY — {datetime.now().strftime('%Y-%m-%d')}",
            f"Candidates analyzed: {total}",
            f"Candidates advanced: {len(advanced)}",
            f"Minimum conviction threshold: +{self.config.min_conviction_score}",
            "",
            "Advanced candidates:",
        ]
        for stock in advanced:
            adv = stock["adversarial"]
            lines.append(
                f"  {stock['ticker']:6s} | Conviction: {adv['conviction_score']:+4d} | "
                f"Bull: {adv['bull']['confidence']}/100 | Bear: {adv['bear']['confidence']}/100"
            )
        return "\n".join(lines)
