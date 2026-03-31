"""Stage 1: Screening — Narrow Russell 1000 to top 50 candidates."""

from __future__ import annotations

import logging
from datetime import datetime

from config.settings import ScreeningConfig
from portfolio_manager.utils.market_data import (
    fetch_batch_stock_data,
    fetch_russell_1000_tickers,
)
from portfolio_manager.utils.reporting import format_screening_report, save_report, save_json
from portfolio_manager.utils.scoring import composite_score, determine_key_signal

logger = logging.getLogger(__name__)


class ScreeningStage:
    """Stage 1: Quantitative screening of the Russell 1000 universe."""

    def __init__(self, config: ScreeningConfig | None = None):
        self.config = config or ScreeningConfig()

    def run(self) -> list[dict]:
        """Execute the full screening pipeline.

        Returns:
            List of top N candidates sorted by composite score (descending).
        """
        logger.info("Stage 1: Screening — Starting")

        # Step 1: Get universe
        tickers = fetch_russell_1000_tickers()
        logger.info("Universe size: %d tickers", len(tickers))

        # Step 2: Fetch data for all tickers
        logger.info("Fetching market data for %d stocks...", len(tickers))
        stock_data = fetch_batch_stock_data(tickers)
        logger.info("Successfully fetched data for %d stocks", len(stock_data))

        # Step 3: Score each stock
        scored = []
        for data in stock_data:
            score = composite_score(data, self.config.weights)
            data["composite_score"] = score
            data["key_signal"] = determine_key_signal(data)
            scored.append(data)

        # Step 4: Rank and select top N
        scored.sort(key=lambda x: x["composite_score"], reverse=True)
        candidates = scored[: self.config.top_n]

        # Step 5: Generate report
        date_str = datetime.now().strftime("%Y-%m-%d")
        report = format_screening_report(date_str, candidates, len(stock_data))
        logger.info("\n%s", report)
        save_report(report, "stage1_screening")
        save_json(candidates, "screening_candidates")

        logger.info(
            "Stage 1 complete: %d candidates selected from %d scored stocks",
            len(candidates),
            len(stock_data),
        )
        return candidates
