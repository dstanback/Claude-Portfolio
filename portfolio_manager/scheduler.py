"""Scheduled execution of the portfolio pipeline.

Runs the pipeline on a daily schedule, executing the full pipeline on the
first run and daily rebalance checks thereafter.
"""

from __future__ import annotations

import logging
import time

import schedule
from dotenv import load_dotenv

from config.settings import PipelineConfig
from portfolio_manager.pipeline import PortfolioPipeline

logger = logging.getLogger(__name__)


def create_scheduler(config: PipelineConfig | None = None) -> None:
    """Set up and run the scheduled portfolio pipeline.

    Schedule:
        - Daily at configured time: Run screening + rebalance check
        - Weekly on Friday: Generate full portfolio report
        - First run: Execute full pipeline
    """
    load_dotenv()
    config = config or PipelineConfig()
    pipeline = PortfolioPipeline(config=config)

    rebalance_time = (
        f"{config.rebalance.rebalance_hour:02d}:{config.rebalance.rebalance_minute:02d}"
    )

    def daily_job() -> None:
        """Daily rebalance check."""
        try:
            if pipeline.portfolio is None:
                logger.info("No portfolio exists. Running full pipeline...")
                pipeline.run_full_pipeline()
            else:
                logger.info("Running daily rebalance check...")
                trades = pipeline.run_daily_rebalance()
                if trades:
                    logger.info("%d trade recommendations generated", len(trades))
                else:
                    logger.info("No trades recommended")
        except Exception:
            logger.exception("Daily job failed")

    def weekly_job() -> None:
        """Weekly portfolio report."""
        try:
            if pipeline.portfolio is not None:
                pipeline.run_weekly_report()
            else:
                logger.info("No portfolio to report on. Skipping weekly report.")
        except Exception:
            logger.exception("Weekly report failed")

    # Schedule jobs
    schedule.every().day.at(rebalance_time).do(daily_job)
    schedule.every().friday.at("18:00").do(weekly_job)

    logger.info("Scheduler initialized. Daily job at %s ET, weekly report Friday 18:00 ET",
                rebalance_time)

    # Run initial full pipeline
    logger.info("Running initial full pipeline...")
    pipeline.run_full_pipeline()

    # Enter scheduling loop
    logger.info("Entering scheduler loop...")
    while True:
        schedule.run_pending()
        time.sleep(60)
