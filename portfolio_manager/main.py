"""CLI entry point for the portfolio management system."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PipelineConfig, LOGS_DIR
from portfolio_manager.pipeline import PortfolioPipeline


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the pipeline."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOGS_DIR / "pipeline.log", mode="a"),
        ],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Claude-powered autonomous portfolio management system",
    )
    parser.add_argument(
        "command",
        choices=["run", "screen", "rebalance", "report"],
        help=(
            "run: Execute full 5-stage pipeline. "
            "screen: Run Stage 1 screening only. "
            "rebalance: Run daily rebalance check. "
            "report: Generate weekly report."
        ),
    )
    parser.add_argument(
        "--portfolio-value",
        type=float,
        default=100_000.0,
        help="Total portfolio value in USD (default: 100000)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-opus-4-6",
        help="Claude model to use for analysis (default: claude-opus-4-6)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of candidates to select in screening (default: 50)",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=15,
        help="Maximum portfolio positions (default: 15)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose/debug logging",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    load_dotenv()
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("Portfolio Manager starting — command: %s", args.command)

    # Build config from CLI args
    from config.settings import ScreeningConfig, PortfolioConfig

    config = PipelineConfig(
        screening=ScreeningConfig(top_n=args.top_n),
        portfolio=PortfolioConfig(max_positions=args.max_positions),
        portfolio_value=args.portfolio_value,
        model=args.model,
    )

    pipeline = PortfolioPipeline(config=config)

    if args.command == "run":
        portfolio = pipeline.run_full_pipeline()
        positions = portfolio.get("positions", [])
        logger.info(
            "Portfolio built with %d positions, expected return: %+.1f%%",
            len(positions),
            portfolio.get("metrics", {}).get("expected_return", 0) * 100,
        )

    elif args.command == "screen":
        candidates = pipeline.run_screening_only()
        logger.info("Screening complete: %d candidates identified", len(candidates))

    elif args.command == "rebalance":
        # Would need to load existing portfolio from disk in production
        logger.info("Running daily rebalance check...")
        logger.info("Note: Run 'run' first to create a portfolio, then use 'rebalance'")

    elif args.command == "report":
        logger.info("Generating weekly report...")
        logger.info("Note: Run 'run' first to create a portfolio, then use 'report'")


if __name__ == "__main__":
    main()
