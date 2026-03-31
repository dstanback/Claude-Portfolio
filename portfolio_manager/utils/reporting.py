"""Report generation and formatting utilities."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from tabulate import tabulate

logger = logging.getLogger(__name__)


def format_screening_report(date: str, candidates: list[dict], total_scored: int) -> str:
    """Format Stage 1 screening results."""
    headers = ["Rank", "Ticker", "Company", "Composite Score", "Sector", "Key Signal"]
    rows = []
    for i, c in enumerate(candidates, 1):
        rows.append([
            i,
            c["ticker"],
            c.get("company", "")[:30],
            f"{c['composite_score']:.1f}",
            c.get("sector", ""),
            c.get("key_signal", ""),
        ])

    table = tabulate(rows, headers=headers, tablefmt="pipe")
    return (
        f"SCREENING RESULTS — {date}\n"
        f"Total stocks scored: {total_scored}\n"
        f"Top {len(candidates)} candidates (ranked):\n\n"
        f"{table}\n"
    )


def format_adversarial_report(ticker: str, company: str, result: dict) -> str:
    """Format Stage 2 adversarial research results."""
    bull = result.get("bull", {})
    bear = result.get("bear", {})

    bull_args = "\n".join(f"  - {a}" for a in bull.get("arguments", [])[:5])
    bear_args = "\n".join(f"  - {a}" for a in bear.get("arguments", [])[:5])

    return (
        f"ADVERSARIAL RESEARCH: {ticker} — {company}\n\n"
        f"BULL CASE SUMMARY ({bull.get('agent_count', 15)} agents):\n{bull_args}\n"
        f"Bull Confidence: {bull.get('confidence', 0)}/100\n\n"
        f"BEAR CASE SUMMARY ({bear.get('agent_count', 15)} agents):\n{bear_args}\n"
        f"Bear Confidence: {bear.get('confidence', 0)}/100\n\n"
        f"ADVERSARIAL CONVICTION SCORE: {result.get('conviction_score', 0):+d}\n"
        f"VERDICT: {result.get('verdict', 'UNKNOWN')}\n"
    )


def format_scenario_report(ticker: str, company: str, current_price: float, model: dict) -> str:
    """Format Stage 3 scenario modeling results."""
    scenarios = model.get("scenarios", {})
    headers = ["Scenario", "Probability", "1M Target", "3M Target", "6M Target", "12M Target"]
    rows = []
    for name in ["bull", "base", "bear"]:
        s = scenarios.get(name, {})
        targets = s.get("targets", {})
        rows.append([
            name.title(),
            f"{s.get('probability', 0):.0%}",
            f"${targets.get('1m', 0):,.2f}",
            f"${targets.get('3m', 0):,.2f}",
            f"${targets.get('6m', 0):,.2f}",
            f"${targets.get('12m', 0):,.2f}",
        ])

    table = tabulate(rows, headers=headers, tablefmt="pipe")
    metrics = model.get("metrics", {})

    return (
        f"SCENARIO MODEL: {ticker} — {company}\n"
        f"Current Price: ${current_price:,.2f}\n\n"
        f"{table}\n\n"
        f"Probability-Weighted Expected Return (12M): "
        f"{metrics.get('expected_return_12m', 0):+.1%}\n"
        f"Expected Downside (12M): {metrics.get('expected_downside_12m', 0):+.1%}\n"
        f"Risk/Reward Ratio: {metrics.get('risk_reward', 0):.2f}x\n"
        f"Sharpe-Adjusted Alpha vs SPY: {metrics.get('alpha_vs_spy', 0):+.1%}\n\n"
        f"Self-Debate Notes: {model.get('self_debate_notes', 'N/A')}\n"
    )


def format_portfolio_report(date: str, portfolio: dict) -> str:
    """Format Stage 4 portfolio construction results."""
    positions = portfolio.get("positions", [])
    total_value = portfolio.get("total_value", 0)

    headers = [
        "#", "Ticker", "Company", "Sector", "Weight",
        "$ Allocation", "Expected Return (12M)", "Risk/Reward",
    ]
    rows = []
    for i, p in enumerate(positions, 1):
        rows.append([
            i,
            p["ticker"],
            p.get("company", "")[:20],
            p.get("sector", ""),
            f"{p['weight']:.1%}",
            f"${p['allocation']:,.2f}",
            f"{p.get('expected_return', 0):+.1%}",
            f"{p.get('risk_reward', 0):.2f}x",
        ])

    table = tabulate(rows, headers=headers, tablefmt="pipe")

    # Sector breakdown
    sector_weights: dict[str, float] = {}
    for p in positions:
        sector = p.get("sector", "Unknown")
        sector_weights[sector] = sector_weights.get(sector, 0) + p["weight"]
    sector_lines = "\n".join(
        f"  - {s}: {w:.1%}" for s, w in sorted(sector_weights.items(), key=lambda x: -x[1])
    )

    metrics = portfolio.get("metrics", {})

    return (
        f"PORTFOLIO CONSTRUCTION — {date}\n"
        f"Total Portfolio Value: ${total_value:,.2f}\n\n"
        f"{table}\n\n"
        f"SECTOR BREAKDOWN:\n{sector_lines}\n\n"
        f"PORTFOLIO METRICS:\n"
        f"  - Expected Return (12M): {metrics.get('expected_return', 0):+.1%}\n"
        f"  - Expected Max Drawdown: {metrics.get('max_drawdown', 0):+.1%}\n"
        f"  - Sharpe Estimate: {metrics.get('sharpe', 0):.2f}\n"
        f"  - Beta to SPY: {metrics.get('beta', 0):.2f}\n"
        f"  - Number of Sectors: {metrics.get('num_sectors', 0)}\n"
    )


def format_trade_recommendation(trade: dict) -> str:
    """Format Stage 5 trade recommendation."""
    return (
        f"TRADE RECOMMENDATION — {trade.get('date', 'N/A')}\n\n"
        f"ACTION: {trade.get('action', 'N/A')}\n"
        f"SELL: {trade.get('sell_ticker', 'N/A')} — {trade.get('sell_reason', '')}\n"
        f"BUY: {trade.get('buy_ticker', 'N/A')} — {trade.get('buy_reason', '')}\n"
        f"Impact on Portfolio Expected Return: {trade.get('return_impact', 0):+.1%}\n"
        f"Impact on Sector Balance: {trade.get('sector_impact', '')}\n\n"
        f"FULL RATIONALE:\n{trade.get('rationale', 'N/A')}\n"
    )


def format_weekly_report(portfolio_state: dict) -> str:
    """Format the weekly portfolio status report."""
    holdings = portfolio_state.get("holdings", [])
    perf = portfolio_state.get("performance", {})
    watchlist = portfolio_state.get("watchlist", [])

    headers = ["Ticker", "Shares", "Avg Cost", "Current", "P&L $", "P&L %"]
    rows = []
    for h in holdings:
        rows.append([
            h["ticker"],
            h.get("shares", 0),
            f"${h.get('avg_cost', 0):,.2f}",
            f"${h.get('current_price', 0):,.2f}",
            f"${h.get('pnl_dollar', 0):+,.2f}",
            f"{h.get('pnl_pct', 0):+.1%}",
        ])
    holdings_table = tabulate(rows, headers=headers, tablefmt="pipe")

    watch_rows = [[w["ticker"], f"{w.get('score', 0):.1f}", w.get("signal", "")]
                  for w in watchlist[:5]]
    watch_table = tabulate(watch_rows, headers=["Ticker", "Score", "Signal"], tablefmt="pipe")

    return (
        f"WEEKLY PORTFOLIO STATUS — {portfolio_state.get('date', 'N/A')}\n\n"
        f"HOLDINGS:\n{holdings_table}\n\n"
        f"PERFORMANCE vs SPY:\n"
        f"  - 1W: Portfolio {perf.get('1w', 0):+.1%} vs SPY {perf.get('spy_1w', 0):+.1%}\n"
        f"  - MTD: Portfolio {perf.get('mtd', 0):+.1%} vs SPY {perf.get('spy_mtd', 0):+.1%}\n"
        f"  - YTD: Portfolio {perf.get('ytd', 0):+.1%} vs SPY {perf.get('spy_ytd', 0):+.1%}\n\n"
        f"TOP 5 WATCHLIST:\n{watch_table}\n"
    )


def save_report(content: str, stage: str, output_dir: Path | None = None) -> Path:
    """Save a report to disk with timestamp."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent.parent / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{stage}_{timestamp}.txt"
    filepath = output_dir / filename
    filepath.write_text(content, encoding="utf-8")
    logger.info("Report saved: %s", filepath)
    return filepath


def save_json(data: Any, name: str, output_dir: Path | None = None) -> Path:
    """Save structured data as JSON."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.json"
    filepath = output_dir / filename
    filepath.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info("Data saved: %s", filepath)
    return filepath
