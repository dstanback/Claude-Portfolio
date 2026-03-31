"""Quantitative scoring functions for stock evaluation."""

from __future__ import annotations

import numpy as np


def score_financial_quality(data: dict) -> float:
    """Score financial quality on 0-100 scale.

    Components: revenue growth, margins, FCF, ROE, debt levels.
    """
    scores = []

    # Revenue growth (0-20)
    rev_growth = data.get("revenue_growth_yoy", 0)
    scores.append(min(20, max(0, rev_growth * 50 + 10)))

    # Gross margin (0-15)
    gross = data.get("gross_margin", 0)
    scores.append(min(15, max(0, gross * 20)))

    # Operating margin (0-15)
    op_margin = data.get("operating_margin", 0)
    scores.append(min(15, max(0, op_margin * 30 + 5)))

    # Profit margin (0-15)
    net_margin = data.get("profit_margin", 0)
    scores.append(min(15, max(0, net_margin * 30 + 5)))

    # ROE (0-15)
    roe = data.get("roe", 0)
    scores.append(min(15, max(0, roe * 30 + 5)))

    # Debt-to-equity penalty (0-20, lower is better)
    dte = data.get("debt_to_equity", 0)
    if dte <= 0:
        scores.append(15)  # No debt or negative (unusual)
    elif dte <= 50:
        scores.append(20)
    elif dte <= 100:
        scores.append(15)
    elif dte <= 200:
        scores.append(10)
    else:
        scores.append(max(0, 20 - dte / 50))

    return min(100, sum(scores))


def score_valuation(data: dict) -> float:
    """Score valuation attractiveness on 0-100 scale.

    Lower valuations score higher. Uses sector-relative heuristics.
    """
    scores = []

    # Forward P/E (0-30): lower is better
    fpe = data.get("pe_forward", 0)
    if fpe <= 0:
        scores.append(5)  # Negative earnings
    elif fpe <= 12:
        scores.append(30)
    elif fpe <= 18:
        scores.append(25)
    elif fpe <= 25:
        scores.append(18)
    elif fpe <= 35:
        scores.append(12)
    elif fpe <= 50:
        scores.append(6)
    else:
        scores.append(2)

    # P/S ratio (0-25): lower is better
    ps = data.get("ps_ratio", 0)
    if ps <= 0:
        scores.append(5)
    elif ps <= 2:
        scores.append(25)
    elif ps <= 5:
        scores.append(18)
    elif ps <= 10:
        scores.append(12)
    elif ps <= 20:
        scores.append(6)
    else:
        scores.append(2)

    # EV/EBITDA (0-25): lower is better
    ev = data.get("ev_ebitda", 0)
    if ev <= 0:
        scores.append(5)
    elif ev <= 8:
        scores.append(25)
    elif ev <= 12:
        scores.append(20)
    elif ev <= 18:
        scores.append(14)
    elif ev <= 25:
        scores.append(8)
    else:
        scores.append(3)

    # PEG ratio (0-20): closer to 1 is ideal
    peg = data.get("peg_ratio", 0)
    if peg <= 0:
        scores.append(5)
    elif peg <= 0.8:
        scores.append(20)
    elif peg <= 1.2:
        scores.append(18)
    elif peg <= 2.0:
        scores.append(12)
    elif peg <= 3.0:
        scores.append(6)
    else:
        scores.append(2)

    return min(100, sum(scores))


def score_momentum(data: dict) -> float:
    """Score momentum and technicals on 0-100 scale.

    Components: multi-period returns, RSI, MA positioning.
    """
    scores = []

    # Multi-period returns (0-50)
    for key, max_pts in [("perf_1m", 8), ("perf_3m", 12), ("perf_6m", 15), ("perf_12m", 15)]:
        ret = data.get(key, 0)
        if ret > 0.30:
            scores.append(max_pts)
        elif ret > 0.15:
            scores.append(max_pts * 0.85)
        elif ret > 0.05:
            scores.append(max_pts * 0.65)
        elif ret > 0:
            scores.append(max_pts * 0.50)
        elif ret > -0.10:
            scores.append(max_pts * 0.30)
        else:
            scores.append(max_pts * 0.10)

    # RSI (0-20): 40-70 is ideal, extremes penalized
    rsi = data.get("rsi_14", 50)
    if 45 <= rsi <= 65:
        scores.append(20)
    elif 35 <= rsi <= 75:
        scores.append(14)
    elif 25 <= rsi <= 80:
        scores.append(8)
    else:
        scores.append(3)

    # Moving average positioning (0-30)
    above_50 = data.get("above_sma50", 0.5)
    above_200 = data.get("above_sma200", 0.5)
    scores.append(above_50 * 15 + above_200 * 15)

    return min(100, sum(scores))


def score_sentiment(data: dict) -> float:
    """Score analyst sentiment on 0-100 scale.

    Components: consensus rating, analyst count, price target upside.
    """
    scores = []

    # Analyst rating (0-40): 1=strong buy, 5=strong sell
    rating = data.get("analyst_rating", 3.0)
    if rating <= 1.5:
        scores.append(40)
    elif rating <= 2.0:
        scores.append(35)
    elif rating <= 2.5:
        scores.append(28)
    elif rating <= 3.0:
        scores.append(18)
    elif rating <= 3.5:
        scores.append(10)
    else:
        scores.append(3)

    # Analyst coverage depth (0-20)
    count = data.get("analyst_count", 0)
    if count >= 20:
        scores.append(20)
    elif count >= 10:
        scores.append(15)
    elif count >= 5:
        scores.append(10)
    else:
        scores.append(5)

    # Price target upside (0-40)
    current = data.get("current_price", 0)
    target = data.get("target_mean", 0)
    if current > 0 and target > 0:
        upside = (target - current) / current
        if upside > 0.30:
            scores.append(40)
        elif upside > 0.20:
            scores.append(33)
        elif upside > 0.10:
            scores.append(25)
        elif upside > 0.05:
            scores.append(18)
        elif upside > 0:
            scores.append(10)
        else:
            scores.append(3)
    else:
        scores.append(10)

    return min(100, sum(scores))


def score_catalyst(data: dict) -> float:
    """Score near-term catalyst potential on 0-100 scale.

    Heuristic based on earnings growth, momentum acceleration, and volatility.
    """
    scores = []

    # Earnings growth momentum (0-35)
    eg = data.get("earnings_growth", 0)
    if eg > 0.50:
        scores.append(35)
    elif eg > 0.25:
        scores.append(28)
    elif eg > 0.10:
        scores.append(20)
    elif eg > 0:
        scores.append(12)
    else:
        scores.append(5)

    # Short-term momentum acceleration (0-30): 1W vs 1M
    perf_1w = data.get("perf_1w", 0)
    perf_1m = data.get("perf_1m", 0)
    weekly_annualized = perf_1w * 52
    monthly_annualized = perf_1m * 12
    if weekly_annualized > monthly_annualized and perf_1w > 0:
        scores.append(min(30, 15 + perf_1w * 200))
    elif perf_1w > 0:
        scores.append(15)
    else:
        scores.append(5)

    # Implied vol / beta as catalyst proxy (0-20)
    beta = data.get("beta", 1.0)
    if 1.0 <= beta <= 1.5:
        scores.append(20)
    elif 0.8 <= beta < 1.0:
        scores.append(15)
    elif 1.5 < beta <= 2.0:
        scores.append(15)
    else:
        scores.append(8)

    # Analyst target dispersion as event proxy (0-15)
    target_high = data.get("target_high", 0)
    target_low = data.get("target_low", 0)
    current = data.get("current_price", 1)
    if current > 0 and target_high > 0 and target_low > 0:
        dispersion = (target_high - target_low) / current
        if dispersion > 0.50:
            scores.append(15)
        elif dispersion > 0.30:
            scores.append(12)
        elif dispersion > 0.15:
            scores.append(8)
        else:
            scores.append(5)
    else:
        scores.append(7)

    return min(100, sum(scores))


def composite_score(data: dict, weights: dict[str, float] | None = None) -> float:
    """Calculate weighted composite score (0-100) from all sub-scores."""
    if weights is None:
        weights = {
            "financial_quality": 0.25,
            "valuation": 0.20,
            "momentum": 0.20,
            "sentiment": 0.15,
            "catalyst": 0.20,
        }

    sub_scores = {
        "financial_quality": score_financial_quality(data),
        "valuation": score_valuation(data),
        "momentum": score_momentum(data),
        "sentiment": score_sentiment(data),
        "catalyst": score_catalyst(data),
    }

    total = sum(sub_scores[k] * weights[k] for k in weights)
    return round(total, 1)


def determine_key_signal(data: dict) -> str:
    """Determine the most notable signal for a stock."""
    signals = []

    # Check for strong momentum
    if data.get("perf_3m", 0) > 0.20:
        signals.append(f"Strong 3M momentum (+{data['perf_3m']:.0%})")
    elif data.get("perf_3m", 0) < -0.15:
        signals.append(f"Weak 3M momentum ({data['perf_3m']:.0%})")

    # Check for strong earnings
    eg = data.get("earnings_growth", 0)
    if eg > 0.25:
        signals.append(f"EPS growth +{eg:.0%}")

    # Check for attractive valuation
    fpe = data.get("pe_forward", 0)
    if 0 < fpe < 15:
        signals.append(f"Low fwd P/E ({fpe:.1f}x)")

    # Check for analyst bullishness
    rating = data.get("analyst_rating", 3)
    if rating <= 1.8:
        signals.append("Strong Buy consensus")

    # Price target upside
    current = data.get("current_price", 0)
    target = data.get("target_mean", 0)
    if current > 0 and target > 0:
        upside = (target - current) / current
        if upside > 0.25:
            signals.append(f"Target upside +{upside:.0%}")

    return signals[0] if signals else "Composite strength"
