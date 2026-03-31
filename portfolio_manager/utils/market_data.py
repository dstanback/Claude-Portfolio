"""Market data fetching utilities using yfinance."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Russell 1000 top constituents (representative subset for screening).
# In production, this would be fetched from an index provider API.
RUSSELL_1000_URL = (
    "https://en.wikipedia.org/wiki/Russell_1000_Index"
)

SECTOR_MAP = {
    "Technology": "Technology",
    "Communication Services": "Communication Services",
    "Consumer Cyclical": "Consumer Cyclical",
    "Consumer Defensive": "Consumer Defensive",
    "Energy": "Energy",
    "Financial Services": "Financials",
    "Financials": "Financials",
    "Healthcare": "Healthcare",
    "Industrials": "Industrials",
    "Basic Materials": "Materials",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
}


def normalize_sector(raw_sector: str | None) -> str:
    """Normalize sector names to a consistent taxonomy."""
    if not raw_sector:
        return "Unknown"
    return SECTOR_MAP.get(raw_sector, raw_sector)


def fetch_russell_1000_tickers() -> list[str]:
    """Fetch Russell 1000 constituent tickers.

    Falls back to a curated list of ~200 large-cap tickers that represent
    the most liquid Russell 1000 constituents across all sectors.
    """
    # Curated representative subset across all 11 GICS sectors
    tickers = [
        # Technology
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE", "ACN", "CSCO",
        "INTC", "IBM", "INTU", "NOW", "QCOM", "TXN", "AMAT", "MU", "LRCX", "ADI",
        "KLAC", "SNPS", "CDNS", "MRVL", "FTNT", "PANW", "CRWD", "ZS", "DDOG", "SNOW",
        # Communication Services
        "META", "GOOGL", "GOOG", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "EA",
        "TTWO", "MTCH", "RBLX", "PINS", "SNAP",
        # Consumer Cyclical
        "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "ABNB",
        "MAR", "GM", "F", "ROST", "DHI", "LEN", "ORLY", "AZO", "CMG", "YUM",
        # Consumer Defensive
        "WMT", "PG", "COST", "KO", "PEP", "PM", "MO", "MDLZ", "CL", "KMB",
        "GIS", "HSY", "SJM", "K", "STZ",
        # Energy
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "PXD", "OXY",
        "HAL", "DVN", "FANG", "HES", "BKR",
        # Financials
        "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "BLK",
        "AXP", "C", "SCHW", "CB", "MMC", "ICE", "CME", "AON", "PGR", "TRV",
        # Healthcare
        "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY",
        "AMGN", "GILD", "ISRG", "MDT", "SYK", "BSX", "VRTX", "REGN", "ZTS", "EW",
        # Industrials
        "CAT", "GE", "RTX", "HON", "UNP", "UPS", "BA", "DE", "LMT", "MMM",
        "GD", "NOC", "WM", "RSG", "EMR", "ITW", "ETN", "PH", "ROK", "CARR",
        # Materials
        "LIN", "APD", "ECL", "SHW", "FCX", "NEM", "NUE", "DOW", "DD", "PPG",
        # Real Estate
        "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "O", "DLR", "WELL", "AVB",
        # Utilities
        "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "WEC",
    ]
    return tickers


def fetch_stock_data(ticker: str) -> dict | None:
    """Fetch comprehensive data for a single stock.

    Returns a dict with financial, valuation, momentum, and sentiment data,
    or None if the fetch fails.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or "symbol" not in info:
            return None

        hist_1y = stock.history(period="1y")
        if hist_1y.empty:
            return None

        current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        if not current_price:
            return None

        # Price performance
        perf = _calculate_performance(hist_1y, current_price)

        # Technical indicators
        technicals = _calculate_technicals(hist_1y)

        return {
            "ticker": ticker,
            "company": info.get("shortName", ticker),
            "sector": normalize_sector(info.get("sector")),
            "industry": info.get("industry", "Unknown"),
            "current_price": current_price,
            "market_cap": info.get("marketCap", 0),
            # Financials
            "revenue_growth_yoy": info.get("revenueGrowth", 0) or 0,
            "earnings_growth": info.get("earningsGrowth", 0) or 0,
            "gross_margin": info.get("grossMargins", 0) or 0,
            "operating_margin": info.get("operatingMargins", 0) or 0,
            "profit_margin": info.get("profitMargins", 0) or 0,
            "free_cash_flow": info.get("freeCashflow", 0) or 0,
            "debt_to_equity": info.get("debtToEquity", 0) or 0,
            "roe": info.get("returnOnEquity", 0) or 0,
            "roic": info.get("returnOnAssets", 0) or 0,  # Proxy
            # Valuation
            "pe_trailing": info.get("trailingPE", 0) or 0,
            "pe_forward": info.get("forwardPE", 0) or 0,
            "ps_ratio": info.get("priceToSalesTrailing12Months", 0) or 0,
            "pb_ratio": info.get("priceToBook", 0) or 0,
            "ev_ebitda": info.get("enterpriseToEbitda", 0) or 0,
            "peg_ratio": info.get("pegRatio", 0) or 0,
            # Momentum
            **perf,
            **technicals,
            # Analyst
            "analyst_rating": info.get("recommendationMean", 3.0) or 3.0,
            "analyst_count": info.get("numberOfAnalystOpinions", 0) or 0,
            "target_mean": info.get("targetMeanPrice", 0) or 0,
            "target_high": info.get("targetHighPrice", 0) or 0,
            "target_low": info.get("targetLowPrice", 0) or 0,
            # Volatility
            "beta": info.get("beta", 1.0) or 1.0,
            "hist_volatility": float(hist_1y["Close"].pct_change().std() * np.sqrt(252))
            if len(hist_1y) > 20
            else 0.25,
        }
    except Exception as e:
        logger.warning("Failed to fetch data for %s: %s", ticker, e)
        return None


def _calculate_performance(hist: pd.DataFrame, current_price: float) -> dict[str, float]:
    """Calculate price performance over multiple lookback periods."""
    result = {}
    periods = {"perf_1w": 5, "perf_1m": 21, "perf_3m": 63, "perf_6m": 126, "perf_12m": 252}
    for key, days in periods.items():
        if len(hist) >= days:
            past_price = float(hist["Close"].iloc[-days])
            result[key] = (current_price - past_price) / past_price if past_price else 0
        else:
            result[key] = 0.0
    return result


def _calculate_technicals(hist: pd.DataFrame) -> dict[str, float]:
    """Calculate RSI and moving average positioning."""
    close = hist["Close"]
    result: dict[str, float] = {}

    # RSI (14-day)
    if len(close) >= 15:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 100
        result["rsi_14"] = float(100 - (100 / (1 + rs)))
    else:
        result["rsi_14"] = 50.0

    # Moving average positioning
    current = float(close.iloc[-1])
    if len(close) >= 50:
        sma50 = float(close.rolling(50).mean().iloc[-1])
        result["above_sma50"] = 1.0 if current > sma50 else 0.0
        result["dist_sma50"] = (current - sma50) / sma50
    else:
        result["above_sma50"] = 0.5
        result["dist_sma50"] = 0.0

    if len(close) >= 200:
        sma200 = float(close.rolling(200).mean().iloc[-1])
        result["above_sma200"] = 1.0 if current > sma200 else 0.0
        result["dist_sma200"] = (current - sma200) / sma200
    else:
        result["above_sma200"] = 0.5
        result["dist_sma200"] = 0.0

    return result


def fetch_benchmark_data(ticker: str = "SPY", period: str = "1y") -> pd.DataFrame:
    """Fetch benchmark price history."""
    spy = yf.Ticker(ticker)
    return spy.history(period=period)


def fetch_batch_stock_data(tickers: list[str]) -> list[dict]:
    """Fetch data for a batch of tickers, filtering out failures."""
    results = []
    for ticker in tickers:
        data = fetch_stock_data(ticker)
        if data:
            results.append(data)
    return results
