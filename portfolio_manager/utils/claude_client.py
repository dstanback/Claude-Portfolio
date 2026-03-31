"""Claude API client for AI-powered analysis stages."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import anthropic

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-4-6"


@dataclass
class AgentResult:
    """Result from a Claude agent analysis."""
    role: str  # "bull" or "bear"
    ticker: str
    arguments: list[str]
    confidence: int
    sources: list[str]
    raw_response: str


def get_client() -> anthropic.Anthropic:
    """Get an Anthropic API client."""
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def run_analysis_prompt(
    prompt: str,
    system: str = "",
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
) -> str:
    """Run a single analysis prompt through Claude and return the response text."""
    client = get_client()
    messages = [{"role": "user", "content": prompt}]
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    return response.content[0].text


def run_bull_bear_analysis(
    ticker: str,
    company: str,
    stock_data: dict,
    role: str,
    agent_count: int = 15,
    model: str = DEFAULT_MODEL,
) -> AgentResult:
    """Run bull or bear analysis for a stock using Claude.

    Simulates multiple independent research agents by asking Claude to
    generate analysis from N independent perspectives.
    """
    role_instruction = {
        "bull": (
            f"You are a team of {agent_count} independent bullish equity research analysts. "
            "Build the strongest possible case for buying this stock. Focus on catalysts, "
            "competitive advantages, underappreciated growth drivers, and favorable risk/reward. "
            "Each analyst should contribute a unique argument with a specific source."
        ),
        "bear": (
            f"You are a team of {agent_count} independent bearish equity research analysts. "
            "Build the strongest possible case for why this stock should be sold or avoided. "
            "Focus on overvaluation risks, competitive threats, margin compression, "
            "macro headwinds, and management concerns. "
            "Each analyst should contribute a unique argument with a specific source."
        ),
    }

    prompt = f"""Analyze {ticker} ({company}) from a {role} perspective.

Current stock data:
{json.dumps(stock_data, indent=2, default=str)}

CONSTRAINTS:
- Only consider information from the last 7 calendar days
- Each argument must cite a specific source (article title, publication, date)
- Provide exactly {agent_count} independent arguments

Respond in this exact JSON format:
{{
    "arguments": ["argument 1 with source citation", "argument 2 with source citation", ...],
    "confidence": <0-100 integer representing team confidence>,
    "sources": ["source 1", "source 2", ...]
}}
"""

    system = role_instruction[role]
    try:
        response_text = run_analysis_prompt(prompt, system=system, model=model)
        # Parse JSON from response
        parsed = _extract_json(response_text)
        return AgentResult(
            role=role,
            ticker=ticker,
            arguments=parsed.get("arguments", []),
            confidence=parsed.get("confidence", 50),
            sources=parsed.get("sources", []),
            raw_response=response_text,
        )
    except Exception as e:
        logger.error("Bull/bear analysis failed for %s (%s): %s", ticker, role, e)
        return AgentResult(
            role=role,
            ticker=ticker,
            arguments=[f"Analysis failed: {e}"],
            confidence=50,
            sources=[],
            raw_response="",
        )


def run_scenario_analysis(
    ticker: str,
    company: str,
    stock_data: dict,
    adversarial_result: dict,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Generate probability-weighted scenario models for a stock."""
    prompt = f"""You are a quantitative equity analyst building scenario models.

Stock: {ticker} ({company})
Current Price: ${stock_data.get('current_price', 0):.2f}

Stock Data:
{json.dumps(stock_data, indent=2, default=str)}

Adversarial Research Summary:
{json.dumps(adversarial_result, indent=2, default=str)}

Build three scenarios (Bull, Base, Bear) with:
1. Probability (Bull: 20-35%, Base: 40-55%, Bear: 15-30%, must sum to 100%)
2. Price targets at 1M, 3M, 6M, 12M horizons
3. Key assumptions for each scenario

SELF-DEBATE CHECKPOINT before finalizing:
- Am I anchoring to a narrative?
- Is my base case actually just a disguised bull case?
- What would make me completely wrong?

Respond in this exact JSON format:
{{
    "scenarios": {{
        "bull": {{
            "probability": <float>,
            "targets": {{"1m": <float>, "3m": <float>, "6m": <float>, "12m": <float>}},
            "assumptions": "<string>"
        }},
        "base": {{
            "probability": <float>,
            "targets": {{"1m": <float>, "3m": <float>, "6m": <float>, "12m": <float>}},
            "assumptions": "<string>"
        }},
        "bear": {{
            "probability": <float>,
            "targets": {{"1m": <float>, "3m": <float>, "6m": <float>, "12m": <float>}},
            "assumptions": "<string>"
        }}
    }},
    "self_debate_notes": "<string>"
}}
"""

    system = (
        "You are a rigorous quantitative analyst. Your scenario models must be "
        "internally consistent and grounded in the data provided. Challenge your "
        "own assumptions before finalizing."
    )

    try:
        response_text = run_analysis_prompt(prompt, system=system, model=model)
        return _extract_json(response_text)
    except Exception as e:
        logger.error("Scenario analysis failed for %s: %s", ticker, e)
        return _default_scenario(stock_data.get("current_price", 100))


def _default_scenario(current_price: float) -> dict:
    """Return a neutral default scenario when analysis fails."""
    return {
        "scenarios": {
            "bull": {
                "probability": 0.25,
                "targets": {
                    "1m": current_price * 1.05,
                    "3m": current_price * 1.10,
                    "6m": current_price * 1.15,
                    "12m": current_price * 1.20,
                },
                "assumptions": "Default bull scenario",
            },
            "base": {
                "probability": 0.50,
                "targets": {
                    "1m": current_price * 1.01,
                    "3m": current_price * 1.03,
                    "6m": current_price * 1.05,
                    "12m": current_price * 1.08,
                },
                "assumptions": "Default base scenario",
            },
            "bear": {
                "probability": 0.25,
                "targets": {
                    "1m": current_price * 0.95,
                    "3m": current_price * 0.90,
                    "6m": current_price * 0.85,
                    "12m": current_price * 0.80,
                },
                "assumptions": "Default bear scenario",
            },
        },
        "self_debate_notes": "Default scenario used due to analysis failure.",
    }


def _extract_json(text: str) -> dict:
    """Extract JSON object from a text response that may contain markdown."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        return json.loads(text[start:end].strip())
    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        return json.loads(text[start:end].strip())

    # Try finding JSON object boundaries
    brace_start = text.find("{")
    brace_end = text.rfind("}") + 1
    if brace_start >= 0 and brace_end > brace_start:
        return json.loads(text[brace_start:brace_end])

    raise ValueError(f"Could not extract JSON from response: {text[:200]}...")
