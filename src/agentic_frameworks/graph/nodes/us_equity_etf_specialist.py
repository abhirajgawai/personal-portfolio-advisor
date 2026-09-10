"""The US Equity/ETF Specialist node: proposes, verifies, and evaluates
US equity/ETF new-buy candidates against the New-Buy Conviction Gate.

First LLM-calling node in this graph. Fails loud, same as every prior
node - no try/except, except pydantic-ai's own bounded retry-on-
validation-failure (Agent(retries=...)), which validates output
correctness rather than swallowing a real error.
"""

from __future__ import annotations

import json

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

from agentic_frameworks.graph.schemas import ProposedCandidate


async def _verify_ticker(symbol: str, toolset: MCPToolset) -> bool:
    """Confirm `symbol` resolves to a real, current instrument via yfinance-mcp.

    A nonexistent/delisted ticker doesn't raise - yfinance-mcp returns a
    near-empty payload instead (confirmed live: {"trailingPegRatio": null}
    for a fake symbol). `longName` being present is the reliable signal
    a real instrument was found.
    """
    result = await toolset.direct_call_tool("yfinance_get_ticker_info", {"symbol": symbol})
    data = json.loads(result)
    return data.get("longName") is not None


US_GAP_SECTORS = [
    "Health Care",
    "Financials",
    "Energy",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Real Estate",
    "Utilities",
    "Communication Services",
    "Materials",
]

MAX_CANDIDATES_PER_SECTOR = 3
MAX_CANDIDATES_TOTAL = 5

_CANDIDATE_GENERATOR_SYSTEM_PROMPT = """You are a US equity/ETF research assistant for a personal \
long-term portfolio advisor. Given a list of sector gaps in the user's \
current US holdings, propose real, currently-traded US-listed stock or \
ETF tickers that could fill each gap - up to 3 tickers per gap sector.

Only propose tickers you are confident are real, currently-listed US \
securities. Do not propose delisted, speculative, or fictional tickers. \
For each candidate, name the ticker symbol and the gap sector it addresses."""


def _build_candidate_generator(model) -> Agent:
    """Build the candidate-generator Agent.

    retries=2: bounded per CLAUDE.md's LLM cost control rule 6 - never
    left at the library default.
    """
    return Agent(
        model,
        output_type=list[ProposedCandidate],
        system_prompt=_CANDIDATE_GENERATOR_SYSTEM_PROMPT,
        retries=2,
    )


def _apply_candidate_cap(candidates: list[ProposedCandidate]) -> list[ProposedCandidate]:
    """Enforce the hard cap in code - never trust the model's own count."""
    per_sector_counts: dict[str, int] = {}
    capped: list[ProposedCandidate] = []
    for candidate in candidates:
        per_sector_counts.setdefault(candidate.sector, 0)
        if per_sector_counts[candidate.sector] < MAX_CANDIDATES_PER_SECTOR:
            capped.append(candidate)
            per_sector_counts[candidate.sector] += 1
    return capped[:MAX_CANDIDATES_TOTAL]
