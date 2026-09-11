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
from pydantic_ai.toolsets.filtered import FilteredToolset

from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset
from agentic_frameworks.graph.models import get_specialist_model
from agentic_frameworks.graph.schemas import (
    US_GAP_SECTORS,
    CandidateEvaluation,
    ProposedCandidate,
)
from agentic_frameworks.graph.state import GraphState, SpecialistState


async def _verify_ticker(symbol: str, toolset: FilteredToolset) -> bool:
    """Confirm `symbol` resolves to a real, current instrument via yfinance-mcp.

    A nonexistent/delisted ticker doesn't raise - yfinance-mcp returns a
    near-empty payload instead (confirmed live: {"trailingPegRatio": null}
    for a fake symbol). `longName` being present is the reliable signal
    a real instrument was found.

    `direct_call_tool` is an MCPToolset-only method (not present on the
    FilteredToolset wrapper `get_yfinance_toolset` now returns), so this
    calls it on `toolset.wrapped` - the underlying MCPToolset. This is a
    direct, out-of-band verification call, not a tool the LLM invokes
    through its registered toolset, so bypassing the exclusion filter
    here is correct: `yfinance_get_ticker_info` isn't one of the excluded
    tools anyway.
    """
    result = await toolset.wrapped.direct_call_tool("yfinance_get_ticker_info", {"symbol": symbol})
    data = json.loads(result)
    return data.get("longName") is not None


MAX_CANDIDATES_PER_SECTOR = 3
MAX_CANDIDATES_TOTAL = 5


def _summarize_us_holdings(holdings_by_bucket: dict[str, dict[str, list[dict]]]) -> str:
    """Build a real, current summary of US equity/ETF holdings for the portfolio-fit check.

    Built fresh from `holdings_by_bucket` on every node invocation - never
    a hardcoded list - so it automatically reflects whatever the user
    actually holds whenever this node runs, with no manual updates needed
    as the portfolio changes.
    """
    lines = []
    for asset_class in ("equity", "etf"):
        for holding in holdings_by_bucket.get(asset_class, {}).get("US", []):
            lines.append(f"- {holding['ticker']} ({holding['name']}), sector: {holding['sector']}")
    if not lines:
        return "The user currently holds no US equities or ETFs."
    return "The user's current US equity/ETF holdings are:\n" + "\n".join(lines)

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


_CONVICTION_GATE_SYSTEM_PROMPT_TEMPLATE = """You are evaluating one US equity/ETF new-buy candidate for a \
personal, long-term portfolio advisor, using the New-Buy Conviction \
Gate. Run these four checks in order, stopping at the first failure:

1. Portfolio-fit: does this candidate address a real, identified gap \
in the user's current holdings? {holdings_summary}
2. Fundamentals: use your tools to check real financials (revenue, \
earnings, debt, margins) - not assumptions.
3. Valuation: is it reasonably priced now versus its own history and \
peers? Richly-valued candidates should be flagged or rejected, not \
rationalized.
4. Style-fit: is it consistent with a long-term, buy-and-hold \
investing approach - no day-trading or speculative framing.

Avoid is the default, not buy: if any check fails, the action should \
reflect that (e.g. Hold or Reduce, not Buy), with the specific reason \
in risk_flags. This is advisory only - your output is never a trade \
instruction, only a recommendation for the user to act on manually."""


def _build_conviction_gate_evaluator(model, toolset: FilteredToolset, holdings_summary: str) -> Agent:
    """Build the Conviction Gate evaluator Agent.

    holdings_summary: real, current US holdings (see _summarize_us_holdings)
    baked into the system prompt so the portfolio-fit check has actual
    data to check against, instead of just being told a sector is a gap
    and asked to confirm it. Built once per node invocation and reused
    across every candidate evaluated in that run, so it stays one stable,
    cacheable string within a run (per CLAUDE.md's LLM cost control rule 3)
    even though it varies run-to-run as the real portfolio changes.
    toolsets=[toolset] gives this agent real fundamentals/valuation
    data access via yfinance-mcp for the fundamentals/valuation checks.
    retries=2: bounded per CLAUDE.md's LLM cost control rule 6.
    """
    system_prompt = _CONVICTION_GATE_SYSTEM_PROMPT_TEMPLATE.format(holdings_summary=holdings_summary)
    return Agent(
        model,
        output_type=CandidateEvaluation,
        system_prompt=system_prompt,
        toolsets=[toolset],
        retries=2,
    )


async def us_equity_etf_specialist_node(state: GraphState) -> SpecialistState:
    """Propose, verify, and evaluate US equity/ETF new-buy candidates.

    Reads `state["holdings_by_bucket"]` (Research node's output) to build
    a real, current US-holdings summary for the Conviction Gate's
    portfolio-fit check - not a hardcoded list, so it automatically
    reflects whatever the user actually holds by the time this node runs.

    Fails loud: any LLM/tool/verification error propagates uncaught.
    Cost control: reject-early (ticker verification before the Conviction
    Gate evaluator ever runs), hard caps enforced in code, minimal
    per-candidate context, bounded retries on both agents.
    """
    model = get_specialist_model()
    toolset = get_yfinance_toolset()
    holdings_summary = _summarize_us_holdings(state["holdings_by_bucket"])

    async with toolset:
        generator = _build_candidate_generator(model)
        gap_sectors_text = ", ".join(US_GAP_SECTORS)
        generation_result = await generator.run(
            f"Propose candidates for these gap sectors: {gap_sectors_text}"
        )
        capped_candidates = _apply_candidate_cap(generation_result.output)

        verified_candidates = [
            candidate
            for candidate in capped_candidates
            if await _verify_ticker(candidate.ticker, toolset)
        ]

        evaluator = _build_conviction_gate_evaluator(model, toolset, holdings_summary)
        evaluations = []
        for candidate in verified_candidates:
            eval_result = await evaluator.run(
                f"Evaluate ticker: {candidate.ticker}, gap sector: {candidate.sector}"
            )
            evaluations.append(eval_result.output.model_dump())

    return {"us_equity_etf_candidates": evaluations}
