"""The India Equity Specialist node: proposes, verifies, and evaluates
India equity new-buy candidates against the New-Buy Conviction Gate.

Same three-stage pipeline as the US Equity/ETF Specialist
(nodes/us_equity_etf_specialist.py): live gap detection -> candidate
generation -> ticker verification -> Conviction Gate evaluation.
Fails loud, same as every other node in this graph.
"""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.toolsets.filtered import FilteredToolset

from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset
from agentic_frameworks.graph.models import get_specialist_model
from agentic_frameworks.graph.nodes._specialist_helpers import _apply_candidate_cap
from agentic_frameworks.graph.nodes.us_equity_etf_specialist import _verify_ticker
from agentic_frameworks.graph.schemas import CandidateEvaluation, GICS_SECTORS, ProposedCandidate
from agentic_frameworks.graph.state import GraphState, SpecialistState

IN_SECTOR_TO_GICS = {
    "Industrial Equipment": "Industrials",
    "Financial Services": "Financials",
    "Pharmaceuticals": "Health Care",
    "Industrials": "Industrials",
    "Steel": "Materials",
    "Water & Environment": "Utilities",
    "Consumer Packaging": "Consumer Staples",
    "Renewables": "Utilities",
    "Power": "Utilities",
    "Auto Components": "Consumer Discretionary",
    "Information Technology": "Information Technology",
    "Banking": "Financials",
    "Agri & Sugar": "Consumer Staples",
    "Chemicals": "Materials",
    "Electrical Equipment": "Industrials",
    "Infrastructure": "Industrials",
    "Media & Entertainment": "Communication Services",
    "Travel & Tourism": "Consumer Discretionary",
}


def _compute_india_equity_gaps(holdings_by_bucket: dict[str, dict[str, list[dict]]]) -> list[str]:
    """Live-compute GICS sectors with zero coverage in real IN equity holdings.

    IN equity only - mutual funds and real estate are excluded (fund
    `sector` values are fund category, not GICS sector, and real
    estate isn't sector-tagged at all; including either would mean
    fabricating coverage that isn't actually known). Rows with
    sector=None are skipped (the two MANUAL_TOTAL platform-total rows,
    a known data-quality case, not a real gap). Any other sector
    string not in IN_SECTOR_TO_GICS raises - fail loud rather than
    silently miscounting.
    """
    covered: set[str] = set()
    for holding in holdings_by_bucket.get("equity", {}).get("IN", []):
        sector = holding["sector"]
        if sector is None:
            continue
        covered.add(IN_SECTOR_TO_GICS[sector])
    return [bucket for bucket in GICS_SECTORS if bucket not in covered]


def _summarize_india_equity_holdings(holdings_by_bucket: dict[str, dict[str, list[dict]]]) -> str:
    """Build a real, current summary of IN equity holdings for the portfolio-fit check.

    Built fresh from holdings_by_bucket on every node invocation - never
    a hardcoded list - so it automatically reflects whatever the user
    actually holds whenever this node runs.
    """
    lines = []
    for holding in holdings_by_bucket.get("equity", {}).get("IN", []):
        lines.append(f"- {holding['ticker']} ({holding['name']}), sector: {holding['sector']}")
    if not lines:
        return "The user currently holds no India equities."
    return "The user's current India equity holdings are:\n" + "\n".join(lines)


_CANDIDATE_GENERATOR_SYSTEM_PROMPT = """You are an India equity research assistant for a personal \
long-term portfolio advisor. Given a list of sector gaps in the user's \
current India equity holdings, propose real, currently-traded \
NSE-listed stock tickers (using the .NS suffix, e.g. RELIANCE.NS) \
that could fill each gap - up to 3 tickers per gap sector.

Only propose tickers you are confident are real, currently-listed NSE \
securities. Do not propose delisted, speculative, or fictional tickers. \
For each candidate, name the ticker symbol (with .NS suffix) and the \
gap sector it addresses."""


def _build_candidate_generator(model) -> Agent:
    """Build the India candidate-generator Agent.

    retries=2: bounded per CLAUDE.md's LLM cost control rule 6 - never
    left at the library default.
    """
    return Agent(
        model,
        output_type=list[ProposedCandidate],
        system_prompt=_CANDIDATE_GENERATOR_SYSTEM_PROMPT,
        retries=2,
    )


_CONVICTION_GATE_SYSTEM_PROMPT_TEMPLATE = """You are evaluating one India equity new-buy candidate for a \
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
    """Build the India Conviction Gate evaluator Agent.

    Same structure as the US specialist's evaluator - holdings_summary
    baked into the system prompt for a real portfolio-fit check,
    toolsets=[toolset] for real fundamentals/valuation data via
    yfinance-mcp (already confirmed to resolve .NS tickers), retries=2
    per CLAUDE.md's LLM cost control rule 6.
    """
    system_prompt = _CONVICTION_GATE_SYSTEM_PROMPT_TEMPLATE.format(holdings_summary=holdings_summary)
    return Agent(
        model,
        output_type=CandidateEvaluation,
        system_prompt=system_prompt,
        toolsets=[toolset],
        retries=2,
    )


async def india_equity_specialist_node(state: GraphState) -> SpecialistState:
    """Propose, verify, and evaluate India equity new-buy candidates.

    Same three-stage pipeline as us_equity_etf_specialist_node, with
    gap sectors computed live from real holdings_by_bucket rather than
    a static file/hardcoded list. Fails loud: any LLM/tool/verification
    error propagates uncaught.
    """
    model = get_specialist_model()
    toolset = get_yfinance_toolset()
    holdings_summary = _summarize_india_equity_holdings(state["holdings_by_bucket"])

    async with toolset:
        generator = _build_candidate_generator(model)
        gap_sectors = _compute_india_equity_gaps(state["holdings_by_bucket"])
        gap_sectors_text = ", ".join(gap_sectors)
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

    return {"india_equity_candidates": evaluations}
