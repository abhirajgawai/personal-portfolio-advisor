"""The India Equity Specialist node: proposes, verifies, and evaluates
India equity new-buy candidates against the New-Buy Conviction Gate.

Same three-stage pipeline as the US Equity/ETF Specialist
(nodes/us_equity_etf_specialist.py): live gap detection -> candidate
generation -> ticker verification -> Conviction Gate evaluation.
Fails loud, same as every other node in this graph.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Optional

from langgraph.store.base import BaseStore
from pydantic_ai import Agent
from pydantic_ai.toolsets.filtered import FilteredToolset

from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset
from agentic_frameworks.graph.models import get_specialist_model
from agentic_frameworks.graph.nodes._candidate_memory import (
    MAX_EXCLUSION_HINTS,
    REJECTED_INDIA_EQUITY_NS,
    _filter_previously_rejected,
    _load_recent_rejections,
    _record_rejection,
)
from agentic_frameworks.graph.nodes._specialist_helpers import (
    _apply_candidate_cap,
    _filter_to_gap_sectors,
)
from agentic_frameworks.graph.nodes.us_equity_etf_specialist import _verify_ticker
from agentic_frameworks.graph.schemas import (
    CandidateEvaluation,
    ConvictionVerdict,
    GICS_SECTORS,
    ProposedCandidate,
    RejectedCandidate,
)
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
    """Live-compute GICS sectors with zero coverage in the real portfolio.

    Rolls up IN equity holdings by sector (as before). Additionally
    treats "Real Estate" as covered if the user holds any IN real
    estate (holdings_by_bucket["real_estate"]["IN"]) - this is a
    directly known fact from the data, unlike mutual fund `sector`
    values (fund category, not GICS sector) which stay excluded
    because including them would mean fabricating coverage that isn't
    actually known. Rows with sector=None are skipped (the two
    MANUAL_TOTAL platform-total rows, a known data-quality case, not a
    real gap). Any other equity sector string not in
    IN_SECTOR_TO_GICS raises - fail loud rather than silently
    miscounting.
    """
    covered: set[str] = set()
    for holding in holdings_by_bucket.get("equity", {}).get("IN", []):
        sector = holding["sector"]
        if sector is None:
            continue
        covered.add(IN_SECTOR_TO_GICS[sector])
    if holdings_by_bucket.get("real_estate", {}).get("IN", []):
        covered.add("Real Estate")
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
    per CLAUDE.md's LLM cost control rule 6, and output_type=
    ConvictionVerdict (judgment only - the node binds the verdict to the
    verified ticker/name afterwards, so the model can never mislabel
    which candidate a verdict belongs to).
    """
    system_prompt = _CONVICTION_GATE_SYSTEM_PROMPT_TEMPLATE.format(holdings_summary=holdings_summary)
    return Agent(
        model,
        output_type=ConvictionVerdict,
        system_prompt=system_prompt,
        toolsets=[toolset],
        retries=2,
    )


async def india_equity_specialist_node(
    state: GraphState, *, store: Optional[BaseStore] = None
) -> SpecialistState:
    """Propose, verify, and evaluate India equity new-buy candidates.

    Same three-stage pipeline as us_equity_etf_specialist_node, with
    gap sectors computed live from real holdings_by_bucket rather than
    a static file/hardcoded list, plus the same rejection-memory
    read/filter/write points (see design spec). `store` injection and
    the fail-loud `store is None` guard mirror the US specialist
    exactly - see its docstring for the reasoning.

    Fails loud: any LLM/tool/verification error propagates uncaught.
    Reject-early on no gap sectors (skips model/toolset/store setup
    entirely when every GICS sector is already covered) and on off-gap
    or previously-rejected candidates (ticker verification and the
    Conviction Gate evaluator run only on candidates that both address
    a real gap AND haven't already been rejected within
    REJECTION_MEMORY_DAYS).
    """
    gap_sectors = _compute_india_equity_gaps(state["holdings_by_bucket"])
    if not gap_sectors:
        return {"india_equity_candidates": []}

    if store is None:
        raise ValueError(
            "india_equity_specialist_node requires a store (compile the graph with "
            "store=... - see graph.py/get_memory_store); without it, rejected "
            "candidates would be silently re-evaluated (and re-paid for) every run."
        )

    model = get_specialist_model()
    toolset = get_yfinance_toolset()
    holdings_summary = _summarize_india_equity_holdings(state["holdings_by_bucket"])

    rejections = await _load_recent_rejections(store, REJECTED_INDIA_EQUITY_NS)

    async with toolset:
        generator = _build_candidate_generator(model)
        gap_sectors_text = ", ".join(gap_sectors)
        prompt = f"Propose candidates for these gap sectors: {gap_sectors_text}"
        if rejections:
            excluded = sorted({r.display_name for r in rejections.values()})[:MAX_EXCLUSION_HINTS]
            prompt += f"\nDo not propose these already-rejected tickers/names: {', '.join(excluded)}."
        generation_result = await generator.run(prompt)
        on_gap_candidates = _filter_to_gap_sectors(generation_result.output, gap_sectors)
        not_previously_rejected = _filter_previously_rejected(
            on_gap_candidates, rejections, key=lambda c: c.ticker.upper()
        )
        capped_candidates = _apply_candidate_cap(not_previously_rejected)

        verified_candidates = []
        for candidate in capped_candidates:
            verified_name = await _verify_ticker(candidate.ticker, toolset)
            if verified_name is not None:
                verified_candidates.append((candidate, verified_name))

        evaluator = _build_conviction_gate_evaluator(model, toolset, holdings_summary)
        evaluations = []
        for candidate, verified_name in verified_candidates:
            eval_result = await evaluator.run(
                f"Evaluate ticker: {candidate.ticker}, gap sector: {candidate.sector}"
            )
            evaluation = CandidateEvaluation.from_verdict(
                eval_result.output,
                identifier=candidate.ticker,
                identifier_kind="ticker",
                display_name=verified_name,
            )
            evaluations.append(evaluation.model_dump())

            if evaluation.action != "Buy":
                await _record_rejection(
                    store,
                    REJECTED_INDIA_EQUITY_NS,
                    RejectedCandidate(
                        identifier=evaluation.identifier.upper(),
                        identifier_kind="ticker",
                        specialist="india_equity",
                        display_name=evaluation.display_name,
                        group=candidate.sector,
                        action=evaluation.action,
                        confidence=evaluation.confidence,
                        risk_flags=evaluation.risk_flags,
                        reasoning=evaluation.reasoning,
                        rejected_at=datetime.now(UTC).isoformat(),
                        thread_id=None,
                        schema_version=1,
                    ),
                )

    return {"india_equity_candidates": evaluations}
