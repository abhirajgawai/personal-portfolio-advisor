"""The US Equity/ETF Specialist node: proposes, verifies, and evaluates
US equity/ETF new-buy candidates against the New-Buy Conviction Gate.

First LLM-calling node in this graph. Fails loud, same as every prior
node - no try/except, except pydantic-ai's own bounded retry-on-
validation-failure (Agent(retries=...)), which validates output
correctness rather than swallowing a real error.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Optional

from langgraph.store.base import BaseStore
from pydantic_ai import Agent
from pydantic_ai.toolsets.filtered import FilteredToolset

from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset
from agentic_frameworks.graph.models import get_specialist_model
from agentic_frameworks.graph.nodes._candidate_memory import (
    MAX_EXCLUSION_HINTS,
    REJECTED_US_EQUITY_NS,
    _filter_previously_rejected,
    _load_recent_rejections,
    _record_rejection,
)
from agentic_frameworks.graph.nodes._specialist_helpers import (
    _apply_candidate_cap,
    _filter_to_gap_sectors,
)
from agentic_frameworks.graph.schemas import (
    GICS_SECTORS,
    CandidateEvaluation,
    ConvictionVerdict,
    ProposedCandidate,
    RejectedCandidate,
)
from agentic_frameworks.graph.state import GraphState, SpecialistState


US_SECTOR_TO_GICS = {
    "Technology": "Information Technology",
    "Semiconductors": "Information Technology",
}


def _compute_us_gaps(holdings_by_bucket: dict[str, dict[str, list[dict]]]) -> list[str]:
    """Live-compute GICS sectors with zero coverage in real US equity holdings.

    Equity only - US ETF `sector` values (e.g. "US Large Cap", "Gold")
    are asset-classification labels, not real GICS sectors, and
    including them would mean fabricating sector coverage that isn't
    actually known. Rows with sector=None are skipped (manual-total
    platform rows, a known data-quality case), not treated as
    unmapped. Any other sector string not in US_SECTOR_TO_GICS raises -
    fail loud rather than silently miscounting.
    """
    covered: set[str] = set()
    for holding in holdings_by_bucket.get("equity", {}).get("US", []):
        sector = holding["sector"]
        if sector is None:
            continue
        covered.add(US_SECTOR_TO_GICS[sector])
    return [bucket for bucket in GICS_SECTORS if bucket not in covered]


async def _verify_ticker(symbol: str, toolset: FilteredToolset) -> str | None:
    """Confirm `symbol` resolves to a real, current instrument via yfinance-mcp.

    Returns the instrument's verified `longName` when it resolves, or
    None when it doesn't - so callers get both the pass/fail signal and
    the real, verified display name in one call, with no second lookup
    and no chance of the name coming from anywhere but this verification
    step (the same "use verified identity downstream" rule the Mutual
    Fund Specialist already follows).

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
    return data.get("longName")


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
    output_type=ConvictionVerdict (not CandidateEvaluation): the model
    produces the judgment only - the node binds it to the verified
    ticker/name afterwards, so the model is never in a position to
    mislabel which candidate a verdict belongs to.
    retries=2: bounded per CLAUDE.md's LLM cost control rule 6.
    """
    system_prompt = _CONVICTION_GATE_SYSTEM_PROMPT_TEMPLATE.format(holdings_summary=holdings_summary)
    return Agent(
        model,
        output_type=ConvictionVerdict,
        system_prompt=system_prompt,
        toolsets=[toolset],
        retries=2,
    )


async def us_equity_etf_specialist_node(
    state: GraphState, *, store: Optional[BaseStore] = None
) -> SpecialistState:
    """Propose, verify, and evaluate US equity/ETF new-buy candidates.

    Reads `state["holdings_by_bucket"]` (Research node's output) to build
    a real, current US-holdings summary for the Conviction Gate's
    portfolio-fit check - not a hardcoded list, so it automatically
    reflects whatever the user actually holds by the time this node runs.

    `store` is injected by LangGraph via signature (compiled with
    `store=...` in graph.py) - a missing store is an error, not a
    silent degradation (see design spec's Error Handling): without this
    check, a call site that forgot `store=` would produce a graph that
    silently re-pays for every rejected candidate forever.

    Fails loud: any LLM/tool/verification error propagates uncaught.
    Cost control: reject-early on no gap sectors (skips model/toolset/
    store setup entirely when every GICS sector is already covered),
    reject-early on off-gap and previously-rejected candidates (ticker
    verification and the Conviction Gate evaluator run only on
    candidates that both address a real gap AND haven't already been
    rejected within REJECTION_MEMORY_DAYS), hard caps enforced in code,
    minimal per-candidate context, bounded retries on both agents.
    """
    gap_sectors = _compute_us_gaps(state["holdings_by_bucket"])
    if not gap_sectors:
        return {"us_equity_etf_candidates": []}

    if store is None:
        raise ValueError(
            "us_equity_etf_specialist_node requires a store (compile the graph with "
            "store=... - see graph.py/get_memory_store); without it, rejected "
            "candidates would be silently re-evaluated (and re-paid for) every run."
        )

    model = get_specialist_model()
    toolset = get_yfinance_toolset()
    holdings_summary = _summarize_us_holdings(state["holdings_by_bucket"])

    rejections = await _load_recent_rejections(store, REJECTED_US_EQUITY_NS)

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
                    REJECTED_US_EQUITY_NS,
                    RejectedCandidate(
                        identifier=evaluation.identifier.upper(),
                        identifier_kind="ticker",
                        specialist="us_equity",
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

    return {"us_equity_etf_candidates": evaluations}
