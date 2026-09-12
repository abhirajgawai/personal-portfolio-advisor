"""The Mutual Fund Specialist node: proposes and evaluates new-buy
mutual fund candidates for live-computed category gaps, mirroring the
US/India equity specialists' three-stage pipeline (propose -> verify ->
evaluate). Fails loud, same as every other node.
"""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.toolsets.filtered import FilteredToolset

from agentic_frameworks.graph.mcp_clients import get_mfapi_toolset, get_yfinance_toolset
from agentic_frameworks.graph.models import get_specialist_model
from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
    MF_CATEGORY_BENCHMARK_TICKERS,
    _compute_benchmark_difference,
    _compute_fund_metrics,
    _compute_mf_category_gaps,
    _fetch_nav_history,
    _resolve_scheme_by_name_and_direct_plan,
)
from agentic_frameworks.graph.nodes._specialist_helpers import (
    _apply_candidate_cap,
    _filter_to_gap_sectors,
)
from agentic_frameworks.graph.nodes.mutual_fund_holdings_analyzer import (
    _format_metrics_for_prompt,
)
from agentic_frameworks.graph.schemas import CandidateEvaluation, ProposedFundCandidate
from agentic_frameworks.graph.state import GraphState, SpecialistState

_CANDIDATE_GENERATOR_SYSTEM_PROMPT = """You are an India mutual fund research assistant for a personal \
long-term portfolio advisor. Given a list of category gaps in the \
user's current mutual fund holdings, propose real, currently-active, \
Direct-plan Indian mutual fund schemes that could fill each gap - up \
to 3 schemes per gap category.

Always propose the Direct plan variant by name (not Regular plan) - \
Direct plans avoid distributor commission and are the correct \
long-term choice. Only propose funds you are confident are real, \
currently-active schemes. Do not propose discontinued or fictional \
funds. For each candidate, name the scheme (including "Direct Plan" \
in the name) and the gap category it addresses."""


def _build_candidate_generator(model) -> Agent:
    """retries=2: bounded per CLAUDE.md's LLM cost control rule 6."""
    return Agent(
        model,
        output_type=list[ProposedFundCandidate],
        system_prompt=_CANDIDATE_GENERATOR_SYSTEM_PROMPT,
        retries=2,
    )


_NEW_BUY_CONVICTION_GATE_SYSTEM_PROMPT = """You are evaluating one new-buy India mutual fund candidate for a \
personal, long-term portfolio advisor, using a 7-check Conviction Gate \
adapted for mutual funds (not equities - no price/valuation checks apply). \
Run these checks in order:

1. Portfolio-fit: does this fund address a real, identified category gap \
in the user's current holdings?
2. Cost: only propose/favor Direct plan funds - Direct plans avoid \
distributor commission drag (exact expense ratios are not available from \
this project's data sources, so this stays a qualitative check).
3. Performance vs benchmark: did it beat its benchmark over its own \
history, where a verified benchmark exists? If unavailable, say so \
explicitly rather than guessing.
4. Risk-adjusted performance: weigh volatility and max drawdown, not just \
raw returns.
5. Track-record length: a short track record (under ~3 years) means \
performance-consistency claims should carry less weight.
6. Style-fit: is this fund's mandate consistent with a long-term, \
buy-and-hold approach?

Avoid is the default, not buy: if checks are weak or data is missing, the \
action should reflect that (Hold, not Buy), with the specific reason in \
risk_flags. Tax context (India, current rules): equity-oriented mutual \
funds (>=65% equity) are taxed as equity capital gains; other funds are \
taxed at the investor's income slab rate regardless of holding period; \
ELSS funds carry a mandatory 3-year lock-in. Your output is advisory only \
- never a trade instruction, only a recommendation for the user to act on \
manually."""


def _build_conviction_gate_evaluator(model, toolset: FilteredToolset) -> Agent:
    """retries=2: bounded per CLAUDE.md's LLM cost control rule 6."""
    return Agent(
        model,
        output_type=CandidateEvaluation,
        system_prompt=_NEW_BUY_CONVICTION_GATE_SYSTEM_PROMPT,
        toolsets=[toolset],
        retries=2,
    )


async def mutual_fund_specialist_node(state: GraphState) -> SpecialistState:
    """Propose, verify, and evaluate new-buy mutual fund candidates.

    Fails loud: any LLM/tool/verification error propagates uncaught,
    except a candidate whose proposed name has no real Direct-plan
    match (ValueError from _resolve_scheme_by_name_and_direct_plan),
    which is dropped - the same reject-early role _verify_ticker plays
    for the equity specialists. Cost control: reject-early (empty gap
    list short-circuits before any model/toolset setup; candidates are
    filtered to real gap categories and capped before the expensive
    resolve/metrics/evaluate steps run).
    """
    gap_categories = _compute_mf_category_gaps(state["holdings_by_bucket"])
    if not gap_categories:
        return {"mutual_fund_candidates": []}

    model = get_specialist_model()
    mfapi_toolset = get_mfapi_toolset()
    yfinance_toolset = get_yfinance_toolset()

    async with mfapi_toolset, yfinance_toolset:
        generator = _build_candidate_generator(model)
        gap_categories_text = ", ".join(gap_categories)
        generation_result = await generator.run(
            f"Propose candidates for these gap categories: {gap_categories_text}"
        )
        on_gap_candidates = _filter_to_gap_sectors(
            generation_result.output, gap_categories, key=lambda c: c.category
        )
        capped_candidates = _apply_candidate_cap(on_gap_candidates, key=lambda c: c.category)

        evaluator = _build_conviction_gate_evaluator(model, yfinance_toolset)
        evaluations = []
        for candidate in capped_candidates:
            try:
                scheme_code, _resolved_category = await _resolve_scheme_by_name_and_direct_plan(
                    candidate.scheme_name, mfapi_toolset
                )
            except ValueError:
                continue

            history = await _fetch_nav_history(scheme_code, mfapi_toolset)
            metrics = _compute_fund_metrics(history)
            benchmark_ticker = MF_CATEGORY_BENCHMARK_TICKERS.get(candidate.category)
            benchmark = await _compute_benchmark_difference(history, benchmark_ticker, yfinance_toolset)
            metrics_text = _format_metrics_for_prompt(metrics, benchmark, {})

            eval_result = await evaluator.run(
                f"Evaluate candidate: {candidate.scheme_name} (category: {candidate.category}).\n{metrics_text}"
            )
            evaluations.append(eval_result.output.model_dump())

    return {"mutual_fund_candidates": evaluations}
