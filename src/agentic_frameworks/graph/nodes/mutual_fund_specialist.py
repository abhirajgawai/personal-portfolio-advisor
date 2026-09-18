"""The Mutual Fund Specialist node: proposes and evaluates new-buy
mutual fund candidates for live-computed category gaps, mirroring the
US/India equity specialists' three-stage pipeline (propose -> verify ->
evaluate). Fails loud, same as every other node.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Optional

from langgraph.store.base import BaseStore
from pydantic_ai import Agent
from pydantic_ai.toolsets.filtered import FilteredToolset

from agentic_frameworks.graph.mcp_clients import get_mfapi_toolset, get_yfinance_toolset
from agentic_frameworks.graph.models import get_specialist_model
from agentic_frameworks.graph.nodes._candidate_memory import (
    MAX_EXCLUSION_HINTS,
    REJECTED_MUTUAL_FUND_BY_NAME_NS,
    REJECTED_MUTUAL_FUND_NS,
    _cache_resolved_scheme,
    _filter_previously_rejected,
    _load_recent_rejections,
    _load_resolved_scheme,
    _normalize_scheme_name,
    _record_rejection,
)
from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
    MF_CATEGORY_BENCHMARK_TICKERS,
    _category_matches_taxonomy,
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
from agentic_frameworks.graph.schemas import (
    CandidateEvaluation,
    ConvictionVerdict,
    ProposedFundCandidate,
    RejectedCandidate,
)
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


def _summarize_mf_holdings(holdings_by_bucket: dict[str, dict[str, list[dict]]]) -> str:
    """Build a real, current summary of IN mutual fund holdings for the
    portfolio-fit check, mirroring _summarize_us_holdings's style
    (us_equity_etf_specialist.py) but for mutual funds (name + category,
    not ticker + GICS sector).

    Built fresh from holdings_by_bucket on every node invocation - never
    a hardcoded list - so it automatically reflects whatever the user
    actually holds whenever this node runs.
    """
    lines = []
    for holding in holdings_by_bucket.get("mutual_fund", {}).get("IN", []):
        lines.append(f"- {holding['name']} (category: {holding['sector']})")
    if not lines:
        return "The user currently holds no India mutual funds."
    return "The user's current India mutual fund holdings are:\n" + "\n".join(lines)


_NEW_BUY_CONVICTION_GATE_SYSTEM_PROMPT_TEMPLATE = """You are evaluating one new-buy India mutual fund candidate for a \
personal, long-term portfolio advisor, using a 7-check Conviction Gate \
adapted for mutual funds (not equities - no price/valuation checks apply). \
Run these checks in order:

1. Portfolio-fit: does this fund address a real, identified category gap \
in the user's current holdings? {holdings_summary} The identified \
category gaps this cycle are: {gap_categories_text}.
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


def _build_conviction_gate_evaluator(
    model, toolset: FilteredToolset, holdings_summary: str, gap_categories_text: str
) -> Agent:
    """holdings_summary/gap_categories_text: real, current MF holdings and
    the real computed category gaps for this run, baked into the system
    prompt so the portfolio-fit check has actual data to check against
    (mirrors the US/India equity specialists' holdings_summary pattern).
    Built once per node invocation and reused across every candidate
    evaluated in that run, so it stays one stable, cacheable string within
    a run (per CLAUDE.md's LLM cost control rule 3).
    retries=2: bounded per CLAUDE.md's LLM cost control rule 6.
    output_type=ConvictionVerdict: judgment only - the node binds each
    verdict to the MFapi-resolved, category-checked scheme code and its
    real scheme name, so the model is never in a position to mislabel
    which fund a verdict belongs to.
    """
    system_prompt = _NEW_BUY_CONVICTION_GATE_SYSTEM_PROMPT_TEMPLATE.format(
        holdings_summary=holdings_summary, gap_categories_text=gap_categories_text
    )
    return Agent(
        model,
        output_type=ConvictionVerdict,
        system_prompt=system_prompt,
        toolsets=[toolset],
        retries=2,
    )


async def mutual_fund_specialist_node(
    state: GraphState, *, store: Optional[BaseStore] = None
) -> SpecialistState:
    """Propose, verify, and evaluate new-buy mutual fund candidates.

    `store` is injected by LangGraph via signature; a missing store
    raises (see design spec's Error Handling, and the equity
    specialists' identical guard). Rejection memory here is two-stage,
    per the design spec: a cheap pre-cap filter on the normalized
    proposed name (catches the common exact-repeat case for free) and
    the authoritative post-resolution filter on the real `scheme_code`,
    run just before the paid evaluator call. Both alias and canonical
    rejection records are written on a non-Buy verdict.

    Also caches the free-but-slow `_resolve_scheme_by_name_and_direct_
    plan` name resolution in the `resolved_schemes/mutual_fund`
    namespace, keyed by normalized proposed name (see design spec
    Decision 3 - "cache the fund ISIN -> scheme_code resolution"; this
    node has no ISIN for a not-yet-held candidate, so it caches the
    resolution it actually performs, keyed by the input it actually
    has). `mutual_fund_holdings_analyzer.py` (the node that resolves
    EXISTING holdings by ISIN) is intentionally left untouched per the
    design spec, so this is the one place in this phase that resolution
    caching applies.

    Fails loud: any LLM/tool/verification error propagates uncaught,
    except a candidate whose proposed name has no real Direct-plan
    match, OR whose match's real scheme_category doesn't correspond to
    the proposed category (both ValueError-equivalent reject-early
    cases - the second guards against _resolve_scheme_by_name_and_
    direct_plan silently accepting a fuzzy/wrong search match and
    attaching a DIFFERENT fund's real NAV/metrics to a candidate shown
    under a different name), which are dropped - the same reject-early
    role _verify_ticker plays for the equity specialists. Cost control:
    reject-early (empty gap list short-circuits before any model/toolset/
    store setup; candidates are filtered to real gap categories, then to
    not-previously-rejected names, then capped, before the expensive
    resolve/metrics/evaluate steps run; a previously-rejected scheme_code
    is caught before NAV history is ever fetched).
    """
    gap_categories = _compute_mf_category_gaps(state["holdings_by_bucket"])
    if not gap_categories:
        return {"mutual_fund_candidates": []}

    if store is None:
        raise ValueError(
            "mutual_fund_specialist_node requires a store (compile the graph with "
            "store=... - see graph.py/get_memory_store); without it, rejected "
            "candidates would be silently re-evaluated (and re-paid for) every run."
        )

    model = get_specialist_model()
    mfapi_toolset = get_mfapi_toolset()
    yfinance_toolset = get_yfinance_toolset()
    holdings_summary = _summarize_mf_holdings(state["holdings_by_bucket"])
    gap_categories_text = ", ".join(gap_categories)

    canonical_rejections = await _load_recent_rejections(store, REJECTED_MUTUAL_FUND_NS)
    alias_rejections = await _load_recent_rejections(store, REJECTED_MUTUAL_FUND_BY_NAME_NS)

    async with mfapi_toolset, yfinance_toolset:
        generator = _build_candidate_generator(model)
        prompt = f"Propose candidates for these gap categories: {gap_categories_text}"
        if alias_rejections:
            excluded = sorted({r.display_name for r in alias_rejections.values()})[:MAX_EXCLUSION_HINTS]
            prompt += f"\nDo not propose these already-rejected funds: {', '.join(excluded)}."
        generation_result = await generator.run(prompt)
        on_gap_candidates = _filter_to_gap_sectors(
            generation_result.output, gap_categories, key=lambda c: c.category
        )
        not_previously_rejected = _filter_previously_rejected(
            on_gap_candidates,
            alias_rejections,
            key=lambda c: _normalize_scheme_name(c.scheme_name),
        )
        capped_candidates = _apply_candidate_cap(not_previously_rejected, key=lambda c: c.category)

        evaluator = _build_conviction_gate_evaluator(
            model, yfinance_toolset, holdings_summary, gap_categories_text
        )
        evaluations = []
        for candidate in capped_candidates:
            normalized_name = _normalize_scheme_name(candidate.scheme_name)
            cached_resolution = await _load_resolved_scheme(store, normalized_name)
            if cached_resolution is not None:
                scheme_code = cached_resolution["scheme_code"]
                resolved_category = cached_resolution["category"]
                resolved_scheme_name = cached_resolution["scheme_name"]
            else:
                try:
                    scheme_code, resolved_category, resolved_scheme_name = (
                        await _resolve_scheme_by_name_and_direct_plan(candidate.scheme_name, mfapi_toolset)
                    )
                except ValueError:
                    continue
                await _cache_resolved_scheme(
                    store, normalized_name, scheme_code, resolved_category, resolved_scheme_name
                )

            if not _category_matches_taxonomy(resolved_category, candidate.category):
                # The search match's real category doesn't correspond to what
                # was proposed - this isn't genuinely the same fund the LLM
                # meant, so drop it rather than attach its real data to a
                # candidate shown under the wrong name (Finding 1).
                continue

            if str(scheme_code) in canonical_rejections:
                # The authoritative rejection filter: runs after resolution,
                # before any paid evaluator call or NAV fetch.
                continue

            history = await _fetch_nav_history(scheme_code, mfapi_toolset)
            metrics = _compute_fund_metrics(history)
            benchmark_ticker = MF_CATEGORY_BENCHMARK_TICKERS.get(candidate.category)
            benchmark = await _compute_benchmark_difference(history, benchmark_ticker, yfinance_toolset)
            metrics_text = _format_metrics_for_prompt(metrics, benchmark, {})

            eval_result = await evaluator.run(
                f"Evaluate candidate: {resolved_scheme_name} (category: {candidate.category}).\n{metrics_text}"
            )
            evaluation = CandidateEvaluation.from_verdict(
                eval_result.output,
                identifier=str(scheme_code),
                identifier_kind="scheme_code",
                display_name=resolved_scheme_name,
            )
            evaluations.append(evaluation.model_dump())

            if evaluation.action != "Buy":
                rejected_at = datetime.now(UTC).isoformat()
                await _record_rejection(
                    store,
                    REJECTED_MUTUAL_FUND_NS,
                    RejectedCandidate(
                        identifier=evaluation.identifier,
                        identifier_kind="scheme_code",
                        specialist="mutual_fund",
                        display_name=evaluation.display_name,
                        group=candidate.category,
                        action=evaluation.action,
                        confidence=evaluation.confidence,
                        risk_flags=evaluation.risk_flags,
                        reasoning=evaluation.reasoning,
                        rejected_at=rejected_at,
                        thread_id=None,
                        schema_version=1,
                    ),
                )
                await _record_rejection(
                    store,
                    REJECTED_MUTUAL_FUND_BY_NAME_NS,
                    RejectedCandidate(
                        identifier=normalized_name,
                        identifier_kind="scheme_name",
                        specialist="mutual_fund",
                        display_name=evaluation.display_name,
                        group=candidate.category,
                        action=evaluation.action,
                        confidence=evaluation.confidence,
                        risk_flags=evaluation.risk_flags,
                        reasoning=evaluation.reasoning,
                        rejected_at=rejected_at,
                        thread_id=None,
                        schema_version=1,
                    ),
                )

    return {"mutual_fund_candidates": evaluations}
