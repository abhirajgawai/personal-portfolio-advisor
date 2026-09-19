"""The Recommendation Synthesizer node: reconciles all four parallel
branch outputs (Research/Risk Analyst context, the three new-buy
specialists' candidate lists, and the Mutual Fund Holdings Analyzer's
verdicts) into one strictly-validated, genuinely two-sided report.

No toolsets, one LLM call per run - see docs/superpowers/specs/
2026-09-19-synthesizer-design.md, Architecture point 2. Judges, does not
gather: everything it needs was already measured or judged by an
upstream node with real data. Fails loud, same as every other node in
this graph.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

from langgraph.store.base import BaseStore
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.settings import ModelSettings

from agentic_frameworks.graph.models import get_synthesizer_model
from agentic_frameworks.graph.nodes._candidate_memory import (
    REJECTED_INDIA_EQUITY_NS,
    REJECTED_MUTUAL_FUND_NS,
    REJECTED_US_EQUITY_NS,
    _load_recent_rejections,
)
from agentic_frameworks.graph.schemas import (
    MAX_NEW_CAPITAL_PCT,
    AdvisorReport,
    ReportCoverage,
    SynthesizedRecommendation,
)
from agentic_frameworks.graph.state import GraphState, SynthesizerState

_TOP_N_CONCENTRATION = 8

_SPECIALIST_BY_STATE_KEY = {
    "us_equity_etf_candidates": "us_equity",
    "india_equity_candidates": "india_equity",
    "mutual_fund_candidates": "mutual_fund",
}

_IDENTIFIER_KIND_BY_SPECIALIST = {
    "us_equity": "ticker",
    "india_equity": "ticker",
    "mutual_fund": "scheme_code",
}

_REJECTION_NS_BY_SPECIALIST = {
    "us_equity": REJECTED_US_EQUITY_NS,
    "india_equity": REJECTED_INDIA_EQUITY_NS,
    "mutual_fund": REJECTED_MUTUAL_FUND_NS,
}


class _SubjectEntry(NamedTuple):
    """One entry in the subject index - the verified identity of a thing
    the Synthesizer is allowed to talk about, keyed by `identifier`.

    side: "existing" (from existing_fund_evaluations) or "new_buy" (a
        Buy-verdict candidate from one of the three specialists). The
        output_validator checks this so an existing-holdings identifier
        can never be smuggled into new_buy_suggestions or vice versa.
    specialist: which specialist produced this candidate - None for the
        existing side (existing_fund_evaluations has no such field).
    """

    identifier_kind: str
    display_name: str
    side: str
    specialist: str | None


def _select_buy_candidates(state: GraphState) -> list[dict]:
    """Code filters to action == "Buy" before the LLM ever sees a candidate
    (design spec Architecture point 4, layer 1) - "avoid is the default"
    means non-Buy candidates are never presented as buyable options.
    Each surviving candidate is tagged with which specialist produced it.
    """
    buy_candidates: list[dict] = []
    for state_key, specialist in _SPECIALIST_BY_STATE_KEY.items():
        for candidate in state[state_key]:
            if candidate["action"] == "Buy":
                buy_candidates.append({**candidate, "specialist": specialist})
    return buy_candidates


def _build_subject_index(state: GraphState, buy_candidates: list[dict]) -> dict[str, _SubjectEntry]:
    """Build the subject index the output_validator checks every LLM-named
    identifier against - in code, from real upstream data, never trusting
    the model's own identifier (design spec Architecture point 3).
    """
    index: dict[str, _SubjectEntry] = {}
    for evaluation in state["existing_fund_evaluations"]:
        index[evaluation["identifier"]] = _SubjectEntry(
            identifier_kind=evaluation["identifier_kind"],
            display_name=evaluation["display_name"],
            side="existing",
            specialist=None,
        )
    for candidate in buy_candidates:
        index[candidate["identifier"]] = _SubjectEntry(
            identifier_kind=candidate["identifier_kind"],
            display_name=candidate["display_name"],
            side="new_buy",
            specialist=candidate["specialist"],
        )
    return index


def _derive_platforms(holdings: list[dict]) -> set[str]:
    """Live-derive the real platform set from this run's actual holdings -
    never hardcoded (design spec Background: zerodha/moomoo/groww/
    sharekhan/real_estate observed live, but this must not assume that
    exact set stays fixed)."""
    return {holding["platform"] for holding in holdings}


def _compute_coverage(holdings: list[dict], subject_index: dict[str, _SubjectEntry]) -> ReportCoverage:
    """Built entirely in code - the honest, real count of how many
    holdings this report judged individually versus only at the
    allocation level (design spec Out of Scope / Risks). Real estate is
    excluded from the denominator's "unassessed" framing: it's a fixed
    input, never a hold/sell/buy subject (never counted as a coverage
    gap, only real hold/sell/buy-eligible asset classes are)."""
    holdings_total = len(holdings)
    assessed_identifiers = {
        identifier for identifier, entry in subject_index.items() if entry.side == "existing"
    }
    assessed_count = sum(1 for h in holdings if h["ticker"] in assessed_identifiers)

    unassessed_buckets = sorted(
        {
            f"{h['asset_class']}/{h['geography']}"
            for h in holdings
            if h["ticker"] not in assessed_identifiers and h["asset_class"] != "real_estate"
        }
    )

    coverage_note = (
        f"{assessed_count} of {holdings_total} holdings have an individual Conviction Gate "
        "verdict (the India mutual funds, judged by the Mutual Fund Holdings Analyzer). "
        "Direct equities and ETFs are covered only at the allocation level in this report - "
        "no Equity Holdings Analyzer exists yet (see design spec's Out of Scope section; "
        "a natural Phase 9 candidate). Real estate is a fixed input, never a hold/sell/buy "
        "subject, so it is excluded from the unassessed-bucket count."
    )
    return ReportCoverage(
        holdings_total=holdings_total,
        holdings_individually_assessed=assessed_count,
        unassessed_buckets=unassessed_buckets,
        coverage_note=coverage_note,
    )


def _summarize_allocation(allocation: dict) -> str:
    """Minimal, derived summary - never the raw 45-row holdings dump
    (CLAUDE.md LLM cost control rule 4)."""
    lines = [f"Total portfolio value (base currency): {allocation['totalBase']:,.0f}"]
    for bucket_name, bucket_key in (("Geography", "byGeography"), ("Asset class", "byAssetClass")):
        parts = [
            f"{label}: {values['pct']:.1f}%" for label, values in allocation.get(bucket_key, {}).items()
        ]
        lines.append(f"{bucket_name}: " + ", ".join(parts))
    return "\n".join(lines)


def _summarize_risk(
    concentration: dict, category_overlap: dict, currency_exposure: dict, liquidity: dict
) -> str:
    """Top-N concentration entries only, not the full holding-by-holding
    list (cost rule 4)."""
    top_holdings = sorted(
        concentration["by_holding"], key=lambda h: h["pct_of_portfolio"], reverse=True
    )[:_TOP_N_CONCENTRATION]
    concentration_lines = [
        f"- {h['name']} ({h['ticker']}): {h['pct_of_portfolio']:.1f}% of portfolio"
        for h in top_holdings
    ]
    overlap_lines = [
        f"- {sector}: {', '.join(e['name'] for e in entries)}"
        for sector, entries in category_overlap.get("by_sector", {}).items()
    ]
    currency_lines = [
        f"- {currency}: {values['pct_of_portfolio']:.1f}% of portfolio"
        for currency, values in currency_exposure.items()
    ]
    liquidity_line = (
        f"Illiquid (real estate): {liquidity['illiquid']['pct_of_portfolio']:.1f}%, "
        f"liquid: {liquidity['liquid']['pct_of_portfolio']:.1f}%"
    )
    return (
        f"Top concentrated holdings:\n{chr(10).join(concentration_lines) or '(none)'}\n\n"
        f"Sector/category overlap (more than one holding in the same bucket):\n"
        f"{chr(10).join(overlap_lines) or '(no overlap)'}\n\n"
        f"Currency exposure:\n{chr(10).join(currency_lines) or '(none)'}\n\n"
        f"Liquidity: {liquidity_line}"
    )


def _format_existing_fund_verdicts(existing_fund_evaluations: list[dict]) -> str:
    """The 9 fund verdicts verbatim (action, reasoning, risk_flags,
    confidence) - this is real, code-verified identity and real judgment,
    so passing it through is not a cost-control violation the way the raw
    holdings dump would be (design spec Data Flow step 3)."""
    if not existing_fund_evaluations:
        return "(no existing mutual fund verdicts this run)"
    lines = []
    for evaluation in existing_fund_evaluations:
        lines.append(
            f"- {evaluation['display_name']} [{evaluation['identifier']}]: "
            f"{evaluation['action']} (confidence {evaluation['confidence']:.2f}) - "
            f"{evaluation['reasoning']} Risk flags: {', '.join(evaluation['risk_flags']) or 'none'}."
        )
    return "\n".join(lines)


def _format_buy_candidates(buy_candidates: list[dict]) -> str:
    """The Buy candidates verbatim - the only candidates the Synthesizer
    may ever surface as a new-buy suggestion (design spec Architecture
    point 4, layer 2)."""
    if not buy_candidates:
        return "(no Buy-verdict candidates this run)"
    lines = []
    for candidate in buy_candidates:
        lines.append(
            f"- {candidate['display_name']} [{candidate['identifier']}] "
            f"(via {candidate['specialist']} specialist): Buy "
            f"(confidence {candidate['confidence']:.2f}) - {candidate['reasoning']} "
            f"Risk flags: {', '.join(candidate['risk_flags']) or 'none'}. "
            f"Suggested entry: {candidate['entry_approach']}, platform: {candidate['suggested_platform']}."
        )
    return "\n".join(lines)


def _summarize_considered_and_rejected(state: GraphState) -> str:
    """Non-Buy candidates as one line each - display_name + action + first
    risk flag only, never the full reasoning (cost rule 4; also design
    spec Architecture point 4, layer 1: these must never look like
    buyable options)."""
    lines = []
    for state_key, specialist in _SPECIALIST_BY_STATE_KEY.items():
        for candidate in state[state_key]:
            if candidate["action"] != "Buy":
                first_flag = candidate["risk_flags"][0] if candidate["risk_flags"] else "no specific flag"
                lines.append(
                    f"- {candidate['display_name']} ({specialist}): {candidate['action']} ({first_flag})"
                )
    if not lines:
        return "(no non-Buy candidates were evaluated this run)"
    return "\n".join(lines)


async def _summarize_store_rejections(state: GraphState, store: BaseStore) -> str:
    """Read-only rejection-memory context (Decision 4): makes the report
    honest about candidates suppressed by cross-run memory before this run
    even proposed them, rather than silently omitted. Costs a handful of
    short strings per specialist namespace - no writes here."""
    lines = []
    for specialist, namespace in _REJECTION_NS_BY_SPECIALIST.items():
        rejections = await _load_recent_rejections(store, namespace)
        if rejections:
            names = sorted({r.display_name for r in rejections.values()})
            lines.append(f"- {specialist}: {', '.join(names)}")
    if not lines:
        return "(no candidates suppressed by rejection memory this run)"
    return "\n".join(lines)


def _deterministic_no_new_buy_reason(state: GraphState, rejection_context: str) -> str:
    """Computed in code, not paid for from the model - a deterministic
    string can't drift (design spec Architecture point 4, layer 4). Real
    counts of how many candidates each specialist proposed/evaluated and
    how many were suppressed from memory before this run even started."""
    per_specialist_counts = []
    for state_key, specialist in _SPECIALIST_BY_STATE_KEY.items():
        evaluated = len(state[state_key])
        non_buy = sum(1 for c in state[state_key] if c["action"] != "Buy")
        per_specialist_counts.append(f"{specialist}: {evaluated} evaluated, {non_buy} not Buy")
    return (
        "No new-buy candidate cleared the Conviction Gate this cycle. "
        f"Per specialist - {'; '.join(per_specialist_counts)}. "
        f"Suppressed by prior-run rejection memory before evaluation: {rejection_context}"
    )


_SYNTHESIZER_SYSTEM_PROMPT = """You are the Recommendation Synthesizer for a personal, long-term \
portfolio advisor. You reconcile measurements and verdicts that other, \
specialized nodes already produced from real data - you do not re-derive \
fundamentals, valuation, NAV, or benchmark judgments yourself, and you have \
no tools. Your job is reconciliation, prioritization, and honest \
allocation-level narrative.

Produce a two-sided report:
1. Guidance on existing holdings you are given verdicts for (Hold, AddTo, \
Reduce, or Sell), grounded in the verdict and risk data you are given - \
never invent a verdict for a holding you were not given data on.
2. New-buy suggestions, chosen ONLY from the Buy-verdict candidates you are \
given - you may select a subset of them (including none), but you must \
never invent, rename, or propose a candidate that isn't in that list. \
Every identifier you use (for both existing-holdings guidance and new-buy \
suggestions) must be one you were explicitly given - never guess, \
abbreviate, or reformat an identifier.

Avoid is the default, not buy: if the Buy-candidate list is empty, or none \
of them is compelling next to the portfolio's real allocation and risk \
picture, it is correct and expected to suggest no new buys. Every \
suggestion you do make must include the real platform (from the list you \
are given) it should be entered on, a LumpSum or Staggered entry approach, \
and a position size, as a percentage of the total portfolio, that stays \
within the stated per-suggestion cap.

This is advisory only - your output is never a trade instruction, only a \
recommendation for the user to act on manually. Long-term investing only - \
never day-trading, options, or market-timing language.

Be concise. For each holding or suggestion give at most 4 risk_flags, each \
one short sentence. Keep every rationale to 2-3 sentences and the overall \
portfolio_assessment to one tight paragraph. This is a hard requirement, \
not a style preference - a response that runs long risks being cut off \
before it can be returned at all."""


def _build_synthesizer_agent(
    model, subject_index: dict[str, _SubjectEntry], platforms: set[str]
) -> Agent:
    """retries=2: bounded per CLAUDE.md's LLM cost control rule 6.
    output_type=SynthesizedRecommendation: the LLM's identity-poor output
    type (see schemas.py module docstring). The output_validator below is
    the structural guardrail design spec Architecture point 3 describes -
    every identifier is checked against subject_index (built in code from
    verified upstream data), with side enforced too (an existing-holdings
    identifier can never appear as a new-buy suggestion or vice versa), the
    suggested_platform is checked against this run's real, live-derived
    platform set, and the run-aggregate position-sizing cap
    (MAX_NEW_CAPITAL_PCT) is enforced here since it's a property of the
    whole list, not one field (the per-suggestion cap is already enforced
    by the schema itself). Any violation raises ModelRetry, bounded by
    retries=2, then propagates - never a silent accept or a code-side
    "fix" of the model's output.
    """
    agent = Agent(
        model,
        output_type=SynthesizedRecommendation,
        system_prompt=_SYNTHESIZER_SYSTEM_PROMPT,
        retries=2,
        # A real run (2026-09-19) exceeded output retries because the raw
        # response was truncated mid-JSON before validation ever ran - a
        # completion-length problem, not a schema-shape one. The prompt's
        # brevity instruction and the schema's new per-field length caps
        # (schemas.py: RISK_FLAG_MAX_CHARS etc.) are the primary fix; this
        # is the secondary safety net, giving the model enough budget to
        # actually finish a full report even before it fully internalizes
        # "be concise" (retries=2 only helps once a response is complete
        # enough to parse and fail *validation* - it does nothing for a
        # response cut off mid-stream).
        model_settings=ModelSettings(max_tokens=24000),
    )

    @agent.output_validator
    def _validate_synthesis(data: SynthesizedRecommendation) -> SynthesizedRecommendation:
        for guidance in data.existing_holdings_guidance:
            entry = subject_index.get(guidance.identifier)
            if entry is None or entry.side != "existing":
                raise ModelRetry(
                    f"'{guidance.identifier}' is not a known existing-holding identifier this "
                    "run evaluated - use only identifiers you were explicitly given."
                )
        for suggestion in data.new_buy_suggestions:
            entry = subject_index.get(suggestion.identifier)
            if entry is None or entry.side != "new_buy":
                raise ModelRetry(
                    f"'{suggestion.identifier}' is not one of this run's Buy-verdict candidates - "
                    "use only identifiers from the new-buy candidate list you were given."
                )
            if suggestion.suggested_platform not in platforms:
                raise ModelRetry(
                    f"'{suggestion.suggested_platform}' is not a real platform this portfolio "
                    f"uses ({', '.join(sorted(platforms))})."
                )
        total_new_capital_pct = sum(s.position_size_pct_of_portfolio for s in data.new_buy_suggestions)
        if total_new_capital_pct > MAX_NEW_CAPITAL_PCT:
            raise ModelRetry(
                f"New-buy suggestions sum to {total_new_capital_pct:.1f}% of the portfolio, over "
                f"the {MAX_NEW_CAPITAL_PCT}% aggregate cap - shrink the set or reduce sizes."
            )
        return data

    return agent


def _build_prompt(
    *,
    allocation_summary: str,
    risk_summary: str,
    existing_verdicts_text: str,
    buy_candidates_text: str | None,
    considered_rejected_text: str,
    platforms: set[str],
    coverage: ReportCoverage,
) -> str:
    """All variable portfolio data goes in the user message, never the
    system prompt (design spec Data Flow step 5 / cost control rule 3 -
    keeps the system prompt byte-identical and cacheable across runs).
    buy_candidates_text=None omits the new-buy block entirely when the Buy
    set is empty (design spec Architecture point 4, layer 4)."""
    sections = [
        f"Portfolio allocation summary:\n{allocation_summary}",
        f"Risk measurements:\n{risk_summary}",
        f"Existing mutual fund holdings - real verdicts:\n{existing_verdicts_text}",
        f"Coverage note (for your portfolio_assessment context): {coverage.coverage_note}",
    ]
    if buy_candidates_text is not None:
        sections.append(
            f"New-buy candidates that cleared the Conviction Gate (Buy verdict only) - "
            f"you may select any subset of these, or none:\n{buy_candidates_text}\n\n"
            f"Real platforms available this run: {', '.join(sorted(platforms))}."
        )
    sections.append(f"Already considered and rejected this cycle (not eligible as new buys):\n{considered_rejected_text}")
    return "\n\n".join(sections)


async def synthesizer_node(
    state: GraphState, *, store: Optional[BaseStore] = None
) -> SynthesizerState:
    """Reconcile all four branch outputs into one validated AdvisorReport.

    `store` is injected by LangGraph via signature; a missing store
    raises (same guard every store-using node in this graph uses) since
    without it the Synthesizer could not honestly report what was
    suppressed by rejection memory (Decision 4).

    Fails loud: any LLM/validation error propagates uncaught. A branch
    that never ran surfaces as a real KeyError on the missing state key -
    a report synthesized from three of four branches would be silently
    wrong, which is worse than a crash (design spec Error Handling).
    """
    if store is None:
        raise ValueError(
            "synthesizer_node requires a store (compile the graph with store=... - see "
            "graph.py/get_memory_store); without it, the report could not honestly say what "
            "was suppressed by rejection memory this run (design spec Decision 4)."
        )

    buy_candidates = _select_buy_candidates(state)
    subject_index = _build_subject_index(state, buy_candidates)
    coverage = _compute_coverage(state["holdings"], subject_index)
    platforms = _derive_platforms(state["holdings"])

    rejection_context = await _summarize_store_rejections(state, store)
    allocation_summary = _summarize_allocation(state["allocation"])
    risk_summary = _summarize_risk(
        state["concentration"], state["category_overlap"], state["currency_exposure"], state["liquidity"]
    )
    existing_verdicts_text = _format_existing_fund_verdicts(state["existing_fund_evaluations"])
    considered_rejected_text = _summarize_considered_and_rejected(state)

    if buy_candidates:
        buy_candidates_text = _format_buy_candidates(buy_candidates)
        deterministic_no_new_buy_reason = None
    else:
        # Reject-early: omit the new-buy block from the prompt entirely
        # rather than paying a model to explain an empty set.
        buy_candidates_text = None
        deterministic_no_new_buy_reason = _deterministic_no_new_buy_reason(state, rejection_context)

    prompt = _build_prompt(
        allocation_summary=allocation_summary,
        risk_summary=risk_summary,
        existing_verdicts_text=existing_verdicts_text,
        buy_candidates_text=buy_candidates_text,
        considered_rejected_text=considered_rejected_text,
        platforms=platforms,
        coverage=coverage,
    )

    model = get_synthesizer_model()
    agent = _build_synthesizer_agent(model, subject_index, platforms)
    result = await agent.run(prompt)
    synthesis = result.output

    if deterministic_no_new_buy_reason is not None:
        # The Buy set was empty going in, so no_new_buy_reason is always
        # code-computed and deterministic, never LLM-authored prose - a
        # deterministic string can't drift (design spec Architecture point
        # 4, layer 4). new_buy_suggestions is forced empty too, defensively
        # matching the reject-early decision already made above.
        synthesis = synthesis.model_copy(
            update={"new_buy_suggestions": [], "no_new_buy_reason": deterministic_no_new_buy_reason}
        )

    report = AdvisorReport.from_synthesis(synthesis, subject_index=subject_index, coverage=coverage)
    return {"recommendation": report.model_dump()}
