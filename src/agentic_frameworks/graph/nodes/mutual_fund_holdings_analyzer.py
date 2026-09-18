"""The Mutual Fund Holdings Analyzer node: judges the real 9 existing IN
mutual fund holdings against a 7-check Conviction Gate, redefined for
funds (see docs/superpowers/specs/2026-09-12-mutual-fund-specialist-
design.md). Measures first (pure functions, no LLM), judges second
(one LLM evaluator call per fund) - mirrors the Risk Analyst's
measure-not-judge separation. Fails loud, same as every other node.
"""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.toolsets.filtered import FilteredToolset

from agentic_frameworks.graph.mcp_clients import get_mfapi_toolset, get_yfinance_toolset
from agentic_frameworks.graph.models import get_specialist_model
from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
    INDEX_FUND_BENCHMARK_TICKERS_BY_ISIN,
    MF_CATEGORY_BENCHMARK_TICKERS,
    MF_CATEGORY_TO_TAXONOMY,
    _compute_benchmark_difference,
    _compute_fund_metrics,
    _compute_redundancy_correlations,
    _fetch_nav_history,
    _resolve_scheme_code,
)
from agentic_frameworks.graph.schemas import CandidateEvaluation, ConvictionVerdict
from agentic_frameworks.graph.state import GraphState, SpecialistState


def _lookup_benchmark_ticker(isin: str, category: str) -> str | None:
    """Index funds resolve via their own ISIN (exact tracked index varies
    per fund, not per category); every other category resolves via the
    shared category-benchmark table."""
    if category == "Index Fund":
        return INDEX_FUND_BENCHMARK_TICKERS_BY_ISIN.get(isin)
    taxonomy_category = MF_CATEGORY_TO_TAXONOMY[category]
    return MF_CATEGORY_BENCHMARK_TICKERS.get(taxonomy_category)


def _format_metrics_for_prompt(
    metrics: dict,
    benchmark: dict | None,
    redundancy: dict[str, float],
    isin_to_name: dict[str, str] | None = None,
) -> str:
    """isin_to_name: real ISIN->fund-name mapping for the redundancy section
    (built from real holdings by the caller). Defaults to {} so an LLM never
    sees a bare, uninterpretable ISIN where a name is unavailable (e.g. the
    Mutual Fund Specialist's new-buy path, which has no redundancy data at
    all and always passes an empty redundancy dict, making this mapping a
    no-op there)."""
    isin_to_name = isin_to_name or {}
    lines = [
        f"Track record: {metrics['track_record_years']} years",
        f"CAGR since inception: {metrics['cagr_since_inception']:.1%}" if metrics["cagr_since_inception"] is not None else "CAGR since inception: undefined (insufficient history)",
        f"CAGR 1yr: {metrics['cagr_1y']:.1%}" if metrics["cagr_1y"] is not None else "CAGR 1yr: insufficient history",
        f"CAGR 3yr: {metrics['cagr_3y']:.1%}" if metrics["cagr_3y"] is not None else "CAGR 3yr: insufficient history",
        f"Annualized volatility: {metrics['volatility']:.1%}" if metrics["volatility"] is not None else "Annualized volatility: undefined (insufficient history)",
        f"Max drawdown: {metrics['max_drawdown']:.1%}" if metrics["max_drawdown"] is not None else "Max drawdown: undefined (insufficient history)",
    ]
    if benchmark is not None:
        lines.append(
            f"Benchmark comparison: fund CAGR {benchmark['fund_cagr']:.1%} vs "
            f"benchmark CAGR {benchmark['benchmark_cagr']:.1%} "
            f"(difference: {benchmark['difference']:+.1%})"
        )
    else:
        lines.append("Benchmark comparison: unavailable for this fund's category")
    if redundancy:
        top = sorted(redundancy.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        lines.append(
            "Highest correlation with other held funds: "
            + ", ".join(f"{isin_to_name.get(isin, isin)}={corr:.2f}" for isin, corr in top)
        )
    else:
        lines.append("Redundancy: no comparable overlap data")
    return "\n".join(lines)


_CONVICTION_GATE_SYSTEM_PROMPT = """You are evaluating one existing India mutual fund holding for a \
personal, long-term portfolio advisor, using a 7-check Conviction Gate \
adapted for mutual funds (not equities - no price/valuation checks apply). \
Run these checks in order, weighing all of them in your final verdict:

1. Fit/redundancy: does this fund still serve a distinct purpose, or is it \
largely redundant with another holding (see the redundancy correlation data)?
2. Cost: funds held via a Direct plan avoid distributor commission drag - \
treat this as a qualitative positive, not a numeric expense-ratio check \
(exact expense ratios are not available from this project's data sources).
3. Performance vs benchmark: did it beat its benchmark over its own history, \
where a verified benchmark exists? If no benchmark is available for this \
fund's category, say so explicitly and weigh the other checks more heavily \
rather than guessing at benchmark performance.
4. Risk-adjusted performance: consider volatility and max drawdown, not just \
raw returns - a fund beating its benchmark by taking on much more risk isn't \
straightforwardly better for a long-term investor.
5. Cross-fund redundancy: real correlation data with the user's other funds \
is provided - flag genuine overlap.
6. Track-record length: a short track record (under ~3 years) means \
performance-consistency claims should carry less weight in your reasoning.
7. Style-fit: is this fund's mandate consistent with a long-term, \
buy-and-hold approach?

Tax context (India, current rules, for informing hold/reduce/sell timing \
advice only - never invent numbers beyond this): equity-oriented mutual \
funds (>=65% equity) are taxed as equity capital gains; other funds \
(debt-oriented, most hybrid funds) are taxed at the investor's income slab \
rate regardless of holding period; ELSS funds carry a mandatory 3-year \
lock-in from each investment date.

Your output is advisory only - never a trade instruction, only a \
recommendation for the user to act on manually."""


def _build_conviction_gate_evaluator(model, toolset: FilteredToolset) -> Agent:
    """retries=2: bounded per CLAUDE.md's LLM cost control rule 6.
    output_type=ConvictionVerdict: judgment only - the node binds each
    verdict to the holding's real ISIN and name from portfolio.db, so
    the model is never in a position to mislabel which fund it judged.
    """
    return Agent(
        model,
        output_type=ConvictionVerdict,
        system_prompt=_CONVICTION_GATE_SYSTEM_PROMPT,
        toolsets=[toolset],
        retries=2,
    )


async def mutual_fund_holdings_analyzer_node(state: GraphState) -> SpecialistState:
    """Judge the real 9 existing IN mutual fund holdings.

    Fails loud: any scheme-resolution/NAV-fetch/LLM error propagates
    uncaught. Cost control: metrics computed once per fund (pure, no
    LLM), reused across the evaluator call; toolsets shared across all
    9 evaluations within one run.
    """
    model = get_specialist_model()
    mfapi_toolset = get_mfapi_toolset()
    yfinance_toolset = get_yfinance_toolset()
    holdings = state["holdings_by_bucket"].get("mutual_fund", {}).get("IN", [])

    async with mfapi_toolset, yfinance_toolset:
        fund_histories: dict[str, list[dict]] = {}
        for holding in holdings:
            scheme_code = await _resolve_scheme_code(holding["name"], holding["ticker"], mfapi_toolset)
            fund_histories[holding["ticker"]] = await _fetch_nav_history(scheme_code, mfapi_toolset)

        redundancy = _compute_redundancy_correlations(fund_histories)
        isin_to_name = {holding["ticker"]: holding["name"] for holding in holdings}

        evaluator = _build_conviction_gate_evaluator(model, yfinance_toolset)
        evaluations = []
        for holding in holdings:
            isin = holding["ticker"]
            category = holding["sector"]
            history = fund_histories[isin]
            metrics = _compute_fund_metrics(history)
            benchmark_ticker = _lookup_benchmark_ticker(isin, category)
            benchmark = await _compute_benchmark_difference(history, benchmark_ticker, yfinance_toolset)
            metrics_text = _format_metrics_for_prompt(
                metrics, benchmark, redundancy.get(isin, {}), isin_to_name
            )

            eval_result = await evaluator.run(
                f"Evaluate existing holding: {holding['name']} (category: {category}).\n{metrics_text}"
            )
            evaluations.append(
                CandidateEvaluation.from_verdict(
                    eval_result.output,
                    identifier=isin,
                    identifier_kind="isin",
                    display_name=holding["name"],
                ).model_dump()
            )

    return {"existing_fund_evaluations": evaluations}
