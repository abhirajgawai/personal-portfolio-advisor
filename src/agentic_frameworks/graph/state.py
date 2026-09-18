"""Shared state schema for the Portfolio Advisor Fleet graph."""

from __future__ import annotations

from typing import TypedDict


class ResearchState(TypedDict):
    """State the Research node produces: raw holdings plus a grouped view.

    holdings: the raw `latest_snapshot` rows (platform, geography, ticker,
        name, currency, sector, asset_class, quantity, price,
        market_value_base, updated_at).
    allocation: the raw `allocation_summary` output (totalBase,
        holdingsCount, byGeography, byPlatform, bySector, byAssetClass).
    holdings_by_bucket: `holdings` regrouped as
        {asset_class: {geography: [holding, ...]}}.
    """

    holdings: list[dict]
    allocation: dict
    holdings_by_bucket: dict[str, dict[str, list[dict]]]


class RiskState(TypedDict):
    """State the Risk Analyst node produces: raw risk measurements, no verdicts.

    concentration: {"by_holding": [{"ticker", "name", "asset_class",
        "pct_of_portfolio", "pct_of_asset_class"}], "by_asset_class":
        [{"asset_class", "pct_of_portfolio"}]}.
    category_overlap: {"by_sector": {"<sector>": [{"ticker", "name",
        "asset_class"}]}} — only sectors with more than one holding.
    currency_exposure: {"<currency>": {"value_base", "pct_of_portfolio"}}.
    liquidity: {"illiquid": {"value_base", "pct_of_portfolio"},
        "liquid": {"value_base", "pct_of_portfolio"}}.
    """

    concentration: dict
    category_overlap: dict
    currency_exposure: dict
    liquidity: dict


class SpecialistState(TypedDict):
    """State the new-buy specialist and holdings-analyzer nodes produce.

    Every list below holds `CandidateEvaluation` dicts, which always
    carry `identifier` + `identifier_kind` + `display_name` (the
    verified identity of the instrument judged, set in node code from
    real data, never by the LLM) alongside the verdict fields. Consumers
    must key off `identifier`, never off list position.

    us_equity_etf_candidates: the US Equity/ETF Specialist's evaluated
        candidates, each a `CandidateEvaluation` dict.
    india_equity_candidates: the India Equity Specialist's evaluated
        candidates, each a `CandidateEvaluation` dict.
    existing_fund_evaluations: the Mutual Fund Holdings Analyzer's
        verdicts on the real 9 existing IN mutual fund holdings, each
        a `CandidateEvaluation` dict.
    mutual_fund_candidates: the Mutual Fund Specialist's evaluated
        new-buy candidates, each a `CandidateEvaluation` dict.
    """

    us_equity_etf_candidates: list[dict]
    india_equity_candidates: list[dict]
    existing_fund_evaluations: list[dict]
    mutual_fund_candidates: list[dict]


class SynthesizerState(TypedDict):
    """State the Synthesizer node produces.

    recommendation: an `AdvisorReport` dict (schemas.AdvisorReport.
        model_dump()) - identity already re-bound in code, never a raw
        SynthesizedRecommendation (the LLM's identity-poor output_type).
    """

    recommendation: dict


class ApprovalState(TypedDict):
    """State the human-approval-gate node produces.

    approval: a `HumanApproval` dict (schemas.HumanApproval.model_dump()) -
        the validated resume payload a human supplied via LangGraph's
        resume-command primitive, invoked from outside the graph.
    """

    approval: dict


class GraphState(TypedDict):
    """Combined state for the compiled graph - union of every node's output.

    Includes every field ResearchState and RiskState define individually.
    Those two schemas remain each node's own documented partial
    contribution and what their own tests assert against; this is what
    the compiled StateGraph itself uses as its state_schema, since
    StateGraph takes exactly one schema, not one per node.

    Every `CandidateEvaluation` dict in the four candidate/evaluation
    lists below carries `identifier` + `identifier_kind` +
    `display_name` (see schemas.CandidateEvaluation) - the verified
    identity of what was judged. Consumers key off `identifier`, never
    off list position.

    Fields:
    holdings: the raw `latest_snapshot` rows from Research node.
    allocation: the raw `allocation_summary` output from Research node.
    holdings_by_bucket: `holdings` regrouped as
        {asset_class: {geography: [holding, ...]}}.
    concentration: concentration risk measurements from Risk Analyst node.
    category_overlap: category overlap risk measurements from Risk Analyst node.
    currency_exposure: currency exposure risk measurements from Risk Analyst node.
    liquidity: liquidity risk measurements from Risk Analyst node.
    us_equity_etf_candidates: the US Equity/ETF Specialist's evaluated candidates,
        each a `CandidateEvaluation` dict.
    india_equity_candidates: the India Equity Specialist's evaluated candidates,
        each a `CandidateEvaluation` dict.
    existing_fund_evaluations: the Mutual Fund Holdings Analyzer's verdicts on
        the real 9 existing IN mutual fund holdings, each a `CandidateEvaluation` dict.
    mutual_fund_candidates: the Mutual Fund Specialist's evaluated new-buy candidates,
        each a `CandidateEvaluation` dict.
    recommendation: the Synthesizer's `AdvisorReport` dict (see
        SynthesizerState) - the graph's one, reconciled, two-sided report.
    approval: the human-approval-gate's `HumanApproval` dict (see
        ApprovalState) - only present on runs that actually required
        approval (a Hold-only, no-new-buy run never reaches that node).
    """

    holdings: list[dict]
    allocation: dict
    holdings_by_bucket: dict[str, dict[str, list[dict]]]
    concentration: dict
    category_overlap: dict
    currency_exposure: dict
    liquidity: dict
    us_equity_etf_candidates: list[dict]
    india_equity_candidates: list[dict]
    existing_fund_evaluations: list[dict]
    mutual_fund_candidates: list[dict]
    recommendation: dict
    approval: dict
