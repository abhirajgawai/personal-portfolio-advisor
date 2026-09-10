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


class GraphState(TypedDict):
    """Combined state for the compiled graph - union of every node's output.

    Includes every field ResearchState and RiskState define individually.
    Those two schemas remain each node's own documented partial
    contribution and what their own tests assert against; this is what
    the compiled StateGraph itself uses as its state_schema, since
    StateGraph takes exactly one schema, not one per node.

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
    """

    holdings: list[dict]
    allocation: dict
    holdings_by_bucket: dict[str, dict[str, list[dict]]]
    concentration: dict
    category_overlap: dict
    currency_exposure: dict
    liquidity: dict
    us_equity_etf_candidates: list[dict]
