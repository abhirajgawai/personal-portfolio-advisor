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
