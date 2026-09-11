"""The India Equity Specialist node: proposes, verifies, and evaluates
India equity new-buy candidates against the New-Buy Conviction Gate.

Same three-stage pipeline as the US Equity/ETF Specialist
(nodes/us_equity_etf_specialist.py): live gap detection -> candidate
generation -> ticker verification -> Conviction Gate evaluation.
Fails loud, same as every other node in this graph.
"""

from __future__ import annotations

from agentic_frameworks.graph.schemas import GICS_SECTORS

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
