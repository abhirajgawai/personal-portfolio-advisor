"""Structured output schemas for LLM-driven nodes.

Every LLM call in this graph must produce a strictly validated
schema - no freeform text leaks through unvalidated, per plan.md.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

GICS_SECTORS = [
    "Health Care",
    "Financials",
    "Energy",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Real Estate",
    "Utilities",
    "Communication Services",
    "Materials",
    "Information Technology",
]

MF_CATEGORY_TAXONOMY = [
    "Large Cap",
    "Mid Cap",
    "Small Cap",
    "Flexi Cap",
    "Multi Cap",
    "ELSS",
    "Aggressive Hybrid",
    "Conservative Hybrid",
    "Liquid",
    "Short Duration",
    "Corporate Bond",
    "Gilt",
    "Index Fund",
]


class ProposedCandidate(BaseModel):
    """One ticker the candidate generator proposes for a specific gap sector.

    sector is validated against the full 11-bucket GICS taxonomy
    (GICS_SECTORS), not a hardcoded subset - which sectors are actual
    gaps for a given specialist/portfolio is computed live at runtime
    (see _compute_us_gaps, _compute_india_equity_gaps), never baked
    into this schema.
    """

    ticker: str
    sector: Literal[*GICS_SECTORS]


class ProposedFundCandidate(BaseModel):
    """One mutual fund the candidate generator proposes for a specific gap category.

    category is validated against the full curated MF_CATEGORY_TAXONOMY,
    not a hardcoded gap-subset - same reasoning as ProposedCandidate.sector:
    which categories are actual gaps is computed live at runtime
    (see _compute_mf_category_gaps), never baked into this schema.
    """

    scheme_name: str
    category: Literal[*MF_CATEGORY_TAXONOMY]


class CandidateEvaluation(BaseModel):
    """A new-buy specialist's verdict on one candidate, after the Conviction Gate."""

    action: Literal["Buy", "Hold", "Sell", "Reduce"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    risk_flags: list[str]
    entry_approach: Literal["LumpSum", "Staggered"] | None
    suggested_platform: str | None
