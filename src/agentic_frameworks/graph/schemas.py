"""Structured output schemas for LLM-driven nodes.

Every LLM call in this graph must produce a strictly validated
schema - no freeform text leaks through unvalidated, per plan.md.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

US_GAP_SECTORS = [
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
]


class ProposedCandidate(BaseModel):
    """One ticker the candidate generator proposes for a specific gap sector."""

    ticker: str
    sector: Literal[*US_GAP_SECTORS]


class CandidateEvaluation(BaseModel):
    """A new-buy specialist's verdict on one candidate, after the Conviction Gate."""

    action: Literal["Buy", "Hold", "Sell", "Reduce"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    risk_flags: list[str]
    entry_approach: Literal["LumpSum", "Staggered"] | None
    suggested_platform: str | None
