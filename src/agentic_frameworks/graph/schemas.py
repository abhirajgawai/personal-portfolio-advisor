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


class ConvictionVerdict(BaseModel):
    """The judgment half of an evaluation: what an LLM evaluator returns.

    This is the `output_type` every Conviction Gate evaluator Agent is
    built with - deliberately carrying NO identity fields. The subject
    of a verdict is established in code from already-verified data
    (a yfinance-confirmed ticker, a portfolio.db ISIN, an MFapi-resolved
    scheme code), never re-stated by the model, so there is no path by
    which a model can mislabel which instrument its verdict belongs to.
    A structural guardrail, not an instructional one (CLAUDE.md
    agentic design principle 7): the capability to name the subject is
    never given to the LLM in the first place.
    """

    action: Literal["Buy", "Hold", "Sell", "Reduce"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    risk_flags: list[str]
    entry_approach: Literal["LumpSum", "Staggered"] | None
    suggested_platform: str | None


class CandidateEvaluation(ConvictionVerdict):
    """A specialist/analyzer node's verdict on one candidate or holding,
    bound to the verified identity of the thing it judges.

    This is what lands in graph state (and what the Synthesizer will
    consume) - never a bare `ConvictionVerdict`. The three identity
    fields are always set in node code from verified data:

    identifier: the machine identifier of the subject. One field rather
        than three mutually-exclusive optional ones (ticker?/isin?/
        scheme_code?) so "exactly one identifier, always present" is a
        type-level guarantee rather than a convention every consumer has
        to re-implement as `ticker or isin or scheme_code`.
    identifier_kind: which namespace `identifier` belongs to -
        "ticker" (US/India equity and ETF candidates, yfinance-verified,
        `.NS`-suffixed for NSE), "isin" (existing IN mutual fund
        holdings, the real ISIN stored in portfolio.db's ticker column),
        or "scheme_code" (new-buy mutual fund candidates, the MFapi.in
        scheme code resolved and category-checked before evaluation).
        Kept explicit rather than inferred from the string's shape,
        because inference would be a guess and this project does not
        guess at identity.
    display_name: the human-readable name of the same verified subject
        (yfinance `longName`, portfolio.db holding name, or the resolved
        MFapi scheme name) - real data from the same verification step
        that produced `identifier`, never the LLM's proposed name. A
        scheme_code like "118989" is unusable in a recommendation shown
        to a human, so identity needs both halves.
    """

    identifier: str
    identifier_kind: Literal["ticker", "isin", "scheme_code"]
    display_name: str

    @classmethod
    def from_verdict(
        cls,
        verdict: ConvictionVerdict,
        *,
        identifier: str,
        identifier_kind: Literal["ticker", "isin", "scheme_code"],
        display_name: str,
    ) -> CandidateEvaluation:
        """Bind an LLM's verdict to a verified identity established in code."""
        return cls(
            identifier=identifier,
            identifier_kind=identifier_kind,
            display_name=display_name,
            **verdict.model_dump(),
        )


class RejectedCandidate(BaseModel):
    """A non-Buy Conviction Gate verdict on a new-buy candidate, as stored
    in the LangGraph Store (`nodes/_candidate_memory.py`) so the same
    candidate isn't re-researched and re-evaluated (a paid LLM call) on
    a later run without new information. See
    docs/superpowers/specs/2026-09-18-langgraph-store-design.md.

    identifier/identifier_kind/display_name reuse CandidateEvaluation's
    own identity convention (never a separately-invented identity
    scheme) - a rejection record names the same candidate the same way
    an evaluation does. "scheme_name" is the one addition beyond
    CandidateEvaluation's ticker/isin/scheme_code: it identifies the
    Mutual Fund Specialist's alias-index record (namespace
    ("rejected_candidates", "mutual_fund", "by_name")), whose identity
    is a normalized proposed fund name rather than an AMFI scheme code
    - a cheap pre-resolution alias, not the authoritative identity
    (that's the scheme_code record, written alongside it).

    identifier: the store key, stored redundantly so a search hit is
        self-describing. Upper-cased ticker for equity/ETF, the AMFI
        scheme_code as a string for the canonical mutual fund record,
        or the normalized proposed name for the alias record.
    specialist: which specialist produced this verdict.
    display_name: the verified name of the same subject
        (CandidateEvaluation.display_name - yfinance longName or the
        MFapi-resolved real scheme name), never the LLM's proposed name.
    group: the gap sector (GICS) or fund category the candidate was
        proposed for - metadata, not part of identity (see design
        spec's "What counts as the same candidate").
    action: the non-Buy verdict. Buy is never stored - see design
        spec's Architecture section 2.
    rejected_at: ISO-8601 UTC, the staleness clock.
    thread_id: which run produced it, for audit/replay in Studio later.
    schema_version: 1 - so a future shape change is detected, not
        silently guessed at.
    """

    identifier: str
    identifier_kind: Literal["ticker", "scheme_code", "scheme_name"]
    specialist: Literal["us_equity", "india_equity", "mutual_fund"]
    display_name: str
    group: str
    action: Literal["Hold", "Reduce", "Sell"]
    confidence: float = Field(ge=0.0, le=1.0)
    risk_flags: list[str]
    reasoning: str
    rejected_at: str
    thread_id: str | None
    schema_version: int = 1
