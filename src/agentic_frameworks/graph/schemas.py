"""Structured output schemas for LLM-driven nodes.

Every LLM call in this graph must produce a strictly validated
schema - no freeform text leaks through unvalidated, per plan.md.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


# --- Synthesizer output/state schemas (Phase 8) ---
#
# Hard numeric caps, enforced in code (Field(max_length=...) / Field(le=...)
# on the schema; MAX_NEW_CAPITAL_PCT via the Synthesizer's output_validator -
# see nodes/synthesizer.py) per CLAUDE.md's LLM cost control rule 5: "hard
# caps on generation, not soft guidance". Never soft prompt wording alone.
MAX_NEW_BUY_SUGGESTIONS = 5
MAX_EXISTING_HOLDINGS_GUIDANCE = 15
ALLOCATION_NOTES_CAP = 8

# Per-string length caps on free-text fields, added 2026-09-19 after a real
# model run genuinely exceeded pydantic-ai's bounded output retries
# (UnexpectedModelBehavior) - the model produced 6-9 risk_flags per holding
# across 9 holdings plus 2 new-buy candidates, each flag hundreds of
# characters long, and the response was truncated mid-JSON before the
# output parser ever got a chance to validate it. Unbounded free-text
# fields are exactly the "soft guidance" CLAUDE.md's cost-control rule 5
# warns against - a verbose model can blow the output token budget before
# a single ModelRetry is even possible, since retries only fire on a
# *validation* failure, not a truncated-response failure. These caps make
# the failure mode itself (running out of output tokens) far less likely,
# on top of the pre-existing hard caps on how many items a list may hold.
RISK_FLAGS_MAX_ITEMS = 4
RISK_FLAG_MAX_CHARS = 220
RATIONALE_MAX_CHARS = 400
GAP_ADDRESSED_MAX_CHARS = 200
PORTFOLIO_ASSESSMENT_MAX_CHARS = 1200
ALLOCATION_NOTE_MAX_CHARS = 220

RiskFlag = Annotated[str, Field(max_length=RISK_FLAG_MAX_CHARS)]
RiskFlagList = Annotated[list[RiskFlag], Field(max_length=RISK_FLAGS_MAX_ITEMS)]
Rationale = Annotated[str, Field(max_length=RATIONALE_MAX_CHARS)]
AllocationNote = Annotated[str, Field(max_length=ALLOCATION_NOTE_MAX_CHARS)]

# Position-sizing bounds. Per-suggestion (MAX_NEW_POSITION_PCT) is enforced
# right here in the schema; the run-aggregate (MAX_NEW_CAPITAL_PCT) is
# enforced in code in the Synthesizer's output_validator (nodes/
# synthesizer.py), since it's a property of the whole list, not one field.
#
# *** NOT A FINAL DECISION - DEFAULTS ONLY ***
# These two numbers are real-money parameters (2% of the real portfolio is
# roughly Rs 2 lakh per suggestion) accepted as a conservative, industry-
# common DEFAULT during an autonomous overnight session (design spec
# docs/superpowers/specs/2026-09-19-synthesizer-design.md, "Decisions" #3).
# The STRUCTURAL cap (a hard numeric ceiling enforced in code, never LLM
# guidance) is non-negotiable and already in effect below - but the specific
# percentages need Abhi's explicit sign-off before any suggestion sized
# against them is trusted. See HANDOFF.md's Phase 8 section.
MAX_NEW_POSITION_PCT = 2.0
MAX_NEW_CAPITAL_PCT = 5.0


class SynthesizedHoldingGuidance(BaseModel):
    """Guidance on one existing holding - what the LLM's output_type
    carries. No identity fields (see ConvictionVerdict's docstring for the
    same reasoning): `identifier` alone is enough for the Synthesizer to
    select which holding it's talking about, but `identifier_kind`/
    `display_name` are re-bound in node code from the subject index built
    from real upstream data, never trusted from the model.
    """

    identifier: str
    action: Literal["Hold", "AddTo", "Reduce", "Sell"]
    rationale: Rationale
    risk_flags: RiskFlagList
    confidence: float = Field(ge=0.0, le=1.0)


class SynthesizedNewBuy(BaseModel):
    """One new-buy suggestion, as the LLM's output_type carries it.

    position_size_pct_of_portfolio is bounded by MAX_NEW_POSITION_PCT right
    in the schema - a pydantic ValidationError (surfaced by pydantic-ai as a
    bounded retry) rather than soft prompt wording, per CLAUDE.md's cost
    control rule 5. suggested_platform is a plain str here (structurally
    unbounded); the live platform-membership check happens in the
    Synthesizer's output_validator, since only node code knows this run's
    real, live-derived platform set (see design spec Background).
    """

    identifier: str
    rationale: Rationale
    gap_addressed: Annotated[str, Field(max_length=GAP_ADDRESSED_MAX_CHARS)]
    position_size_pct_of_portfolio: float = Field(gt=0.0, le=MAX_NEW_POSITION_PCT)
    entry_approach: Literal["LumpSum", "Staggered"]
    suggested_platform: str
    risk_flags: RiskFlagList
    confidence: float = Field(ge=0.0, le=1.0)


class SynthesizedRecommendation(BaseModel):
    """The Synthesizer Agent's output_type - identity-poor by design (see
    module docstring's identifier-only convention). Deliberately carries NO
    `display_name`/`identifier_kind` fields anywhere in this model or its
    two list members - those are re-bound in node code from a subject index
    the node itself builds from verified upstream data (see
    nodes/synthesizer.py._build_subject_index), exactly like
    ConvictionVerdict -> CandidateEvaluation.

    "Avoid is the default" is structural here, not instructional: the
    model_validator below enforces that an empty new_buy_suggestions list
    always carries an explained no_new_buy_reason, and a non-empty one never
    does (that string is deterministic, code-computed - see design spec
    Architecture point 4).
    """

    portfolio_assessment: Annotated[str, Field(max_length=PORTFOLIO_ASSESSMENT_MAX_CHARS)]
    allocation_notes: Annotated[
        list[AllocationNote], Field(max_length=ALLOCATION_NOTES_CAP)
    ]
    existing_holdings_guidance: list[SynthesizedHoldingGuidance] = Field(
        max_length=MAX_EXISTING_HOLDINGS_GUIDANCE
    )
    new_buy_suggestions: list[SynthesizedNewBuy] = Field(max_length=MAX_NEW_BUY_SUGGESTIONS)
    no_new_buy_reason: str | None = None

    @model_validator(mode="after")
    def _check_no_new_buy_reason_biconditional(self) -> SynthesizedRecommendation:
        is_empty = len(self.new_buy_suggestions) == 0
        has_reason = self.no_new_buy_reason is not None
        if is_empty != has_reason:
            raise ValueError(
                "no_new_buy_reason must be set if and only if new_buy_suggestions is empty "
                f"(new_buy_suggestions empty={is_empty}, no_new_buy_reason set={has_reason})."
            )
        return self


class ExistingHoldingGuidance(SynthesizedHoldingGuidance):
    """What lands in state for one existing-holding guidance item -
    SynthesizedHoldingGuidance plus the identity re-bound in code from the
    subject index (never from the model). See AdvisorReport.from_synthesis.
    """

    identifier_kind: Literal["ticker", "isin", "scheme_code"]
    display_name: str


class NewBuySuggestion(SynthesizedNewBuy):
    """What lands in state for one new-buy suggestion - SynthesizedNewBuy
    plus the identity and originating specialist, both re-bound in code
    from the subject index (never from the model). See
    AdvisorReport.from_synthesis.
    """

    identifier_kind: Literal["ticker", "scheme_code"]
    display_name: str
    specialist: Literal["us_equity", "india_equity", "mutual_fund"]


class ReportCoverage(BaseModel):
    """Built entirely in code (nodes/synthesizer.py._compute_coverage),
    never by the LLM - the honest, real count of how much of the portfolio
    this report actually judged individually, versus only at the
    allocation level. See design spec's Out of Scope section: no node in
    this system judges the 36 IN equities / 3 US equities / 4 ETFs
    individually yet.
    """

    holdings_total: int
    holdings_individually_assessed: int
    unassessed_buckets: list[str]
    coverage_note: str


class AdvisorReport(BaseModel):
    """What actually lands in GraphState["recommendation"] - the
    Synthesizer's validated output plus code-rebound identity and
    code-computed coverage. Never constructed directly from an LLM
    output_type; always via from_synthesis, so the binding step can never
    be skipped.
    """

    portfolio_assessment: str
    allocation_notes: list[str]
    existing_holdings_guidance: list[ExistingHoldingGuidance]
    new_buy_suggestions: list[NewBuySuggestion]
    no_new_buy_reason: str | None
    coverage: ReportCoverage
    generated_at: str
    schema_version: int = 1

    @classmethod
    def from_synthesis(
        cls,
        synthesis: SynthesizedRecommendation,
        *,
        subject_index: dict[str, tuple[str, str, str, str | None]],
        coverage: ReportCoverage,
    ) -> AdvisorReport:
        """Bind an LLM's synthesis to verified identity established in code.

        subject_index: {identifier: (identifier_kind, display_name, side,
            specialist)} - specialist is None for existing-holdings entries.
        The output_validator that ran before this is what guarantees every
        identifier referenced below is actually present in subject_index
        (and on the correct side) - this method trusts that guarantee
        rather than re-checking it, so a KeyError here would mean the
        validator was bypassed, which should never happen.
        """
        existing_holdings_guidance = []
        for guidance in synthesis.existing_holdings_guidance:
            identifier_kind, display_name, _side, _specialist = subject_index[guidance.identifier]
            existing_holdings_guidance.append(
                ExistingHoldingGuidance(
                    identifier_kind=identifier_kind,
                    display_name=display_name,
                    **guidance.model_dump(),
                )
            )

        new_buy_suggestions = []
        for suggestion in synthesis.new_buy_suggestions:
            identifier_kind, display_name, _side, specialist = subject_index[suggestion.identifier]
            new_buy_suggestions.append(
                NewBuySuggestion(
                    identifier_kind=identifier_kind,
                    display_name=display_name,
                    specialist=specialist,
                    **suggestion.model_dump(),
                )
            )

        return cls(
            portfolio_assessment=synthesis.portfolio_assessment,
            allocation_notes=synthesis.allocation_notes,
            existing_holdings_guidance=existing_holdings_guidance,
            new_buy_suggestions=new_buy_suggestions,
            no_new_buy_reason=synthesis.no_new_buy_reason,
            coverage=coverage,
            generated_at=datetime.now(UTC).isoformat(),
            schema_version=1,
        )


class HumanApproval(BaseModel):
    """The resume payload contract for human_approval_gate_node's
    interrupt(). extra="forbid" and no defaults on decision/approved_by:
    a missing, empty, or malformed resume payload must raise, never fall
    through to an implicit approval (design spec's "why nothing in this
    system can auto-approve it", reason 2).
    """

    model_config = ConfigDict(extra="forbid")

    decision: Literal["approved", "rejected", "approved_with_exclusions"]
    approved_by: str = Field(min_length=1)
    excluded_identifiers: list[str] = Field(default_factory=list)
    note: str | None = None
    decided_at: str
