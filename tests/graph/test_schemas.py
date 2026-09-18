"""Tests for LLM output schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_candidate_evaluation_accepts_valid_data():
    from agentic_frameworks.graph.schemas import CandidateEvaluation

    evaluation = CandidateEvaluation(
        identifier="XOM",
        identifier_kind="ticker",
        display_name="Exxon Mobil Corporation",
        action="Buy",
        confidence=0.8,
        reasoning="Fills the Energy gap with strong fundamentals.",
        risk_flags=["sector_concentration"],
        entry_approach="Staggered",
        suggested_platform="moomoo",
    )

    assert evaluation.action == "Buy"
    assert evaluation.confidence == 0.8
    assert evaluation.identifier == "XOM"


def test_candidate_evaluation_rejects_invalid_action():
    from agentic_frameworks.graph.schemas import CandidateEvaluation

    with pytest.raises(ValidationError):
        CandidateEvaluation(
            identifier="XOM",
            identifier_kind="ticker",
            display_name="Exxon Mobil Corporation",
            action="Speculate",
            confidence=0.8,
            reasoning="not a real action",
            risk_flags=[],
            entry_approach=None,
            suggested_platform=None,
        )


def test_candidate_evaluation_rejects_out_of_range_confidence():
    from agentic_frameworks.graph.schemas import CandidateEvaluation

    with pytest.raises(ValidationError):
        CandidateEvaluation(
            identifier="XOM",
            identifier_kind="ticker",
            display_name="Exxon Mobil Corporation",
            action="Buy",
            confidence=7.0,
            reasoning="confidence should be a 0-1 probability, not 7.0",
            risk_flags=[],
            entry_approach=None,
            suggested_platform=None,
        )


def test_candidate_evaluation_requires_identity_fields():
    """A verdict with no subject is the bug this schema change fixes:
    identity must be impossible to omit, not merely conventional."""
    from agentic_frameworks.graph.schemas import CandidateEvaluation

    with pytest.raises(ValidationError):
        CandidateEvaluation(
            action="Buy",
            confidence=0.8,
            reasoning="no identifier - must not validate",
            risk_flags=[],
            entry_approach=None,
            suggested_platform=None,
        )


def test_candidate_evaluation_rejects_unknown_identifier_kind():
    from agentic_frameworks.graph.schemas import CandidateEvaluation

    with pytest.raises(ValidationError):
        CandidateEvaluation(
            identifier="INF179K01WN9",
            identifier_kind="folio_number",
            display_name="Some Fund",
            action="Hold",
            confidence=0.5,
            reasoning="folio_number is not one of the three real namespaces",
            risk_flags=[],
            entry_approach=None,
            suggested_platform=None,
        )


def test_conviction_verdict_has_no_identity_fields():
    """Structural guardrail: the LLM's output schema cannot name its own
    subject, so it can never mislabel one."""
    from agentic_frameworks.graph.schemas import ConvictionVerdict

    assert "identifier" not in ConvictionVerdict.model_fields
    assert "identifier_kind" not in ConvictionVerdict.model_fields
    assert "display_name" not in ConvictionVerdict.model_fields

    verdict = ConvictionVerdict(
        action="Hold",
        confidence=0.6,
        reasoning="Judgment only, no subject.",
        risk_flags=[],
        entry_approach=None,
        suggested_platform=None,
    )

    assert verdict.action == "Hold"


def test_candidate_evaluation_from_verdict_binds_identity_to_judgment():
    from agentic_frameworks.graph.schemas import CandidateEvaluation, ConvictionVerdict

    verdict = ConvictionVerdict(
        action="Buy",
        confidence=0.72,
        reasoning="Beats its benchmark with moderate drawdown.",
        risk_flags=["short_track_record"],
        entry_approach="Staggered",
        suggested_platform="Groww",
    )

    evaluation = CandidateEvaluation.from_verdict(
        verdict,
        identifier="118632",
        identifier_kind="scheme_code",
        display_name="Nippon India Large Cap Fund - Direct Plan - Growth Option",
    )

    assert evaluation.identifier == "118632"
    assert evaluation.identifier_kind == "scheme_code"
    assert evaluation.display_name.startswith("Nippon India Large Cap Fund")
    assert evaluation.action == "Buy"
    assert evaluation.confidence == 0.72
    assert evaluation.risk_flags == ["short_track_record"]
    assert set(evaluation.model_dump()) == {
        "identifier",
        "identifier_kind",
        "display_name",
        "action",
        "confidence",
        "reasoning",
        "risk_flags",
        "entry_approach",
        "suggested_platform",
    }


def test_proposed_candidate_requires_ticker_and_sector():
    from agentic_frameworks.graph.schemas import ProposedCandidate

    candidate = ProposedCandidate(ticker="XOM", sector="Energy")

    assert candidate.ticker == "XOM"
    assert candidate.sector == "Energy"


def test_proposed_candidate_rejects_unknown_sector():
    from agentic_frameworks.graph.schemas import ProposedCandidate

    with pytest.raises(ValidationError):
        ProposedCandidate(ticker="XOM", sector="Not A Real Sector")


def test_proposed_candidate_accepts_information_technology():
    from agentic_frameworks.graph.schemas import ProposedCandidate

    candidate = ProposedCandidate(ticker="TCS.NS", sector="Information Technology")

    assert candidate.sector == "Information Technology"


def test_gics_sectors_has_eleven_standard_buckets():
    from agentic_frameworks.graph.schemas import GICS_SECTORS

    assert len(GICS_SECTORS) == 11
    assert "Information Technology" in GICS_SECTORS
    assert "Energy" in GICS_SECTORS
    assert "Real Estate" in GICS_SECTORS


def test_mf_category_taxonomy_has_expected_entries():
    from agentic_frameworks.graph.schemas import MF_CATEGORY_TAXONOMY

    assert "Large Cap" in MF_CATEGORY_TAXONOMY
    assert "Gilt" in MF_CATEGORY_TAXONOMY
    assert "Index Fund" in MF_CATEGORY_TAXONOMY
    assert len(MF_CATEGORY_TAXONOMY) == 13


def test_proposed_fund_candidate_requires_scheme_name_and_category():
    from agentic_frameworks.graph.schemas import ProposedFundCandidate

    candidate = ProposedFundCandidate(scheme_name="Some Large Cap Fund Direct Growth", category="Large Cap")

    assert candidate.scheme_name == "Some Large Cap Fund Direct Growth"
    assert candidate.category == "Large Cap"


def test_proposed_fund_candidate_rejects_unknown_category():
    from agentic_frameworks.graph.schemas import ProposedFundCandidate

    with pytest.raises(ValidationError):
        ProposedFundCandidate(scheme_name="Some Fund", category="Not A Real Category")
