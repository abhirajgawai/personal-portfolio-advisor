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


# --- SynthesizedRecommendation / AdvisorReport / HumanApproval (Phase 8) ---


def _synthesized_holding_guidance(**overrides):
    from agentic_frameworks.graph.schemas import SynthesizedHoldingGuidance

    defaults = dict(
        identifier="INF179K01WN9",
        action="Hold",
        rationale="Beats its benchmark modestly, short track record.",
        risk_flags=["short_track_record"],
        confidence=0.6,
    )
    defaults.update(overrides)
    return SynthesizedHoldingGuidance(**defaults)


def _synthesized_new_buy(**overrides):
    from agentic_frameworks.graph.schemas import SynthesizedNewBuy

    defaults = dict(
        identifier="XLV",
        rationale="Fills the Health Care gap.",
        gap_addressed="Health Care",
        position_size_pct_of_portfolio=1.5,
        entry_approach="Staggered",
        suggested_platform="moomoo",
        risk_flags=[],
        confidence=0.7,
    )
    defaults.update(overrides)
    return SynthesizedNewBuy(**defaults)


def test_synthesized_recommendation_empty_new_buys_with_reason_validates():
    from agentic_frameworks.graph.schemas import SynthesizedRecommendation

    recommendation = SynthesizedRecommendation(
        portfolio_assessment="Diversified but real-estate heavy.",
        allocation_notes=["Real estate is 34% of the portfolio."],
        existing_holdings_guidance=[_synthesized_holding_guidance()],
        new_buy_suggestions=[],
        no_new_buy_reason="No candidate cleared the Conviction Gate this cycle.",
    )

    assert recommendation.new_buy_suggestions == []
    assert recommendation.no_new_buy_reason is not None


def test_synthesized_recommendation_empty_new_buys_with_none_reason_fails():
    from agentic_frameworks.graph.schemas import SynthesizedRecommendation

    with pytest.raises(ValidationError):
        SynthesizedRecommendation(
            portfolio_assessment="Diversified but real-estate heavy.",
            allocation_notes=[],
            existing_holdings_guidance=[],
            new_buy_suggestions=[],
            no_new_buy_reason=None,
        )


def test_synthesized_recommendation_nonempty_new_buys_with_reason_fails():
    from agentic_frameworks.graph.schemas import SynthesizedRecommendation

    with pytest.raises(ValidationError):
        SynthesizedRecommendation(
            portfolio_assessment="Diversified but real-estate heavy.",
            allocation_notes=[],
            existing_holdings_guidance=[],
            new_buy_suggestions=[_synthesized_new_buy()],
            no_new_buy_reason="This should not be set alongside a non-empty list.",
        )


def test_synthesized_new_buy_rejects_position_size_over_cap():
    from agentic_frameworks.graph.schemas import MAX_NEW_POSITION_PCT

    with pytest.raises(ValidationError):
        _synthesized_new_buy(position_size_pct_of_portfolio=MAX_NEW_POSITION_PCT + 0.5)


def test_synthesized_new_buy_rejects_zero_or_negative_position_size():
    with pytest.raises(ValidationError):
        _synthesized_new_buy(position_size_pct_of_portfolio=0.0)


def test_synthesized_recommendation_enforces_max_new_buy_suggestions_cap():
    from agentic_frameworks.graph.schemas import MAX_NEW_BUY_SUGGESTIONS, SynthesizedRecommendation

    with pytest.raises(ValidationError):
        SynthesizedRecommendation(
            portfolio_assessment="x",
            allocation_notes=[],
            existing_holdings_guidance=[],
            new_buy_suggestions=[
                _synthesized_new_buy(identifier=f"T{i}") for i in range(MAX_NEW_BUY_SUGGESTIONS + 1)
            ],
            no_new_buy_reason=None,
        )


def test_synthesized_recommendation_has_no_identity_fields():
    """Structural half of the identity guarantee (design spec Architecture
    point 3): the LLM's output_type must never be able to name a
    display_name/identifier_kind - those are always re-bound in code."""
    from agentic_frameworks.graph.schemas import (
        SynthesizedHoldingGuidance,
        SynthesizedNewBuy,
        SynthesizedRecommendation,
    )

    assert "display_name" not in SynthesizedRecommendation.model_fields
    assert "identifier_kind" not in SynthesizedRecommendation.model_fields
    assert "display_name" not in SynthesizedHoldingGuidance.model_fields
    assert "identifier_kind" not in SynthesizedHoldingGuidance.model_fields
    assert "display_name" not in SynthesizedNewBuy.model_fields
    assert "identifier_kind" not in SynthesizedNewBuy.model_fields
    assert "specialist" not in SynthesizedNewBuy.model_fields


def test_advisor_report_from_synthesis_binds_identity_from_subject_index():
    from agentic_frameworks.graph.schemas import (
        AdvisorReport,
        ReportCoverage,
        SynthesizedRecommendation,
    )

    synthesis = SynthesizedRecommendation(
        portfolio_assessment="Diversified but real-estate heavy.",
        allocation_notes=["Real estate is 34% of the portfolio."],
        existing_holdings_guidance=[_synthesized_holding_guidance()],
        new_buy_suggestions=[_synthesized_new_buy()],
        no_new_buy_reason=None,
    )
    subject_index = {
        "INF179K01WN9": (
            "isin",
            "HDFC BSE Sensex Index Fund - Direct Plan - Growth Option",
            "existing",
            None,
        ),
        "XLV": ("ticker", "Health Care Select Sector SPDR Fund", "new_buy", "us_equity"),
    }
    coverage = ReportCoverage(
        holdings_total=54,
        holdings_individually_assessed=9,
        unassessed_buckets=["equity/IN", "equity/US"],
        coverage_note="9 of 54 holdings individually assessed.",
    )

    report = AdvisorReport.from_synthesis(synthesis, subject_index=subject_index, coverage=coverage)

    assert report.existing_holdings_guidance[0].display_name == (
        "HDFC BSE Sensex Index Fund - Direct Plan - Growth Option"
    )
    assert report.existing_holdings_guidance[0].identifier_kind == "isin"
    assert report.new_buy_suggestions[0].display_name == "Health Care Select Sector SPDR Fund"
    assert report.new_buy_suggestions[0].identifier_kind == "ticker"
    assert report.new_buy_suggestions[0].specialist == "us_equity"
    assert report.coverage.holdings_total == 54
    assert report.schema_version == 1
    assert report.generated_at


def test_human_approval_rejects_extra_keys():
    from agentic_frameworks.graph.schemas import HumanApproval

    with pytest.raises(ValidationError):
        HumanApproval(
            decision="approved",
            approved_by="abhi",
            excluded_identifiers=[],
            note=None,
            decided_at="2026-09-19T00:00:00+00:00",
            extra_field="not allowed",
        )


def test_human_approval_rejects_empty_approved_by():
    from agentic_frameworks.graph.schemas import HumanApproval

    with pytest.raises(ValidationError):
        HumanApproval(
            decision="approved",
            approved_by="",
            excluded_identifiers=[],
            note=None,
            decided_at="2026-09-19T00:00:00+00:00",
        )


def test_human_approval_rejects_missing_approved_by():
    from agentic_frameworks.graph.schemas import HumanApproval

    with pytest.raises(ValidationError):
        HumanApproval(
            decision="approved",
            excluded_identifiers=[],
            note=None,
            decided_at="2026-09-19T00:00:00+00:00",
        )


def test_human_approval_rejects_unknown_decision():
    from agentic_frameworks.graph.schemas import HumanApproval

    with pytest.raises(ValidationError):
        HumanApproval(
            decision="maybe",
            approved_by="abhi",
            excluded_identifiers=[],
            note=None,
            decided_at="2026-09-19T00:00:00+00:00",
        )


def test_human_approval_accepts_valid_approval():
    from agentic_frameworks.graph.schemas import HumanApproval

    approval = HumanApproval(
        decision="approved_with_exclusions",
        approved_by="abhi",
        excluded_identifiers=["XLV"],
        note="Skipping XLV for now.",
        decided_at="2026-09-19T00:00:00+00:00",
    )

    assert approval.decision == "approved_with_exclusions"
    assert approval.excluded_identifiers == ["XLV"]
