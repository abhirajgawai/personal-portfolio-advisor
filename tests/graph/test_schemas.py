"""Tests for LLM output schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_candidate_evaluation_accepts_valid_data():
    from agentic_frameworks.graph.schemas import CandidateEvaluation

    evaluation = CandidateEvaluation(
        action="Buy",
        confidence=0.8,
        reasoning="Fills the Energy gap with strong fundamentals.",
        risk_flags=["sector_concentration"],
        entry_approach="Staggered",
        suggested_platform="moomoo",
    )

    assert evaluation.action == "Buy"
    assert evaluation.confidence == 0.8


def test_candidate_evaluation_rejects_invalid_action():
    from agentic_frameworks.graph.schemas import CandidateEvaluation

    with pytest.raises(ValidationError):
        CandidateEvaluation(
            action="Speculate",
            confidence=0.8,
            reasoning="not a real action",
            risk_flags=[],
            entry_approach=None,
            suggested_platform=None,
        )


def test_proposed_candidate_requires_ticker_and_sector():
    from agentic_frameworks.graph.schemas import ProposedCandidate

    candidate = ProposedCandidate(ticker="XOM", sector="Energy")

    assert candidate.ticker == "XOM"
    assert candidate.sector == "Energy"
