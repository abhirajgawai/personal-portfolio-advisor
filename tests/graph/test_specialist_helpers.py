"""Tests for cap-enforcement logic shared across specialist nodes."""

from __future__ import annotations


def test_apply_candidate_cap_enforces_per_sector_and_total_limits():
    from agentic_frameworks.graph.nodes._specialist_helpers import (
        _apply_candidate_cap,
    )
    from agentic_frameworks.graph.schemas import ProposedCandidate

    candidates = [
        ProposedCandidate(ticker="A1", sector="Energy"),
        ProposedCandidate(ticker="A2", sector="Energy"),
        ProposedCandidate(ticker="A3", sector="Energy"),
        ProposedCandidate(ticker="A4", sector="Energy"),
        ProposedCandidate(ticker="B1", sector="Health Care"),
        ProposedCandidate(ticker="B2", sector="Health Care"),
        ProposedCandidate(ticker="C1", sector="Financials"),
    ]

    capped = _apply_candidate_cap(candidates)

    assert len(capped) == 5
    energy_tickers = [c.ticker for c in capped if c.sector == "Energy"]
    assert len(energy_tickers) == 3
    assert "A4" not in energy_tickers


def test_filter_to_gap_sectors_drops_off_gap_candidates():
    from agentic_frameworks.graph.nodes._specialist_helpers import (
        _filter_to_gap_sectors,
    )
    from agentic_frameworks.graph.schemas import ProposedCandidate

    candidates = [
        ProposedCandidate(ticker="A1", sector="Energy"),
        ProposedCandidate(ticker="B1", sector="Information Technology"),
    ]

    filtered = _filter_to_gap_sectors(candidates, gap_sectors=["Energy"])

    assert len(filtered) == 1
    assert filtered[0].ticker == "A1"


def test_apply_candidate_cap_supports_custom_grouping_key():
    from agentic_frameworks.graph.nodes._specialist_helpers import (
        _apply_candidate_cap,
    )

    class FakeCandidate:
        def __init__(self, name, category):
            self.name = name
            self.category = category

    candidates = [
        FakeCandidate("A1", "Large Cap"),
        FakeCandidate("A2", "Large Cap"),
        FakeCandidate("A3", "Large Cap"),
        FakeCandidate("A4", "Large Cap"),
        FakeCandidate("B1", "Gilt"),
    ]

    capped = _apply_candidate_cap(candidates, key=lambda c: c.category)

    assert len(capped) == 4
    large_cap_names = [c.name for c in capped if c.category == "Large Cap"]
    assert len(large_cap_names) == 3
    assert "A4" not in large_cap_names


def test_filter_to_gap_sectors_supports_custom_grouping_key():
    from agentic_frameworks.graph.nodes._specialist_helpers import (
        _filter_to_gap_sectors,
    )

    class FakeCandidate:
        def __init__(self, name, category):
            self.name = name
            self.category = category

    candidates = [
        FakeCandidate("A1", "Gilt"),
        FakeCandidate("B1", "Large Cap"),
    ]

    filtered = _filter_to_gap_sectors(candidates, gap_sectors=["Gilt"], key=lambda c: c.category)

    assert len(filtered) == 1
    assert filtered[0].name == "A1"
