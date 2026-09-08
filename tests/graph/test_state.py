from typing import get_type_hints

from agentic_frameworks.graph.state import ResearchState


def test_research_state_has_expected_keys():
    hints = get_type_hints(ResearchState)
    assert set(hints.keys()) == {"holdings", "allocation", "holdings_by_bucket"}
