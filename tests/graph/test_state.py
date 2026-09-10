from typing import get_type_hints

from agentic_frameworks.graph.state import ResearchState


def test_research_state_has_expected_keys():
    hints = get_type_hints(ResearchState)
    assert set(hints.keys()) == {"holdings", "allocation", "holdings_by_bucket"}


def test_risk_state_has_expected_keys():
    from agentic_frameworks.graph.state import RiskState

    hints = get_type_hints(RiskState)
    assert set(hints.keys()) == {
        "concentration",
        "category_overlap",
        "currency_exposure",
        "liquidity",
    }
    assert hints["concentration"] is dict
    assert hints["category_overlap"] is dict
    assert hints["currency_exposure"] is dict
    assert hints["liquidity"] is dict


def test_specialist_state_has_expected_keys():
    from agentic_frameworks.graph.state import SpecialistState

    hints = get_type_hints(SpecialistState)
    assert set(hints.keys()) == {"us_equity_etf_candidates"}
    assert hints["us_equity_etf_candidates"] == list[dict]


def test_graph_state_has_expected_keys():
    from agentic_frameworks.graph.state import GraphState

    hints = get_type_hints(GraphState)
    assert set(hints.keys()) == {
        "holdings",
        "allocation",
        "holdings_by_bucket",
        "concentration",
        "category_overlap",
        "currency_exposure",
        "liquidity",
        "us_equity_etf_candidates",
    }


def test_graph_state_includes_us_equity_etf_candidates():
    from agentic_frameworks.graph.state import GraphState

    hints = get_type_hints(GraphState)
    assert "us_equity_etf_candidates" in hints
    assert hints["us_equity_etf_candidates"] == list[dict]
