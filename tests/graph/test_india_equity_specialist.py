"""Tests for the India Equity Specialist node and its stages."""

from __future__ import annotations


async def test_compute_india_equity_gaps_finds_only_energy_given_real_holdings_and_real_estate_ownership():
    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _compute_india_equity_gaps,
    )
    from agentic_frameworks.graph.nodes.research import research_node

    research_result = await research_node({})

    gaps = _compute_india_equity_gaps(research_result["holdings_by_bucket"])

    assert set(gaps) == {"Energy"}


def test_compute_india_equity_gaps_raises_on_unmapped_sector():
    import pytest

    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _compute_india_equity_gaps,
    )

    holdings_by_bucket = {
        "equity": {
            "IN": [{"ticker": "ZZZ.NS", "name": "Fictional Co", "sector": "Not A Real Sector"}],
        }
    }

    with pytest.raises(KeyError):
        _compute_india_equity_gaps(holdings_by_bucket)


def test_compute_india_equity_gaps_skips_null_sector_rows():
    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _compute_india_equity_gaps,
    )

    holdings_by_bucket = {
        "equity": {
            "IN": [{"ticker": "MANUAL_TOTAL", "name": "Manual Total", "sector": None}],
        }
    }

    gaps = _compute_india_equity_gaps(holdings_by_bucket)

    assert len(gaps) == 11


def test_compute_india_equity_gaps_treats_real_estate_holding_as_covered():
    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _compute_india_equity_gaps,
    )

    holdings_by_bucket = {
        "equity": {"IN": []},
        "real_estate": {"IN": [{"ticker": "PROPERTY", "name": "Residential Property", "sector": "Residential"}]},
    }

    gaps = _compute_india_equity_gaps(holdings_by_bucket)

    assert "Real Estate" not in gaps


def test_compute_india_equity_gaps_still_flags_real_estate_when_no_real_estate_holding():
    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _compute_india_equity_gaps,
    )

    holdings_by_bucket = {"equity": {"IN": []}}

    gaps = _compute_india_equity_gaps(holdings_by_bucket)

    assert "Real Estate" in gaps


def test_summarize_india_equity_holdings_lists_real_in_equity():
    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _summarize_india_equity_holdings,
    )

    holdings_by_bucket = {
        "equity": {
            "IN": [{"ticker": "TCS", "name": "Tata Consultancy Services", "sector": "Information Technology"}],
            "US": [{"ticker": "GOOGL", "name": "Alphabet Inc", "sector": "Technology"}],
        },
    }

    summary = _summarize_india_equity_holdings(holdings_by_bucket)

    assert "TCS" in summary
    assert "GOOGL" not in summary


def test_summarize_india_equity_holdings_handles_no_in_holdings():
    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _summarize_india_equity_holdings,
    )

    summary = _summarize_india_equity_holdings({})

    assert "no India equities" in summary


async def test_india_candidate_generator_produces_proposed_candidates():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _build_candidate_generator,
    )

    generator = _build_candidate_generator(TestModel())
    result = await generator.run("propose candidates for: Energy")

    assert isinstance(result.output, list)
    assert all(hasattr(c, "ticker") and hasattr(c, "sector") for c in result.output)


async def test_india_conviction_gate_evaluator_produces_candidate_evaluation():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset
    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _build_conviction_gate_evaluator,
    )

    toolset = get_yfinance_toolset()
    async with toolset:
        evaluator = _build_conviction_gate_evaluator(
            TestModel(), toolset, "The user currently holds no India equities."
        )
        result = await evaluator.run("Evaluate ticker: RELIANCE.NS, gap sector: Energy")

        assert result.output.action in {"Buy", "Hold", "Sell", "Reduce"}
        assert isinstance(result.output.confidence, float)


async def test_india_equity_specialist_node_returns_candidate_evaluations():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes import india_equity_specialist as specialist_module

    def fake_get_specialist_model():
        return TestModel()

    state = {
        "holdings_by_bucket": {
            "equity": {
                "IN": [{"ticker": "TCS", "name": "Tata Consultancy Services", "sector": "Information Technology"}],
            },
        }
    }

    original = specialist_module.get_specialist_model
    specialist_module.get_specialist_model = fake_get_specialist_model
    try:
        result = await specialist_module.india_equity_specialist_node(state)
    finally:
        specialist_module.get_specialist_model = original

    assert "india_equity_candidates" in result
    assert isinstance(result["india_equity_candidates"], list)
    assert result["india_equity_candidates"]
    for candidate in result["india_equity_candidates"]:
        assert candidate["action"] in {"Buy", "Hold", "Sell", "Reduce"}


async def test_india_equity_specialist_node_returns_empty_list_when_no_gaps_and_skips_model_setup():
    from agentic_frameworks.graph.nodes import india_equity_specialist as specialist_module

    def compute_no_gaps(holdings_by_bucket):
        return []

    def fail_if_called():
        raise AssertionError("get_specialist_model should not be called when there are no gaps")

    original_compute = specialist_module._compute_india_equity_gaps
    original_model_getter = specialist_module.get_specialist_model
    specialist_module._compute_india_equity_gaps = compute_no_gaps
    specialist_module.get_specialist_model = fail_if_called
    try:
        result = await specialist_module.india_equity_specialist_node({"holdings_by_bucket": {}})
    finally:
        specialist_module._compute_india_equity_gaps = original_compute
        specialist_module.get_specialist_model = original_model_getter

    assert result == {"india_equity_candidates": []}
