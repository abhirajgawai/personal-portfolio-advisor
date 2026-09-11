"""Tests for the US Equity/ETF Specialist node and its stages."""

from __future__ import annotations


async def test_verify_ticker_confirms_real_instrument():
    from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset
    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import _verify_ticker

    toolset = get_yfinance_toolset()
    async with toolset:
        assert await _verify_ticker("AAPL", toolset) is True


async def test_verify_ticker_rejects_fake_ticker():
    from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset
    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import _verify_ticker

    toolset = get_yfinance_toolset()
    async with toolset:
        assert await _verify_ticker("NOTAREALTICKER999", toolset) is False


async def test_candidate_generator_produces_proposed_candidates():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import (
        _build_candidate_generator,
    )

    generator = _build_candidate_generator(TestModel())
    result = await generator.run("propose candidates for: Energy")

    assert isinstance(result.output, list)
    assert all(hasattr(c, "ticker") and hasattr(c, "sector") for c in result.output)


def test_summarize_us_holdings_lists_real_equity_and_etf_holdings():
    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import (
        _summarize_us_holdings,
    )

    holdings_by_bucket = {
        "equity": {
            "US": [{"ticker": "GOOGL", "name": "Alphabet Inc", "sector": "Technology"}],
            "IN": [{"ticker": "TCS", "name": "Tata Consultancy Services", "sector": "IT"}],
        },
        "etf": {
            "US": [{"ticker": "VOO", "name": "Vanguard S&P 500 ETF", "sector": "US Large Cap"}],
        },
    }

    summary = _summarize_us_holdings(holdings_by_bucket)

    assert "GOOGL" in summary
    assert "VOO" in summary
    assert "TCS" not in summary


def test_summarize_us_holdings_handles_no_us_holdings():
    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import (
        _summarize_us_holdings,
    )

    summary = _summarize_us_holdings({})

    assert "no US equities or ETFs" in summary


async def test_conviction_gate_evaluator_produces_candidate_evaluation():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset
    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import (
        _build_conviction_gate_evaluator,
    )

    toolset = get_yfinance_toolset()
    async with toolset:
        evaluator = _build_conviction_gate_evaluator(
            TestModel(), toolset, "The user currently holds no US equities or ETFs."
        )
        result = await evaluator.run("Evaluate ticker: XOM, gap sector: Energy")

        assert result.output.action in {"Buy", "Hold", "Sell", "Reduce"}
        assert isinstance(result.output.confidence, float)


async def test_compute_us_gaps_excludes_information_technology_given_real_holdings():
    from agentic_frameworks.graph.nodes.research import research_node
    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import (
        _compute_us_gaps,
    )

    research_result = await research_node({})

    gaps = _compute_us_gaps(research_result["holdings_by_bucket"])

    assert "Information Technology" not in gaps
    assert "Energy" in gaps
    assert "Health Care" in gaps
    assert len(gaps) == 10


def test_compute_us_gaps_raises_on_unmapped_sector():
    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import _compute_us_gaps

    holdings_by_bucket = {
        "equity": {
            "US": [{"ticker": "ZZZ", "name": "Fictional Co", "sector": "Not A Real Sector"}],
        }
    }

    with __import__("pytest").raises(KeyError):
        _compute_us_gaps(holdings_by_bucket)


def test_compute_us_gaps_skips_null_sector_rows():
    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import _compute_us_gaps

    holdings_by_bucket = {
        "equity": {
            "US": [{"ticker": "MANUAL_TOTAL", "name": "Manual Total", "sector": None}],
        }
    }

    gaps = _compute_us_gaps(holdings_by_bucket)

    assert len(gaps) == 11


async def test_us_equity_etf_specialist_node_returns_candidate_evaluations():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes import us_equity_etf_specialist as specialist_module

    def fake_get_specialist_model():
        return TestModel()

    state = {
        "holdings_by_bucket": {
            "equity": {
                "US": [{"ticker": "GOOGL", "name": "Alphabet Inc", "sector": "Technology"}],
            },
            "etf": {
                "US": [{"ticker": "VOO", "name": "Vanguard S&P 500 ETF", "sector": "US Large Cap"}],
            },
        }
    }

    original = specialist_module.get_specialist_model
    specialist_module.get_specialist_model = fake_get_specialist_model
    try:
        result = await specialist_module.us_equity_etf_specialist_node(state)
    finally:
        specialist_module.get_specialist_model = original

    assert "us_equity_etf_candidates" in result
    assert isinstance(result["us_equity_etf_candidates"], list)
    assert result["us_equity_etf_candidates"]
    for candidate in result["us_equity_etf_candidates"]:
        assert candidate["action"] in {"Buy", "Hold", "Sell", "Reduce"}


async def test_us_equity_etf_specialist_node_returns_empty_list_when_no_gaps_and_skips_model_setup():
    from agentic_frameworks.graph.nodes import us_equity_etf_specialist as specialist_module

    def compute_no_gaps(holdings_by_bucket):
        return []

    def fail_if_called():
        raise AssertionError("get_specialist_model should not be called when there are no gaps")

    original_compute = specialist_module._compute_us_gaps
    original_model_getter = specialist_module.get_specialist_model
    specialist_module._compute_us_gaps = compute_no_gaps
    specialist_module.get_specialist_model = fail_if_called
    try:
        result = await specialist_module.us_equity_etf_specialist_node({"holdings_by_bucket": {}})
    finally:
        specialist_module._compute_us_gaps = original_compute
        specialist_module.get_specialist_model = original_model_getter

    assert result == {"us_equity_etf_candidates": []}
