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


def test_apply_candidate_cap_enforces_per_sector_and_total_limits():
    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import (
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


async def test_candidate_generator_produces_proposed_candidates():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import (
        _build_candidate_generator,
    )

    generator = _build_candidate_generator(TestModel())
    result = await generator.run("propose candidates for: Energy")

    assert isinstance(result.output, list)
    assert all(hasattr(c, "ticker") and hasattr(c, "sector") for c in result.output)


async def test_conviction_gate_evaluator_produces_candidate_evaluation():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset
    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import (
        _build_conviction_gate_evaluator,
    )

    toolset = get_yfinance_toolset()
    async with toolset:
        evaluator = _build_conviction_gate_evaluator(TestModel(), toolset)
        result = await evaluator.run("Evaluate ticker: XOM, gap sector: Energy")

        assert result.output.action in {"Buy", "Hold", "Sell", "Reduce"}
        assert isinstance(result.output.confidence, float)


async def test_us_equity_etf_specialist_node_returns_candidate_evaluations():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes import us_equity_etf_specialist as specialist_module

    def fake_get_specialist_model():
        return TestModel()

    original = specialist_module.get_specialist_model
    specialist_module.get_specialist_model = fake_get_specialist_model
    try:
        result = await specialist_module.us_equity_etf_specialist_node({})
    finally:
        specialist_module.get_specialist_model = original

    assert "us_equity_etf_candidates" in result
    assert isinstance(result["us_equity_etf_candidates"], list)
    for candidate in result["us_equity_etf_candidates"]:
        assert candidate["action"] in {"Buy", "Hold", "Sell", "Reduce"}
