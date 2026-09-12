"""Tests for the Mutual Fund Holdings Analyzer node."""

from __future__ import annotations


def test_lookup_benchmark_ticker_uses_index_fund_isin_mapping_first():
    from agentic_frameworks.graph.nodes.mutual_fund_holdings_analyzer import (
        _lookup_benchmark_ticker,
    )

    ticker = _lookup_benchmark_ticker(isin="INF179K01WN9", category="Index Fund")

    assert ticker == "^BSESN"


def test_lookup_benchmark_ticker_falls_back_to_category_table():
    from agentic_frameworks.graph.nodes.mutual_fund_holdings_analyzer import (
        _lookup_benchmark_ticker,
    )

    ticker = _lookup_benchmark_ticker(isin="INF879O01027", category="Flexi Cap")

    assert ticker == "^CRSLDX"


async def test_conviction_gate_evaluator_produces_candidate_evaluation():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset
    from agentic_frameworks.graph.nodes.mutual_fund_holdings_analyzer import (
        _build_conviction_gate_evaluator,
    )

    toolset = get_yfinance_toolset()
    async with toolset:
        evaluator = _build_conviction_gate_evaluator(TestModel(), toolset)
        result = await evaluator.run(
            "Evaluate existing holding: Parag Parikh Flexi Cap Fund (category: Flexi Cap). "
            "Metrics: 3yr CAGR 18%, volatility 15%, max drawdown -22%, track record 12 years. "
            "Benchmark (Nifty 500) difference: +4% annualized. Redundancy: low correlation with other holdings."
        )

        assert result.output.action in {"Buy", "Hold", "Sell", "Reduce"}
        assert isinstance(result.output.confidence, float)


async def test_mutual_fund_holdings_analyzer_node_returns_one_evaluation_per_real_fund():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes import mutual_fund_holdings_analyzer as node_module
    from agentic_frameworks.graph.nodes.research import research_node

    def fake_get_specialist_model():
        return TestModel()

    research_result = await research_node({})
    state = {"holdings_by_bucket": research_result["holdings_by_bucket"]}

    original = node_module.get_specialist_model
    node_module.get_specialist_model = fake_get_specialist_model
    try:
        result = await node_module.mutual_fund_holdings_analyzer_node(state)
    finally:
        node_module.get_specialist_model = original

    assert "existing_fund_evaluations" in result
    assert len(result["existing_fund_evaluations"]) == 9
    for evaluation in result["existing_fund_evaluations"]:
        assert evaluation["action"] in {"Buy", "Hold", "Sell", "Reduce"}
