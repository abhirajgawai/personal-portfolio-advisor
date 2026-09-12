"""Tests for the Mutual Fund Specialist (new-buy) node."""

from __future__ import annotations


async def test_candidate_generator_produces_proposed_fund_candidates():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes.mutual_fund_specialist import (
        _build_candidate_generator,
    )

    generator = _build_candidate_generator(TestModel())
    result = await generator.run("propose candidates for: Gilt")

    assert isinstance(result.output, list)
    assert all(hasattr(c, "scheme_name") and hasattr(c, "category") for c in result.output)


async def test_mutual_fund_specialist_node_returns_candidate_evaluations():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes import mutual_fund_specialist as node_module
    from agentic_frameworks.graph.nodes.research import research_node

    def fake_get_specialist_model():
        return TestModel()

    research_result = await research_node({})
    state = {"holdings_by_bucket": research_result["holdings_by_bucket"]}

    original = node_module.get_specialist_model
    node_module.get_specialist_model = fake_get_specialist_model
    try:
        result = await node_module.mutual_fund_specialist_node(state)
    finally:
        node_module.get_specialist_model = original

    assert "mutual_fund_candidates" in result
    assert isinstance(result["mutual_fund_candidates"], list)
    for candidate in result["mutual_fund_candidates"]:
        assert candidate["action"] in {"Buy", "Hold", "Sell", "Reduce"}
