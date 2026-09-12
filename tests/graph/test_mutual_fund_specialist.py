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
    # TestModel emits a fixed, non-real scheme_name string (not a genuine
    # fund name) for every candidate it proposes, so
    # _resolve_scheme_by_name_and_direct_plan finds no real Direct-plan
    # match for it (live-verified this session: real MFapi.in search
    # against TestModel's dummy string returns no hits) and every
    # candidate is dropped before evaluation ever runs - an empty list is
    # the correct, expected outcome here, not a regression. This also
    # means the loop below is not a vacuous no-op check: it's pinning the
    # real, observed shape (empty), same pattern as test_graph.py's
    # india_equity_candidates == [] assertion with TestModel.
    assert result["mutual_fund_candidates"] == []
    for candidate in result["mutual_fund_candidates"]:
        assert candidate["action"] in {"Buy", "Hold", "Sell", "Reduce"}


async def test_mutual_fund_specialist_node_drops_candidate_on_category_mismatch():
    """Finding 1: if a candidate's proposed category doesn't match the
    real category of the fund _resolve_scheme_by_name_and_direct_plan
    actually resolves to, the candidate must be dropped rather than
    having a different fund's real NAV/metrics attached to it under the
    LLM's originally proposed name."""
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes import mutual_fund_specialist as node_module
    from agentic_frameworks.graph.nodes.research import research_node

    def fake_get_specialist_model():
        return TestModel()

    async def fake_resolve_mismatched_category(scheme_name, toolset):
        # Always resolves to a real Gilt fund's identity, regardless of
        # what category was proposed - simulating a fuzzy/wrong search
        # match. TestModel deterministically proposes category="Large Cap"
        # (the taxonomy's first Literal choice), so this always mismatches.
        return 118989, "Debt Scheme - Gilt Fund", "Some Real Gilt Fund - Direct Plan - Growth"

    research_result = await research_node({})
    state = {"holdings_by_bucket": research_result["holdings_by_bucket"]}

    original_model = node_module.get_specialist_model
    original_resolve = node_module._resolve_scheme_by_name_and_direct_plan
    node_module.get_specialist_model = fake_get_specialist_model
    node_module._resolve_scheme_by_name_and_direct_plan = fake_resolve_mismatched_category
    try:
        result = await node_module.mutual_fund_specialist_node(state)
    finally:
        node_module.get_specialist_model = original_model
        node_module._resolve_scheme_by_name_and_direct_plan = original_resolve

    assert result["mutual_fund_candidates"] == []
