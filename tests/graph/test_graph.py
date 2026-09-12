"""Tests for the graph's node/edge wiring and end-to-end execution."""

from __future__ import annotations

from uuid import uuid4


def test_build_graph_wires_all_five_nodes():
    from agentic_frameworks.graph.graph import build_graph

    graph = build_graph()

    assert set(graph.nodes.keys()) == {
        "research",
        "risk_analyst",
        "us_equity_etf_specialist",
        "india_equity_specialist",
        "mutual_fund_holdings_analyzer",
        "mutual_fund_specialist",
    }


async def test_graph_runs_end_to_end_with_real_data_and_persists_checkpoint():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.checkpointer import get_checkpointer
    from agentic_frameworks.graph.graph import build_graph
    from agentic_frameworks.graph.nodes import india_equity_specialist as india_module
    from agentic_frameworks.graph.nodes import mutual_fund_holdings_analyzer as mf_holdings_module
    from agentic_frameworks.graph.nodes import mutual_fund_specialist as mf_specialist_module
    from agentic_frameworks.graph.nodes import us_equity_etf_specialist as us_module

    def fake_get_specialist_model():
        return TestModel()

    modules = [india_module, mf_holdings_module, mf_specialist_module, us_module]
    originals = [m.get_specialist_model for m in modules]
    for m in modules:
        m.get_specialist_model = fake_get_specialist_model
    try:
        async with get_checkpointer() as checkpointer:
            compiled = build_graph().compile(checkpointer=checkpointer)
            thread_id = f"test-graph-harness-e2e-{uuid4()}"
            config = {"configurable": {"thread_id": thread_id}}

            result = await compiled.ainvoke({}, config=config)

            assert set(result.keys()) == {
                "holdings",
                "allocation",
                "holdings_by_bucket",
                "concentration",
                "category_overlap",
                "currency_exposure",
                "liquidity",
                "us_equity_etf_candidates",
                "india_equity_candidates",
                "existing_fund_evaluations",
                "mutual_fund_candidates",
            }
            assert len(result["holdings_by_bucket"]["real_estate"]["IN"]) == 1
            assert result["concentration"]["by_holding"]
            assert result["currency_exposure"]
            assert result["liquidity"]["illiquid"]["value_base"] > 0
            assert result["us_equity_etf_candidates"]
            # TestModel deterministically emits sector="Health Care" (the
            # first Literal choice) regardless of prompt content. "Health
            # Care" is a real gap for the US specialist's real holdings but
            # not for India's (real India gaps are Energy/Real Estate), so
            # the (correct, intentional) gap-filter added to fix the
            # off-gap-candidate finding drops it here - an empty list is
            # the right outcome, not a regression.
            assert result["india_equity_candidates"] == []
            assert len(result["existing_fund_evaluations"]) == 9

            state_snapshot = await compiled.aget_state(config)
            assert state_snapshot.values
            assert state_snapshot.config["configurable"]["thread_id"] == thread_id
    finally:
        for m, original in zip(modules, originals):
            m.get_specialist_model = original
