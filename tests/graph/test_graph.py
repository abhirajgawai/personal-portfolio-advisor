"""Tests for the graph's node/edge wiring and end-to-end execution."""

from __future__ import annotations

from uuid import uuid4


def test_build_graph_wires_research_and_risk_analyst():
    from agentic_frameworks.graph.graph import build_graph

    graph = build_graph()

    assert set(graph.nodes.keys()) == {"research", "risk_analyst"}


async def test_graph_runs_end_to_end_with_real_data_and_persists_checkpoint():
    from agentic_frameworks.graph.checkpointer import get_checkpointer
    from agentic_frameworks.graph.graph import build_graph

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
        }
        assert len(result["holdings_by_bucket"]["real_estate"]["IN"]) == 1
        assert result["concentration"]["by_holding"]
        assert result["currency_exposure"]
        assert result["liquidity"]["illiquid"]["value_base"] > 0

        state_snapshot = await compiled.aget_state(config)
        assert state_snapshot.values
        assert state_snapshot.config["configurable"]["thread_id"] == thread_id
