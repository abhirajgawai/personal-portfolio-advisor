"""Tests for the graph's node/edge wiring and end-to-end execution."""

from __future__ import annotations

from uuid import uuid4


def test_build_graph_wires_all_six_nodes():
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
    from agentic_frameworks.graph.store import get_memory_store

    def fake_get_specialist_model():
        return TestModel()

    modules = [india_module, mf_holdings_module, mf_specialist_module, us_module]
    originals = [m.get_specialist_model for m in modules]
    for m in modules:
        m.get_specialist_model = fake_get_specialist_model
    try:
        async with get_checkpointer() as checkpointer, get_memory_store() as store:
            compiled = build_graph().compile(checkpointer=checkpointer, store=store)
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


async def test_graph_coexists_checkpointer_and_store_on_same_file_under_real_fan_out(tmp_path):
    """The test the design spec calls out as catching a future SQLite-
    concurrency regression: compile the real graph with both a
    checkpointer and a store pointed at one temp file, run the real
    four-parallel-branch fan-out with TestModel, and confirm both a
    checkpoint and a real store ROW (not just the store table existing -
    `store.setup()` creates that table unconditionally, whether or not
    any node ever actually writes to it) exist afterwards with no lock
    errors.

    TestModel alone can't produce that row: it deterministically emits
    action="Buy", the one verdict this design never persists to the
    rejection store (see the specialist suppression tests). So the US
    branch's Conviction Gate evaluator is substituted with a fake that
    returns a real "Hold" verdict - same fake-evaluator pattern already
    used by the specialists' own cross-run-suppression tests - forcing
    this run to actually write a rejection row."""
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.store.sqlite.aio import AsyncSqliteStore
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.graph import build_graph
    from agentic_frameworks.graph.nodes import india_equity_specialist as india_module
    from agentic_frameworks.graph.nodes import mutual_fund_holdings_analyzer as mf_holdings_module
    from agentic_frameworks.graph.nodes import mutual_fund_specialist as mf_specialist_module
    from agentic_frameworks.graph.nodes import us_equity_etf_specialist as us_module
    from agentic_frameworks.graph.schemas import ConvictionVerdict

    def fake_get_specialist_model():
        return TestModel()

    class _FakeHoldEvalResult:
        output = ConvictionVerdict(
            action="Hold",
            confidence=0.5,
            reasoning="Fairly valued, no clear near-term catalyst.",
            risk_flags=["no_clear_edge"],
            entry_approach=None,
            suggested_platform=None,
        )

    class _FakeHoldEvaluator:
        async def run(self, prompt):
            return _FakeHoldEvalResult()

    modules = [india_module, mf_holdings_module, mf_specialist_module, us_module]
    originals = [m.get_specialist_model for m in modules]
    for m in modules:
        m.get_specialist_model = fake_get_specialist_model

    original_build_evaluator = us_module._build_conviction_gate_evaluator
    us_module._build_conviction_gate_evaluator = lambda *args, **kwargs: _FakeHoldEvaluator()

    db_path = str(tmp_path / "test_coexistence.db")
    try:
        async with (
            AsyncSqliteSaver.from_conn_string(db_path) as checkpointer,
            AsyncSqliteStore.from_conn_string(db_path) as store,
        ):
            await store.setup()
            compiled = build_graph().compile(checkpointer=checkpointer, store=store)
            config = {"configurable": {"thread_id": f"test-coexistence-{uuid4()}"}}

            result = await compiled.ainvoke({}, config=config)

            assert result["us_equity_etf_candidates"]
            assert result["us_equity_etf_candidates"][0]["action"] == "Hold"
    finally:
        for m, original in zip(modules, originals):
            m.get_specialist_model = original
        us_module._build_conviction_gate_evaluator = original_build_evaluator

    async with aiosqlite.connect(db_path) as conn:
        cursor = await conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in await cursor.fetchall()}
        assert {"checkpoints", "writes", "store", "store_migrations"} <= tables

        row_cursor = await conn.execute("SELECT COUNT(*) FROM store")
        (row_count,) = await row_cursor.fetchone()

    assert row_count > 0
