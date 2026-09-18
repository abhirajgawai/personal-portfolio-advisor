"""Tests for the graph's node/edge wiring and end-to-end execution."""

from __future__ import annotations

from uuid import uuid4


def test_build_graph_wires_all_eight_nodes():
    from agentic_frameworks.graph.graph import build_graph

    graph = build_graph()

    assert set(graph.nodes.keys()) == {
        "research",
        "risk_analyst",
        "us_equity_etf_specialist",
        "india_equity_specialist",
        "mutual_fund_holdings_analyzer",
        "mutual_fund_specialist",
        "synthesizer",
        "human_approval_gate",
    }


def test_build_compiled_graph_requires_checkpointer_and_store_as_keyword_only():
    """Design spec Decision 5: both parameters are keyword-only and
    required, so there is no way to compile the advisor graph without the
    two things its guardrails structurally depend on."""
    import inspect

    from agentic_frameworks.graph.graph import build_compiled_graph

    signature = inspect.signature(build_compiled_graph)
    for name in ("checkpointer", "store"):
        param = signature.parameters[name]
        assert param.kind == inspect.Parameter.KEYWORD_ONLY
        assert param.default is inspect.Parameter.empty


def _empty_synthesis_function_model():
    """A FunctionModel that always returns a schema-valid, empty-both-
    sides SynthesizedRecommendation, regardless of prompt content - valid
    no matter which real identifiers the run actually produced, since an
    empty list can never fail the output_validator's membership checks.
    Used by graph-wiring/fan-in tests below, whose job is to prove the
    graph's edges and state plumbing, not to exercise the Synthesizer's
    own judgment (that's test_synthesizer.py's job)."""
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    def fn(messages, info):
        tool_name = info.output_tools[0].name
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=tool_name,
                    args={
                        "portfolio_assessment": "Diversified but real-estate heavy.",
                        "allocation_notes": [],
                        "existing_holdings_guidance": [],
                        "new_buy_suggestions": [],
                        "no_new_buy_reason": "No candidate suggestions surfaced in this test run.",
                    },
                )
            ]
        )

    return FunctionModel(fn)


async def test_graph_runs_end_to_end_with_real_data_and_persists_checkpoint():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.checkpointer import get_checkpointer
    from agentic_frameworks.graph.graph import build_graph
    from agentic_frameworks.graph.nodes import india_equity_specialist as india_module
    from agentic_frameworks.graph.nodes import mutual_fund_holdings_analyzer as mf_holdings_module
    from agentic_frameworks.graph.nodes import mutual_fund_specialist as mf_specialist_module
    from agentic_frameworks.graph.nodes import synthesizer as synth_module
    from agentic_frameworks.graph.nodes import us_equity_etf_specialist as us_module
    from agentic_frameworks.graph.store import get_memory_store

    def fake_get_specialist_model():
        return TestModel()

    modules = [india_module, mf_holdings_module, mf_specialist_module, us_module]
    originals = [m.get_specialist_model for m in modules]
    for m in modules:
        m.get_specialist_model = fake_get_specialist_model
    original_synth_model = synth_module.get_synthesizer_model
    synth_module.get_synthesizer_model = _empty_synthesis_function_model
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
                "recommendation",
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
            # The Synthesizer's report was produced, with an empty,
            # schema-valid two-sided result (see
            # _empty_synthesis_function_model) - nothing consequential, so
            # the conditional edge routes straight to END, never
            # human_approval_gate.
            assert result["recommendation"]["new_buy_suggestions"] == []
            assert result["recommendation"]["no_new_buy_reason"] is not None
            assert "__interrupt__" not in result

            state_snapshot = await compiled.aget_state(config)
            assert state_snapshot.values
            assert state_snapshot.config["configurable"]["thread_id"] == thread_id
            assert state_snapshot.next == ()
    finally:
        for m, original in zip(modules, originals):
            m.get_specialist_model = original
        synth_module.get_synthesizer_model = original_synth_model


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
    from agentic_frameworks.graph.nodes import synthesizer as synth_module
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

    original_synth_model = synth_module.get_synthesizer_model
    synth_module.get_synthesizer_model = _empty_synthesis_function_model

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
        synth_module.get_synthesizer_model = original_synth_model

    async with aiosqlite.connect(db_path) as conn:
        cursor = await conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in await cursor.fetchall()}
        assert {"checkpoints", "writes", "store", "store_migrations"} <= tables

        row_cursor = await conn.execute("SELECT COUNT(*) FROM store")
        (row_count,) = await row_cursor.fetchone()

    assert row_count > 0


async def test_synthesizer_runs_exactly_once_after_all_four_branches(tmp_path):
    """Design spec Testing item 7: plain add_edge is the fan-in mechanism -
    no reducer needed. Captures real node-call order and the exact state
    keys visible to the Synthesizer when it runs, proving it only ever
    runs once, after all four parallel branches have completed, and sees
    every one of their outputs."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.store.sqlite.aio import AsyncSqliteStore
    from pydantic_ai.models.test import TestModel

    import agentic_frameworks.graph.graph as graph_module
    from agentic_frameworks.graph.nodes import india_equity_specialist as india_module
    from agentic_frameworks.graph.nodes import mutual_fund_holdings_analyzer as mf_holdings_module
    from agentic_frameworks.graph.nodes import mutual_fund_specialist as mf_specialist_module
    from agentic_frameworks.graph.nodes import synthesizer as synth_module
    from agentic_frameworks.graph.nodes import us_equity_etf_specialist as us_module

    def fake_get_specialist_model():
        return TestModel()

    modules = [india_module, mf_holdings_module, mf_specialist_module, us_module]
    originals = [m.get_specialist_model for m in modules]
    for m in modules:
        m.get_specialist_model = fake_get_specialist_model
    original_synth_model = synth_module.get_synthesizer_model
    synth_module.get_synthesizer_model = _empty_synthesis_function_model

    call_order: list[str] = []
    synthesizer_calls: list[dict] = []
    original_synthesizer_node = graph_module.synthesizer_node

    async def counting_synthesizer_node(state, *, store=None):
        synthesizer_calls.append(dict(state))
        return await original_synthesizer_node(state, store=store)

    graph_module.synthesizer_node = counting_synthesizer_node

    db_path = str(tmp_path / "test_fan_in.db")
    try:
        async with (
            AsyncSqliteSaver.from_conn_string(db_path) as checkpointer,
            AsyncSqliteStore.from_conn_string(db_path) as store,
        ):
            await store.setup()
            compiled = graph_module.build_graph().compile(checkpointer=checkpointer, store=store)
            config = {"configurable": {"thread_id": f"test-fan-in-{uuid4()}"}}

            async for step in compiled.astream({}, config=config, stream_mode="updates"):
                call_order.extend(step.keys())
    finally:
        for m, original in zip(modules, originals):
            m.get_specialist_model = original
        synth_module.get_synthesizer_model = original_synth_model
        graph_module.synthesizer_node = original_synthesizer_node

    assert len(synthesizer_calls) == 1
    branch_names = (
        "us_equity_etf_specialist",
        "india_equity_specialist",
        "mutual_fund_holdings_analyzer",
        "mutual_fund_specialist",
    )
    branch_indices = [call_order.index(name) for name in branch_names]
    synthesizer_index = call_order.index("synthesizer")
    assert all(branch_index < synthesizer_index for branch_index in branch_indices)

    synthesizer_input_state = synthesizer_calls[0]
    assert {
        "us_equity_etf_candidates",
        "india_equity_candidates",
        "existing_fund_evaluations",
        "mutual_fund_candidates",
    } <= set(synthesizer_input_state.keys())


async def test_full_graph_pending_approval_reaches_gate_with_persisted_recommendation(tmp_path):
    """Design spec Testing item 8 (first half), on the real, full portfolio
    graph rather than a stub: forcing a Sell on a real, currently-held
    fund routes the run to human_approval_gate instead of END, and the
    recommendation is already persisted in state at the pending point."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.store.sqlite.aio import AsyncSqliteStore
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.graph import build_graph
    from agentic_frameworks.graph.nodes import india_equity_specialist as india_module
    from agentic_frameworks.graph.nodes import mutual_fund_holdings_analyzer as mf_holdings_module
    from agentic_frameworks.graph.nodes import mutual_fund_specialist as mf_specialist_module
    from agentic_frameworks.graph.nodes import synthesizer as synth_module
    from agentic_frameworks.graph.nodes import us_equity_etf_specialist as us_module

    def fake_get_specialist_model():
        return TestModel()

    modules = [india_module, mf_holdings_module, mf_specialist_module, us_module]
    originals = [m.get_specialist_model for m in modules]
    for m in modules:
        m.get_specialist_model = fake_get_specialist_model

    def sell_first_existing_holding_model():
        import re

        def fn(messages, info):
            # Pull the first real ISIN out of the prompt's "[ISIN]" bracket
            # notation (see synthesizer._format_existing_fund_verdicts) -
            # real, run-specific identity, never hardcoded, since the real
            # 9 mutual fund holdings can change over time.
            prompt_text = "\n".join(
                getattr(part, "content", "") or "" for message in messages for part in message.parts
            )
            match = re.search(r"\[(\w[\w.\-]*)\]", prompt_text)
            assert match, "expected at least one [identifier] in the Synthesizer's prompt"
            first_identifier = match.group(1)

            tool_name = info.output_tools[0].name
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=tool_name,
                        args={
                            "portfolio_assessment": "One fund flagged for exit.",
                            "allocation_notes": [],
                            "existing_holdings_guidance": [
                                {
                                    "identifier": first_identifier,
                                    "action": "Sell",
                                    "rationale": "Redundant with another holding.",
                                    "risk_flags": ["redundant"],
                                    "confidence": 0.6,
                                }
                            ],
                            "new_buy_suggestions": [],
                            "no_new_buy_reason": "No Buy-verdict candidates surfaced this test run.",
                        },
                    )
                ]
            )

        return FunctionModel(fn)

    original_synth_model = synth_module.get_synthesizer_model
    synth_module.get_synthesizer_model = sell_first_existing_holding_model

    db_path = str(tmp_path / "test_pending_approval.db")
    try:
        async with (
            AsyncSqliteSaver.from_conn_string(db_path) as checkpointer,
            AsyncSqliteStore.from_conn_string(db_path) as store,
        ):
            await store.setup()
            compiled = build_graph().compile(checkpointer=checkpointer, store=store)
            config = {"configurable": {"thread_id": f"test-pending-approval-{uuid4()}"}}

            result = await compiled.ainvoke({}, config=config)

            assert "__interrupt__" in result

            state_snapshot = await compiled.aget_state(config)
            assert state_snapshot.next == ("human_approval_gate",)
            assert "recommendation" in state_snapshot.values
            assert (
                state_snapshot.values["recommendation"]["existing_holdings_guidance"][0]["action"] == "Sell"
            )
    finally:
        for m, original in zip(modules, originals):
            m.get_specialist_model = original
        synth_module.get_synthesizer_model = original_synth_model
