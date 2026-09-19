"""Tests for runner.py - the one function that compiles and invokes the
advisor graph (design spec Scope, Testing items 5 and 7 second half).

Like test_graph.py's own real-graph tests, these hit the REAL
sandbox-data/agent_memory.db (get_checkpointer()/get_memory_store() are
not parameterizable - that's the whole point of there being exactly one
agent_memory.db) - each test uses a fresh uuid4-based thread_id, the
same convention every other real-path test in this suite already
follows, so runs never collide.
"""

from __future__ import annotations

from uuid import uuid4

from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel


def _patch_specialist_models(monkeypatch):
    from agentic_frameworks.graph.nodes import india_equity_specialist as india_module
    from agentic_frameworks.graph.nodes import mutual_fund_holdings_analyzer as mf_holdings_module
    from agentic_frameworks.graph.nodes import mutual_fund_specialist as mf_specialist_module
    from agentic_frameworks.graph.nodes import us_equity_etf_specialist as us_module

    def fake_get_specialist_model():
        return TestModel()

    for module in (india_module, mf_holdings_module, mf_specialist_module, us_module):
        monkeypatch.setattr(module, "get_specialist_model", fake_get_specialist_model)


def _sell_first_existing_holding_model():
    """Same FunctionModel pattern as test_graph.py's
    test_full_graph_pending_approval_reaches_gate_with_persisted_recommendation
    - pulls a real, run-specific ISIN out of the Synthesizer's own
    prompt so this never depends on a hardcoded fund identifier."""
    import re

    def fn(messages, info):
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


def _empty_synthesis_function_model():
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


async def test_run_advisor_once_pending_approval_end_to_end_never_resumes(monkeypatch):
    """Design spec Testing item 5 - the proof that the runner observed
    the gate without answering it: forces a Sell on a real, currently-
    held fund via run_advisor_once (not the compiled graph directly, so
    this exercises the exact function the Temporal Activity calls),
    asserts status == "pending_approval", a JSON-serializable
    pending_payload, and - critically - that the thread is STILL pending
    afterwards (no Command, no resume, anywhere in this call path)."""
    import json

    from agentic_frameworks.graph.checkpointer import get_checkpointer
    from agentic_frameworks.graph.graph import build_compiled_graph
    from agentic_frameworks.graph.nodes import synthesizer as synth_module
    from agentic_frameworks.graph.store import get_memory_store
    from agentic_frameworks.runner import run_advisor_once

    _patch_specialist_models(monkeypatch)
    monkeypatch.setattr(synth_module, "get_synthesizer_model", _sell_first_existing_holding_model)

    thread_id = f"test-runner-pending-{uuid4()}"
    outcome = await run_advisor_once(thread_id=thread_id)

    assert outcome.status == "pending_approval"
    assert outcome.thread_id == thread_id
    assert outcome.action_count == 1
    assert outcome.new_buy_count == 0
    assert outcome.pending_payload is not None
    json.dumps(outcome.pending_payload)  # must be JSON-serializable - it crosses a Temporal Activity boundary.
    assert outcome.pending_payload["consequential_holdings_guidance"][0]["action"] == "Sell"

    # The thread is still, genuinely pending - run_advisor_once observed
    # the gate but never answered it.
    async with get_checkpointer() as checkpointer, get_memory_store() as store:
        compiled = build_compiled_graph(checkpointer=checkpointer, store=store)
        state_snapshot = await compiled.aget_state({"configurable": {"thread_id": thread_id}})
        assert state_snapshot.next == ("human_approval_gate",)


async def test_run_advisor_once_completed_path_reports_zero_pending_payload(monkeypatch):
    from agentic_frameworks.graph.nodes import synthesizer as synth_module
    from agentic_frameworks.runner import run_advisor_once

    _patch_specialist_models(monkeypatch)
    monkeypatch.setattr(synth_module, "get_synthesizer_model", _empty_synthesis_function_model)

    thread_id = f"test-runner-completed-{uuid4()}"
    outcome = await run_advisor_once(thread_id=thread_id)

    assert outcome.status == "completed"
    assert outcome.pending_payload is None
    assert outcome.new_buy_count == 0
    assert outcome.action_count == 0


async def test_run_advisor_once_resumes_after_a_crash_without_rerunning_completed_nodes(monkeypatch):
    """Design spec Testing item 7 (second half) - the test that proves
    the two durability layers actually compose: a real-graph run,
    interrupted mid-way (risk_analyst raises on its first call, after
    research already completed and checkpointed), then re-invoked with
    the SAME thread_id - research must not run a second time."""
    import agentic_frameworks.graph.graph as graph_module
    from agentic_frameworks.graph.nodes import synthesizer as synth_module
    from agentic_frameworks.runner import run_advisor_once

    _patch_specialist_models(monkeypatch)
    monkeypatch.setattr(synth_module, "get_synthesizer_model", _empty_synthesis_function_model)

    call_counts = {"research": 0, "risk_analyst": 0}
    original_research_node = graph_module.research_node
    original_risk_analyst_node = graph_module.risk_analyst_node

    async def counting_research_node(state):
        call_counts["research"] += 1
        return await original_research_node(state)

    def failing_once_risk_analyst_node(state):
        call_counts["risk_analyst"] += 1
        if call_counts["risk_analyst"] == 1:
            raise RuntimeError("simulated mid-run crash")
        return original_risk_analyst_node(state)

    monkeypatch.setattr(graph_module, "research_node", counting_research_node)
    monkeypatch.setattr(graph_module, "risk_analyst_node", failing_once_risk_analyst_node)

    thread_id = f"test-runner-resume-{uuid4()}"

    try:
        await run_advisor_once(thread_id=thread_id)
        raise AssertionError("expected the simulated crash to propagate")
    except RuntimeError as exc:
        assert "simulated mid-run crash" in str(exc)

    assert call_counts == {"research": 1, "risk_analyst": 1}

    outcome = await run_advisor_once(thread_id=thread_id)

    assert outcome.status == "completed"
    # research did NOT run a second time - the checkpointer resumed
    # after it; risk_analyst's second call is the one that succeeds.
    assert call_counts["research"] == 1
    assert call_counts["risk_analyst"] == 2
