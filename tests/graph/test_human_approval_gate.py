"""Tests for the human-approval-gate node: the pure routing predicate,
the fail-loud checkpointer guard, and - the whole point of this phase -
that a real `interrupt()` genuinely suspends the run and can be resumed,
including across a real process boundary.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _report(*, new_buy_suggestions=None, existing_holdings_guidance=None) -> dict:
    return {
        "new_buy_suggestions": new_buy_suggestions or [],
        "existing_holdings_guidance": existing_holdings_guidance or [],
    }


def _new_buy(identifier="XLV") -> dict:
    return {
        "identifier": identifier,
        "rationale": "x",
        "gap_addressed": "Health Care",
        "position_size_pct_of_portfolio": 1.0,
        "entry_approach": "Staggered",
        "suggested_platform": "moomoo",
        "risk_flags": [],
        "confidence": 0.5,
        "identifier_kind": "ticker",
        "display_name": "Health Care Select Sector SPDR Fund",
        "specialist": "us_equity",
    }


def _guidance(action="Hold", identifier="INF179K01WN9") -> dict:
    return {
        "identifier": identifier,
        "action": action,
        "rationale": "x",
        "risk_flags": [],
        "confidence": 0.5,
        "identifier_kind": "isin",
        "display_name": "HDFC BSE Sensex Index Fund - Direct Plan - Growth Option",
    }


# --- _requires_human_approval (pure) ---


def test_requires_approval_true_when_new_buy_suggestions_present():
    from agentic_frameworks.graph.nodes.human_approval_gate import _requires_human_approval

    report = _report(new_buy_suggestions=[_new_buy()])
    assert _requires_human_approval(report) is True


def test_requires_approval_true_on_sell():
    from agentic_frameworks.graph.nodes.human_approval_gate import _requires_human_approval

    report = _report(existing_holdings_guidance=[_guidance(action="Sell")])
    assert _requires_human_approval(report) is True


def test_requires_approval_true_on_reduce():
    from agentic_frameworks.graph.nodes.human_approval_gate import _requires_human_approval

    report = _report(existing_holdings_guidance=[_guidance(action="Reduce")])
    assert _requires_human_approval(report) is True


def test_requires_approval_true_on_add_to():
    """Design spec Decision 2: AddTo commits fresh capital, so it triggers
    approval too, not just Sell/Reduce."""
    from agentic_frameworks.graph.nodes.human_approval_gate import _requires_human_approval

    report = _report(existing_holdings_guidance=[_guidance(action="AddTo")])
    assert _requires_human_approval(report) is True


def test_requires_approval_false_when_hold_only():
    from agentic_frameworks.graph.nodes.human_approval_gate import _requires_human_approval

    report = _report(existing_holdings_guidance=[_guidance(action="Hold")])
    assert _requires_human_approval(report) is False


def test_requires_approval_false_when_empty():
    from agentic_frameworks.graph.nodes.human_approval_gate import _requires_human_approval

    report = _report()
    assert _requires_human_approval(report) is False


# --- fail-loud checkpointer guard ---


async def test_gate_node_raises_without_checkpointer():
    from agentic_frameworks.graph.nodes.human_approval_gate import human_approval_gate_node

    state = {"recommendation": _report(existing_holdings_guidance=[_guidance(action="Sell")])}
    config = {"configurable": {}}  # no __pregel_checkpointer key at all

    with pytest.raises(ValueError, match="requires the graph to be compiled with a real checkpointer"):
        await human_approval_gate_node(state, config)


async def test_gate_node_raises_when_checkpointer_explicitly_none():
    from agentic_frameworks.graph.nodes.human_approval_gate import human_approval_gate_node

    state = {"recommendation": _report(existing_holdings_guidance=[_guidance(action="Sell")])}
    config = {"configurable": {"__pregel_checkpointer": None}}

    with pytest.raises(ValueError, match="requires the graph to be compiled with a real checkpointer"):
        await human_approval_gate_node(state, config)


# --- genuinely pending + same-process resume (design spec Testing item 8) ---


async def test_gate_genuinely_pends_and_resumes_without_rerunning_upstream_node(tmp_path):
    """A minimal, real two-node graph (stub producer -> the REAL
    human_approval_gate_node), with a real AsyncSqliteSaver checkpointer:
    ainvoke returns __interrupt__, aget_state().next ==
    ("human_approval_gate",), the recommendation is already persisted, and
    resuming with Command(resume=...) leaves the producer's call count at
    1 while the gate's goes to 2 - proving resume re-runs only the
    interrupted node (design spec Background, "Resume re-runs only the
    interrupted node, not the upstream LLM node").
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command

    from agentic_frameworks.graph.nodes.human_approval_gate import human_approval_gate_node
    from agentic_frameworks.graph.schemas import HumanApproval
    from agentic_frameworks.graph.state import GraphState
    from tests.graph._cross_process_resume_helper import _stub_recommendation

    call_counts = {"produce": 0, "gate": 0}

    async def counting_producer(state):
        call_counts["produce"] += 1
        return {"recommendation": _stub_recommendation()}

    async def counting_gate(state, config):
        call_counts["gate"] += 1
        return await human_approval_gate_node(state, config)

    graph = StateGraph(GraphState)
    graph.add_node("produce", counting_producer)
    graph.add_node("human_approval_gate", counting_gate)
    graph.add_edge(START, "produce")
    graph.add_edge("produce", "human_approval_gate")
    graph.add_edge("human_approval_gate", END)

    db_path = str(tmp_path / "test_resume.db")
    thread_id = f"test-resume-{uuid4()}"
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        compiled = graph.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        result = await compiled.ainvoke({}, config=config)
        assert "__interrupt__" in result

        state_snapshot = await compiled.aget_state(config)
        assert state_snapshot.next == ("human_approval_gate",)
        assert "recommendation" in state_snapshot.values

        assert call_counts == {"produce": 1, "gate": 1}

        resume_payload = HumanApproval(
            decision="approved",
            approved_by="abhi",
            excluded_identifiers=[],
            note=None,
            decided_at="2026-09-19T00:00:00+00:00",
        ).model_dump()
        resumed_result = await compiled.ainvoke(Command(resume=resume_payload), config=config)

        assert resumed_result["approval"]["decision"] == "approved"
        assert call_counts["produce"] == 1
        assert call_counts["gate"] == 2

        final_state = await compiled.aget_state(config)
        assert final_state.next == ()


# --- genuinely pending across a real process boundary (design spec Testing item 8) ---


async def test_gate_resume_survives_a_real_process_boundary(tmp_path):
    """The single most load-bearing test in this phase: a second, separate
    OS process opens the same checkpoint file and resumes the pending
    interrupt, proving the pause is a real, durable pending state - not
    an in-memory illusion that only looks resumable within one process.
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from tests.graph._cross_process_resume_helper import build_mini_graph

    db_path = str(tmp_path / "test_cross_process_resume.db")
    thread_id = f"test-cross-process-{uuid4()}"

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        compiled = build_mini_graph().compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        result = await compiled.ainvoke({}, config=config)
        assert "__interrupt__" in result

    # Process 1's checkpointer connection is closed at this point (the
    # `async with` block above exited) - process 2 below opens the same
    # SQLite file completely fresh, from a different OS process.
    proc = subprocess.run(
        [sys.executable, "-m", "tests.graph._cross_process_resume_helper", db_path, thread_id],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr

    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["next_before"] == ["human_approval_gate"]
    assert payload["recommendation_present_before"] is True
    assert payload["approval"]["decision"] == "approved"
    assert payload["approval"]["approved_by"] == "abhi"
    assert payload["next_after"] == []
