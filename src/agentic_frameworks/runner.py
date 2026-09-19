"""The one function in this codebase that compiles and invokes the
advisor graph (design spec Scope). Both the Temporal activity
(temporal/activities.py's run_advisor_graph) and any manual run go
through `run_advisor_once` - one code path, two triggers, which is what
makes the OS lock in graph/checkpointer.py / graph/store.py airtight
and what keeps this project from quietly forking its behaviour into "the
scheduled version" and "the manual version" (design spec Architecture 1).

No parameter on `run_advisor_once` can carry a resume value, and
`Command` is never imported here or by anything this module touches -
resuming a pending thread is a different, not-yet-written function that
belongs with Phase 8's own approval contract, not here (design spec
Architecture 2, reason 2; Scope's "Out of scope").
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable
from datetime import UTC, datetime

from agentic_frameworks.graph.checkpointer import get_checkpointer
from agentic_frameworks.graph.graph import build_compiled_graph
from agentic_frameworks.graph.store import get_memory_store
from agentic_frameworks.temporal.shared import AdvisorRunOutcome

_ACTIONS_COUNTED = {"Sell", "Reduce", "AddTo"}


def _extract_interrupt_payload(state_snapshot) -> dict:
    """Pull the interrupt's decision-request payload straight off the
    checkpointer's own state snapshot - the same payload
    human_approval_gate_node built and passed to interrupt() - rather
    than re-deriving it, so there is exactly one place
    (_build_decision_request) that decides what a pending approval looks
    like."""
    for task in state_snapshot.tasks:
        for pending_interrupt in task.interrupts:
            return pending_interrupt.value
    raise ValueError(
        "Graph state is pending at human_approval_gate but carries no interrupt payload - "
        "a malformed/absent interrupt must raise, never be reported as a normal pending state."
    )


def _action_count(recommendation: dict) -> int:
    """How many existing-holdings guidance items are Sell/Reduce/AddTo -
    the same action set that routes a run to human_approval_gate at all
    (see graph/nodes/human_approval_gate.py's _ACTIONS_REQUIRING_APPROVAL).
    Reported only as a count, never a verdict - this module gathers, it
    doesn't judge."""
    return sum(
        1
        for guidance in recommendation.get("existing_holdings_guidance", [])
        if guidance["action"] in _ACTIONS_COUNTED
    )


def _summary(*, status: str, thread_id: str, recommendation: dict, new_buy_count: int, action_count: int) -> str:
    if status == "pending_approval":
        return (
            f"{thread_id}: pending approval - {new_buy_count} new-buy suggestion(s), "
            f"{action_count} existing-holding action(s) awaiting a human decision."
        )
    return (
        f"{thread_id}: completed - {new_buy_count} new-buy suggestion(s), "
        f"{action_count} existing-holding action(s), no approval required."
    )


async def run_advisor_once(
    *, thread_id: str, progress: Callable[[str], None] | None = None
) -> AdvisorRunOutcome:
    """Run exactly one advisor cycle on `thread_id` and classify the
    outcome. Never constructs or accepts a resume value.

    Opens the checkpointer and store together (both take the real OS
    lock - db_lock.py - so a concurrent second opener raises
    AdvisorMemoryLockedError right here, before any MCP subprocess is
    spawned or any token is spent), compiles the real graph
    (build_compiled_graph, Phase 8), and drives it with astream so
    `progress` (activity.heartbeat, when called from the Temporal
    Activity) can be forwarded per node.
    """
    async with get_checkpointer() as checkpointer, get_memory_store() as store:
        compiled = build_compiled_graph(checkpointer=checkpointer, store=store)
        config = {"configurable": {"thread_id": thread_id}}

        # A brand-new thread_id has no checkpoint yet (next == ()) and
        # needs a real input dict to enter at START. A thread_id that
        # already has an incomplete checkpoint - a prior Activity
        # attempt crashed partway (next == (<some pending node>,)) -
        # must be resumed with `None`, which tells LangGraph to
        # continue from the last completed superstep rather than
        # re-running the whole graph from START (live-verified: passing
        # `{}` again would silently start a brand-new run and re-pay for
        # every already-completed node, defeating the whole point of the
        # checkpointer/Temporal seam - design spec Architecture 4). This
        # is never how a pending *approval* gets answered: resuming an
        # interrupt() with `None` (no resume value) just re-pends it
        # at the same interrupt, live-verified to never auto-approve.
        existing_state = await compiled.aget_state(config)
        run_input = None if existing_state.next else {}

        async for chunk in compiled.astream(run_input, config=config, stream_mode="updates"):
            if progress is None:
                continue
            for node_name in chunk:
                if node_name == "__interrupt__":
                    continue
                progress(node_name)

        state_snapshot = await compiled.aget_state(config)
        recommendation = state_snapshot.values.get("recommendation", {})
        new_buy_count = len(recommendation.get("new_buy_suggestions", []))
        action_count = _action_count(recommendation)

        if state_snapshot.next == ("human_approval_gate",):
            status = "pending_approval"
            pending_payload = _extract_interrupt_payload(state_snapshot)
        else:
            status = "completed"
            pending_payload = None

        return AdvisorRunOutcome(
            status=status,
            thread_id=thread_id,
            summary=_summary(
                status=status,
                thread_id=thread_id,
                recommendation=recommendation,
                new_buy_count=new_buy_count,
                action_count=action_count,
            ),
            pending_payload=pending_payload,
            new_buy_count=new_buy_count,
            action_count=action_count,
        )


def _cli() -> None:
    """`python -m agentic_frameworks.runner` - the manual-run entry
    point HANDOFF.md's ad-hoc `python -c` snippets should be replaced
    with, so the documented manual path and the scheduled path are the
    same code (design spec Components)."""
    parser = argparse.ArgumentParser(description="Run one advisor cycle manually, outside Temporal.")
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Defaults to advisor-manual-<today's UTC date>.",
    )
    args = parser.parse_args()
    thread_id = args.thread_id or f"advisor-manual-{datetime.now(UTC).date().isoformat()}"

    outcome = asyncio.run(run_advisor_once(thread_id=thread_id, progress=print))
    print(outcome)


if __name__ == "__main__":
    _cli()
