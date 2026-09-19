"""Temporal Activities - ordinary Python, no restrictions (design spec
Determinism section): this is where all the I/O, LLM calls, MCP
subprocesses, and SQLite access this package touches actually happens.
`workflows.py` imports none of this directly - only the plain
dataclasses in shared.py cross the workflow/activity boundary.

None of these three Activities has a resume-shaped parameter anywhere
on its signature - see each docstring below and design spec Architecture
2, reason 2.
"""

from __future__ import annotations

from temporalio import activity

from agentic_frameworks.graph.store import get_memory_store
from agentic_frameworks.runner import run_advisor_once
from agentic_frameworks.temporal.shared import AdvisorRunOutcome, AdvisorRunRequest, Notification

PENDING_APPROVALS_NAMESPACE = ("pending_approvals", "advisor")


@activity.defn
async def check_pending_approvals() -> list[str]:
    """Cheap, read-only check of the ("pending_approvals", "advisor")
    Store namespace for prior runs still awaiting a human decision
    (design spec Architecture 1 diagram; Decision 4: skip today's run if
    one is found). No LLM, no MCP, no graph invocation."""
    async with get_memory_store() as store:
        items = await store.asearch(PENDING_APPROVALS_NAMESPACE, limit=1000)
        return [
            item.key
            for item in items
            if item.namespace == PENDING_APPROVALS_NAMESPACE and item.value.get("status") == "pending_approval"
        ]


@activity.defn
async def run_advisor_graph(req: AdvisorRunRequest) -> AdvisorRunOutcome:
    """THE graph run - calls the same run_advisor_once a human calls
    from the command line (design spec Architecture 1). Forwards
    per-node progress to activity.heartbeat, which both surfaces live
    progress in the Temporal UI and lets a dead worker be detected in
    minutes rather than at the full start_to_close_timeout (design spec
    Architecture 4, "Activity timeouts and retries")."""
    return await run_advisor_once(thread_id=req.thread_id, progress=activity.heartbeat)


@activity.defn
async def notify(n: Notification) -> None:
    """Writes/updates the authoritative pending-approvals Store record
    and prints a human-readable console line - the two local-only
    notification surfaces this phase ships (design spec Architecture 5 /
    Decision 2). No email/Telegram/Slack/file log.

    Idempotent on (thread_id, kind): Temporal guarantees at-least-once
    Activity execution, and `aput` on the same (namespace, key) is a
    plain overwrite with the same content, so a duplicate notify can
    never produce a duplicate or conflicting record.
    """
    print(f"[advisor-monitor] {n.kind}: {n.thread_id} - {n.summary}")

    status = "pending_approval" if n.kind == "needs_approval" else n.kind
    async with get_memory_store() as store:
        await store.aput(
            PENDING_APPROVALS_NAMESPACE,
            n.thread_id,
            {
                "status": status,
                "thread_id": n.thread_id,
                "kind": n.kind,
                "summary": n.summary,
                "payload": n.payload,
            },
        )
