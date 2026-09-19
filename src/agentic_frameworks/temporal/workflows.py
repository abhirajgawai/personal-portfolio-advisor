"""PortfolioAdvisorDailyWorkflow - deterministic orchestration only, no
I/O, no LLM, no DB access, no `Command` (design spec Determinism
section, Architecture 1/2).

Temporal workflow code is replayed from history and must be
deterministic, and the SDK additionally runs it inside an import
sandbox - so this module imports nothing heavy: only `temporalio`,
`datetime`, and shared.py's plain dataclasses (via
`workflow.unsafe.imports_passed_through()`, the standard pattern for
letting a lightweight, pure-dataclass module through the sandbox
unchanged).

No `@workflow.signal`, no `@workflow.query` that could carry a
decision, anywhere in this class - a human cannot approve a pending run
from the Temporal UI any more than from inside this codebase, because
there is nothing here to send a decision *to* (design spec Architecture
2, reason 3).
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError

with workflow.unsafe.imports_passed_through():
    from agentic_frameworks.temporal.shared import (
        AdvisorRunOutcome,
        AdvisorRunRequest,
        DailyRunResult,
        Notification,
    )

RUN_ACTIVITY_START_TO_CLOSE_TIMEOUT = timedelta(minutes=45)
RUN_ACTIVITY_HEARTBEAT_TIMEOUT = timedelta(minutes=2)
CHEAP_ACTIVITY_TIMEOUT = timedelta(seconds=30)


@workflow.defn
class PortfolioAdvisorDailyWorkflow:
    """One `@workflow.run`. Picks today's deterministic thread_id,
    checks for a still-pending prior approval (reject-early, no LLM
    spend if one exists - design spec Decision 4), runs the graph via
    one coarse Activity, branches on the outcome, notifies, and
    finishes. It never waits, never polls, never resumes anything -
    the workflow's terminal action on every path is "write a record and
    stop" (design spec Architecture 2, reason 4)."""

    @workflow.run
    async def run(self) -> DailyRunResult:
        thread_id = f"advisor-daily-{workflow.now().date().isoformat()}"

        pending = await workflow.execute_activity(
            "check_pending_approvals",
            start_to_close_timeout=CHEAP_ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=3),
        )
        if pending:
            summary = f"{len(pending)} prior run(s) still awaiting approval: {', '.join(pending)}."
            await workflow.execute_activity(
                "notify",
                Notification(kind="still_pending", thread_id=thread_id, summary=summary, payload=None),
                start_to_close_timeout=CHEAP_ACTIVITY_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            return DailyRunResult(status="skipped_prior_pending", thread_id=thread_id, summary=summary)

        try:
            outcome: AdvisorRunOutcome = await workflow.execute_activity(
                "run_advisor_graph",
                AdvisorRunRequest(thread_id=thread_id),
                start_to_close_timeout=RUN_ACTIVITY_START_TO_CLOSE_TIMEOUT,
                heartbeat_timeout=RUN_ACTIVITY_HEARTBEAT_TIMEOUT,
                retry_policy=RetryPolicy(
                    maximum_attempts=3,
                    non_retryable_error_types=["ValueError", "ValidationError"],
                ),
            )
        except ActivityError as exc:
            # AdvisorMemoryLockedError (db_lock.py) is the one Activity
            # failure this workflow classifies rather than re-raising -
            # an overlapping manual run that never released the lock
            # within the whole retry budget. Anything else (a missing
            # OPENROUTER_API_KEY, an unreachable MCP server, a real bug
            # in a node) propagates and fails the workflow visibly in
            # the Temporal UI, per the fail-loud standing rule.
            cause = exc.cause
            if not (isinstance(cause, ApplicationError) and cause.type == "AdvisorMemoryLockedError"):
                raise
            summary = (
                f"{thread_id}: skipped - agent_memory.db was locked by another run "
                "for the whole retry budget."
            )
            await workflow.execute_activity(
                "notify",
                Notification(kind="skipped_locked", thread_id=thread_id, summary=summary, payload=None),
                start_to_close_timeout=CHEAP_ACTIVITY_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            return DailyRunResult(status="skipped_locked", thread_id=thread_id, summary=summary)

        if outcome.status == "pending_approval":
            await workflow.execute_activity(
                "notify",
                Notification(
                    kind="needs_approval",
                    thread_id=thread_id,
                    summary=outcome.summary,
                    payload=outcome.pending_payload,
                ),
                start_to_close_timeout=CHEAP_ACTIVITY_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            return DailyRunResult(status="pending_approval", thread_id=thread_id, summary=outcome.summary)

        await workflow.execute_activity(
            "notify",
            Notification(kind="completed", thread_id=thread_id, summary=outcome.summary, payload=None),
            start_to_close_timeout=CHEAP_ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=3),
        )
        return DailyRunResult(status="completed", thread_id=thread_id, summary=outcome.summary)
