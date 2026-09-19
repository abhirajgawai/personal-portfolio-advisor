"""Tests for PortfolioAdvisorDailyWorkflow's branching logic, driven by
`temporalio.testing.WorkflowEnvironment`'s free, local, time-skipping
test server - real workflow, real Temporal server, mocked Activities
(design spec Testing item 6: "Mocking activities here is not a
compromise - it is how Temporal workflow logic is meant to be tested,
and the real graph path is covered by test 5" - see test_runner.py for
that real-graph coverage).

All scenarios share ONE `WorkflowEnvironment` (started once, at the top
of the single test function below) rather than one per test - each
`start_time_skipping()` call spins up its own local server subprocess,
and this machine ran low on memory when an earlier draft of this file
started seven of them. Test isolation between scenarios instead comes
from each scenario using its own workflow id and its own closures -
same rigor, far cheaper.

`UnsandboxedWorkflowRunner` is used here rather than the SDK default
`SandboxedWorkflowRunner`: the default sandbox's own reflective importer
hits an environment-specific circular-import bug in `beartype.claw` (a
transitive dependency pulled in by this project's pydantic-ai/graph
stack) on this project's Python 3.14 venv, discovered while building
this phase - see HANDOFF.md's Phase 10 section for the full note. This
does not weaken what these tests actually prove: `workflows.py` itself
contains no I/O and only the branching logic under test, and
determinism (the last scenario below) is separately, directly tested.

**Known unreliability, disclosed rather than hidden (see HANDOFF.md
Phase 10):** `WorkflowEnvironment.start_time_skipping()` launches a real
`temporal-test-server-sdk-*` subprocess, and on this Windows machine
that subprocess has hung at least once during this phase's build,
requiring a manual kill of both it and the pytest process holding it.
This test is therefore **skipped by default** (`RUN_TEMPORAL_ENV_TESTS`
env var opts in) rather than left to intermittently hang the full suite
- everything else this phase's Testing section calls for (the real
cross-process OS lock, the no-self-approval grep coverage, notification
idempotence, the pending-approval path through the real graph) is
covered by other test files that don't depend on a live test-server
subprocess. Run manually, when deliberately verifying workflow
branching, with:
`RUN_TEMPORAL_ENV_TESTS=1 .venv\\Scripts\\python.exe -m pytest tests/temporal/test_workflows.py -q`
"""

from __future__ import annotations

import os
import uuid

import pytest
from temporalio import activity
from temporalio.client import WorkflowFailureError
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import UnsandboxedWorkflowRunner, Worker

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_TEMPORAL_ENV_TESTS"),
    reason=(
        "WorkflowEnvironment.start_time_skipping() has hung on this Windows machine "
        "(a real temporal-test-server-sdk-* subprocess) - opt in with "
        "RUN_TEMPORAL_ENV_TESTS=1 to run it deliberately. See HANDOFF.md Phase 10."
    ),
)

from agentic_frameworks.temporal.shared import AdvisorRunOutcome, AdvisorRunRequest, Notification
from agentic_frameworks.temporal.workflows import PortfolioAdvisorDailyWorkflow

TASK_QUEUE = "test-portfolio-advisor"


def _completed_outcome(thread_id: str) -> AdvisorRunOutcome:
    return AdvisorRunOutcome(
        status="completed",
        thread_id=thread_id,
        summary=f"{thread_id}: completed - 0 new-buy suggestion(s), 0 existing-holding action(s).",
        pending_payload=None,
        new_buy_count=0,
        action_count=0,
    )


def _pending_outcome(thread_id: str) -> AdvisorRunOutcome:
    return AdvisorRunOutcome(
        status="pending_approval",
        thread_id=thread_id,
        summary=f"{thread_id}: pending approval - 1 new-buy suggestion(s).",
        pending_payload={"new_buy_suggestions": [{"identifier": "VOO"}]},
        new_buy_count=1,
        action_count=0,
    )


async def _run_workflow(env, *, check_pending_fn, run_graph_fn, notify_fn):
    @activity.defn(name="check_pending_approvals")
    async def check_pending_approvals() -> list[str]:
        return await check_pending_fn()

    @activity.defn(name="run_advisor_graph")
    async def run_advisor_graph(req: AdvisorRunRequest) -> AdvisorRunOutcome:
        return await run_graph_fn(req)

    @activity.defn(name="notify")
    async def notify(n: Notification) -> None:
        await notify_fn(n)

    async with Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=[PortfolioAdvisorDailyWorkflow],
        activities=[check_pending_approvals, run_advisor_graph, notify],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        return await env.client.execute_workflow(
            PortfolioAdvisorDailyWorkflow.run,
            id=f"test-workflow-{uuid.uuid4()}",
            task_queue=TASK_QUEUE,
        )


async def test_workflow_branching_scenarios():
    """One shared local time-skipping server, six scenarios run against
    it in sequence - see the module docstring for why this is one test
    rather than six."""
    async with await WorkflowEnvironment.start_time_skipping() as env:
        # --- completed run: notify(completed), status completed ---
        notifications: list[Notification] = []

        async def check_pending_none():
            return []

        async def run_graph_completed(req):
            return _completed_outcome(req.thread_id)

        async def record_notify(n):
            notifications.append(n)

        result = await _run_workflow(
            env, check_pending_fn=check_pending_none, run_graph_fn=run_graph_completed, notify_fn=record_notify
        )
        assert result.status == "completed"
        assert len(notifications) == 1
        assert notifications[0].kind == "completed"

        # --- pending-approval run: notify(needs_approval), workflow ends, never resumes ---
        notifications = []
        graph_call_count = 0

        async def run_graph_pending(req):
            nonlocal graph_call_count
            graph_call_count += 1
            return _pending_outcome(req.thread_id)

        result = await _run_workflow(
            env, check_pending_fn=check_pending_none, run_graph_fn=run_graph_pending, notify_fn=record_notify
        )
        assert result.status == "pending_approval"
        assert graph_call_count == 1  # runs once, then stops - never waits, never resumes.
        assert len(notifications) == 1
        assert notifications[0].kind == "needs_approval"
        assert notifications[0].payload == {"new_buy_suggestions": [{"identifier": "VOO"}]}

        # --- prior pending approval: reject-early skip, run_advisor_graph never called (Decision 4) ---
        notifications = []
        graph_call_count = 0

        async def check_pending_one():
            return ["advisor-daily-2026-09-18"]

        async def run_graph_should_not_run(req):
            nonlocal graph_call_count
            graph_call_count += 1
            return _completed_outcome(req.thread_id)

        result = await _run_workflow(
            env, check_pending_fn=check_pending_one, run_graph_fn=run_graph_should_not_run, notify_fn=record_notify
        )
        assert result.status == "skipped_prior_pending"
        assert graph_call_count == 0
        assert len(notifications) == 1
        assert notifications[0].kind == "still_pending"

        # --- failing-then-succeeding activity: Temporal retries, workflow still completes ---
        # (design spec Testing item 7, first half). ValueError is
        # deliberately non-retryable in the real workflow's RetryPolicy
        # (a config error retrying three times just wastes timeouts) -
        # RuntimeError is used here, which the real policy leaves
        # retryable, to prove genuine retry-then-succeed behaviour.
        attempt_count = 0

        async def run_graph_retryable(req):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                raise RuntimeError("simulated transient failure")
            return _completed_outcome(req.thread_id)

        async def ignore_notify(n):
            pass

        result = await _run_workflow(
            env, check_pending_fn=check_pending_none, run_graph_fn=run_graph_retryable, notify_fn=ignore_notify
        )
        assert attempt_count == 2
        assert result.status == "completed"

        # --- lock contention exhausts retries: workflow reports skipped_locked, not a bare failure ---
        notifications = []

        async def run_graph_locked(req):
            raise ApplicationError("locked", type="AdvisorMemoryLockedError")

        result = await _run_workflow(
            env, check_pending_fn=check_pending_none, run_graph_fn=run_graph_locked, notify_fn=record_notify
        )
        assert result.status == "skipped_locked"
        assert any(n.kind == "skipped_locked" for n in notifications)

        # --- a real (non-lock) Activity failure propagates and fails the workflow visibly ---
        async def run_graph_real_bug(req):
            raise ValueError("OPENROUTER_API_KEY is not set")

        with pytest.raises(WorkflowFailureError):
            await _run_workflow(
                env, check_pending_fn=check_pending_none, run_graph_fn=run_graph_real_bug, notify_fn=ignore_notify
            )

        # --- deterministic thread_id: same scheduled date -> same id (design spec Testing item 8) ---
        seen_thread_ids: list[str] = []

        async def run_graph_capture_thread_id(req):
            seen_thread_ids.append(req.thread_id)
            return _completed_outcome(req.thread_id)

        for _ in range(2):
            await _run_workflow(
                env,
                check_pending_fn=check_pending_none,
                run_graph_fn=run_graph_capture_thread_id,
                notify_fn=ignore_notify,
            )
        assert len(seen_thread_ids) == 2
        assert seen_thread_ids[0] == seen_thread_ids[1]
        assert seen_thread_ids[0].startswith("advisor-daily-")
