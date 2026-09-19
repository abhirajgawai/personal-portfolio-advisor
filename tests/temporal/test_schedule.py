"""schedule.py's own module docstring and `_build_schedule`'s docstring
both claim the pure, schedule-object-building half is unit-tested without
a live server - an independent review found that claim was never actually
backed by a test. This is that test. It must never import or exercise
`create_daily_schedule` or the module's `__main__` block, since those
require a live Temporal server and this project's schedule is
deliberately not activated yet (design spec Decision 6).
"""

from __future__ import annotations

from temporalio.client import ScheduleActionStartWorkflow, ScheduleOverlapPolicy

from agentic_frameworks.temporal.schedule import (
    DAILY_CRON_EXPRESSION_UTC,
    TASK_QUEUE,
    WORKFLOW_ID,
    _build_schedule,
)
from agentic_frameworks.temporal.workflows import PortfolioAdvisorDailyWorkflow


def test_build_schedule_is_pure_and_makes_no_network_call():
    """No Client, no server, no await - if this needed either, it
    wouldn't be importable/callable in a plain sync test."""
    schedule = _build_schedule()

    assert isinstance(schedule.action, ScheduleActionStartWorkflow)
    # The SDK resolves this to the workflow's registered name (a string),
    # not the function object itself.
    assert schedule.action.workflow == PortfolioAdvisorDailyWorkflow.__temporal_workflow_definition.name
    assert schedule.action.id == WORKFLOW_ID
    assert schedule.action.task_queue == TASK_QUEUE
    assert schedule.spec.cron_expressions == [DAILY_CRON_EXPRESSION_UTC]
    # SKIP, not the Temporal-only dedup this project's own design spec
    # says is insufficient on its own (db_lock.py's OS lock is the real
    # safety mechanism) - just a cheap first layer.
    assert schedule.policy.overlap == ScheduleOverlapPolicy.SKIP
