"""Create-or-update the real, recurring "portfolio-advisor-daily"
Temporal Schedule.

**Deliberately NOT invoked anywhere in this codebase, on import, or by
test collection.** Design spec Decision 6: the cadence (daily) is
accepted, but the live Schedule itself is not activated by this
implementation - the design spec's own risk section names the first
scheduled run as "this project's first unattended LLM spend" and
recommends one attended run first. That attended run, and the decision
to turn the Schedule on, is Abhi's to make by hand:

    .venv\\Scripts\\python.exe -m agentic_frameworks.temporal.schedule

against a real, already-running Temporal server (dev server or
otherwise) with real `.env` credentials. Nothing in the test suite or
any other module calls `create_daily_schedule` or runs this file's
`__main__` block - it is exercised only via its pure, schedule-object-
building half, `_build_schedule`, which is unit-tested without a server
in tests/temporal/test_schedule.py.
"""

from __future__ import annotations

import asyncio

from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleAlreadyRunningError,
    SchedulePolicy,
    ScheduleOverlapPolicy,
    ScheduleSpec,
    ScheduleUpdate,
    ScheduleUpdateInput,
)

from agentic_frameworks.temporal.worker import DEFAULT_TARGET_HOST, TASK_QUEUE
from agentic_frameworks.temporal.workflows import PortfolioAdvisorDailyWorkflow

SCHEDULE_ID = "portfolio-advisor-daily"
WORKFLOW_ID = "portfolio-advisor-daily-workflow"

# Overnight IST, per design spec Decision 6 ("daily at a fixed overnight
# IST time to start"). 21:30 UTC == 03:00 IST next day.
DAILY_CRON_EXPRESSION_UTC = "30 21 * * *"


def _build_schedule() -> Schedule:
    """The pure half: builds the Schedule object with no network call,
    so its shape is unit-testable without a live Temporal server.

    ScheduleOverlapPolicy.SKIP (design spec Architecture 3, "Why
    Temporal's own mutual exclusion is not sufficient, explicitly") is a
    cheap first layer that avoids even starting a doomed second
    workflow - not the safety mechanism itself, which is the OS lock in
    db_lock.py.
    """
    return Schedule(
        action=ScheduleActionStartWorkflow(
            PortfolioAdvisorDailyWorkflow.run,
            id=WORKFLOW_ID,
            task_queue=TASK_QUEUE,
        ),
        spec=ScheduleSpec(cron_expressions=[DAILY_CRON_EXPRESSION_UTC]),
        policy=SchedulePolicy(overlap=ScheduleOverlapPolicy.SKIP),
    )


async def create_daily_schedule(*, target_host: str = DEFAULT_TARGET_HOST) -> None:
    """Create-or-update the real "portfolio-advisor-daily" Schedule.
    Idempotent - safe to re-run. THE call this whole module exists to
    make available, and the one call this codebase never makes on
    Abhi's behalf."""
    client = await Client.connect(target_host)
    schedule = _build_schedule()
    try:
        await client.create_schedule(SCHEDULE_ID, schedule)
    except ScheduleAlreadyRunningError:
        handle = client.get_schedule_handle(SCHEDULE_ID)

        def _updater(_input: ScheduleUpdateInput) -> ScheduleUpdate:
            return ScheduleUpdate(schedule=schedule)

        await handle.update(_updater)


if __name__ == "__main__":
    asyncio.run(create_daily_schedule())
