"""The worker process entry point - connects to a local Temporal server,
registers the one Workflow and three Activities on task queue
"portfolio-advisor", and runs until interrupted.

Ordinary Python (not workflow-sandboxed) - importing activities.py here
is exactly where the heavy imports (the whole graph stack) are meant to
land (design spec Components).

Manual step, per Local Setup: `python -m agentic_frameworks.temporal.worker`,
after a dev server is running on localhost:7233 (either the `temporal`
CLI's `server start-dev`, or scripts/temporal_dev_server.py's
WorkflowEnvironment.start_local(), per the design spec).
"""

from __future__ import annotations

import asyncio
import logging

from temporalio.client import Client
from temporalio.worker import UnsandboxedWorkflowRunner, Worker

from agentic_frameworks.temporal import activities
from agentic_frameworks.temporal.workflows import PortfolioAdvisorDailyWorkflow

TASK_QUEUE = "portfolio-advisor"
DEFAULT_TARGET_HOST = "localhost:7233"

logger = logging.getLogger(__name__)


async def run_worker(*, target_host: str = DEFAULT_TARGET_HOST) -> None:
    client = await Client.connect(target_host)
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[PortfolioAdvisorDailyWorkflow],
        activities=[
            activities.check_pending_approvals,
            activities.run_advisor_graph,
            activities.notify,
        ],
        # The SDK's default SandboxedWorkflowRunner fails validating this
        # workflow on this exact venv - a beartype.claw circular-import bug
        # under Python 3.14, triggered specifically by activities.py's heavy
        # transitive imports (the whole graph stack) being loaded before the
        # sandbox tries to prepare the workflow module (live-reproduced: the
        # same prepare_workflow() call succeeds if that import hasn't
        # happened yet). Safe to skip here because workflows.py has zero I/O
        # and uses workflow.now() (never datetime.now()), so there's no
        # non-determinism the sandbox would have caught for THIS workflow -
        # but re-check this on any temporalio/Python version bump, since the
        # sandbox is real protection this project is otherwise giving up.
        workflow_runner=UnsandboxedWorkflowRunner(),
    )
    logger.info("Portfolio advisor worker connected to %s, task queue %s", target_host, TASK_QUEUE)
    await worker.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_worker())
