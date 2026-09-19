"""Regression test for a real bug an independent review caught
(2026-09-19): the SDK's default SandboxedWorkflowRunner fails validating
PortfolioAdvisorDailyWorkflow on this project's venv - a beartype.claw
circular-import bug under Python 3.14, triggered specifically once
activities.py's heavy transitive imports (the whole graph stack) have
already loaded, which is exactly the import order worker.py uses. This
crashed the actual production worker entry point (`python -m
agentic_frameworks.temporal.worker`) even though HANDOFF.md claimed the
fix was already applied there - it was only applied in a test file.

This test constructs the same workflow definition worker.py registers
and prepares it with both runners, without needing a live Temporal
server or client, so it runs every time (no RUN_TEMPORAL_ENV_TESTS gate)
and would catch either regression: the sandboxed runner being reachable
again (an upstream fix, fine) getting silently un-swapped back in, or
worker.py drifting from run_worker's actual construction.
"""

from __future__ import annotations

import inspect

import pytest
from temporalio import workflow
from temporalio.worker import UnsandboxedWorkflowRunner


def test_sandboxed_runner_fails_to_prepare_this_workflow_on_this_venv():
    """Documents *why* the unsandboxed runner is necessary here - if this
    ever stops raising (e.g. an upstream beartype/temporalio fix), it's a
    prompt to reconsider re-enabling the real sandbox, not a failure."""
    from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

    from agentic_frameworks.temporal import activities  # noqa: F401 - heavy imports first, matches worker.py's real order
    from agentic_frameworks.temporal.workflows import PortfolioAdvisorDailyWorkflow

    defn = workflow._Definition.must_from_class(PortfolioAdvisorDailyWorkflow)
    with pytest.raises(Exception):  # noqa: B017 - the point is any prepare-time crash, not a specific type
        SandboxedWorkflowRunner().prepare_workflow(defn)


def test_unsandboxed_runner_prepares_this_workflow_successfully():
    from agentic_frameworks.temporal import activities  # noqa: F401 - heavy imports first, matches worker.py's real order
    from agentic_frameworks.temporal.workflows import PortfolioAdvisorDailyWorkflow

    defn = workflow._Definition.must_from_class(PortfolioAdvisorDailyWorkflow)
    UnsandboxedWorkflowRunner().prepare_workflow(defn)  # must not raise


def test_run_worker_actually_uses_the_unsandboxed_runner():
    """The real regression this review caught: a previous version of
    worker.py never passed workflow_runner= to Worker(...) at all, so the
    SDK default (the sandboxed runner) was used - even though
    UnsandboxedWorkflowRunner was imported and used correctly in test
    fixtures elsewhere. Asserting against run_worker's own source is a
    deliberately blunt check for exactly that class of drift."""
    from agentic_frameworks.temporal import worker as worker_module

    source = inspect.getsource(worker_module.run_worker)
    assert "workflow_runner=UnsandboxedWorkflowRunner()" in source, (
        "run_worker() must construct Worker(...) with "
        "workflow_runner=UnsandboxedWorkflowRunner() - see this test's module "
        "docstring for why the SDK default crashes on this venv."
    )
