"""Design spec Testing item 4 - reinforces the no-self-approval
guarantee specifically for this phase's new code. Phase 8's existing
test_no_self_approval.py::test_no_module_under_src_references_command
already covers src/agentic_frameworks/temporal/ and runner.py
automatically (it walks every .py file under src/agentic_frameworks/,
and this phase's files live there) - these tests add the two checks the
design spec calls out that grep alone can't express: no resume-shaped
parameter on run_advisor_once, and no @workflow.signal anywhere on
PortfolioAdvisorDailyWorkflow (so nothing in this design can answer the
approval gate, from the Temporal UI or otherwise).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "agentic_frameworks"


def test_temporal_package_and_runner_are_covered_by_the_command_grep():
    """Confirms coverage, doesn't re-implement it - the actual assertion
    lives in test_no_self_approval.py and must not be weakened or
    special-cased."""
    temporal_files = list((SRC_ROOT / "temporal").rglob("*.py"))
    assert temporal_files, "expected the temporal package to exist and contain files"
    assert (SRC_ROOT / "runner.py").exists()

    for path in [*temporal_files, SRC_ROOT / "runner.py"]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            assert not (isinstance(node, ast.ImportFrom) and any(a.name == "Command" for a in node.names)), path
            assert not (isinstance(node, ast.Name) and node.id == "Command"), path
            assert not (isinstance(node, ast.Attribute) and node.attr == "Command"), path


def test_run_advisor_once_has_no_resume_shaped_parameter():
    from agentic_frameworks.runner import run_advisor_once

    signature = inspect.signature(run_advisor_once)
    resume_shaped_names = {"resume", "resume_value", "command", "decision", "approval"}
    assert set(signature.parameters) & resume_shaped_names == set()
    assert set(signature.parameters) == {"thread_id", "progress"}


def test_activities_have_no_resume_shaped_parameter():
    from agentic_frameworks.temporal import activities

    for fn in (activities.check_pending_approvals, activities.run_advisor_graph, activities.notify):
        signature = inspect.signature(fn)
        resume_shaped_names = {"resume", "resume_value", "command", "decision", "approval"}
        assert set(signature.parameters) & resume_shaped_names == set()


def test_workflow_defines_no_signal_or_query():
    """A human cannot approve a pending run from the Temporal UI any
    more than from inside this codebase - there is nothing here to send
    a decision to (design spec Architecture 2, reason 3)."""
    from agentic_frameworks.temporal.workflows import PortfolioAdvisorDailyWorkflow

    for _name, member in inspect.getmembers(PortfolioAdvisorDailyWorkflow):
        assert not hasattr(member, "__temporal_signal_definition"), _name
        assert not hasattr(member, "__temporal_query_definition"), _name
