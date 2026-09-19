"""Stdlib dataclasses crossing the Temporal workflow/activity boundary.

Dataclass-only, no pydantic - safe to import into the workflow's import
sandbox (see workflows.py's `workflow.unsafe.imports_passed_through()`
block) and serializable by Temporal's *default* JSON data converter, so
no `pydantic` extra and no custom data converter are needed (design
spec Components).

None of these carry a resume-shaped field anywhere - see workflows.py
and activities.py docstrings for why that's load-bearing, not
incidental.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdvisorRunRequest:
    """What the workflow passes into the run_advisor_graph Activity."""

    thread_id: str


@dataclass
class AdvisorRunOutcome:
    """What run_advisor_once (and, one level up, run_advisor_graph)
    returns - a plain, JSON-serializable classification of one advisor
    cycle, never anything resume-shaped.

    status: "completed" | "pending_approval". run_advisor_once itself
        never returns "failed_lock" - a lock that can't be acquired
        raises AdvisorMemoryLockedError instead (db_lock.py), which
        surfaces as a retryable Activity failure; "failed_lock" is the
        DailyRunResult-level status the *workflow* reports once that
        retry budget is exhausted (see workflows.py), named here too so
        the whole status vocabulary lives in one place.
    """

    status: str
    thread_id: str
    summary: str
    pending_payload: dict | None
    new_buy_count: int
    action_count: int


@dataclass
class Notification:
    """What the workflow passes into the notify Activity."""

    kind: str
    thread_id: str
    summary: str
    payload: dict | None = None


@dataclass
class DailyRunResult:
    """The workflow's own return value - what shows up in the Temporal
    UI and the Schedule's recent-actions list."""

    status: str
    thread_id: str
    summary: str
