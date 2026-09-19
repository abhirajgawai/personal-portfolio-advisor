"""Tests for the three Temporal Activities, run as plain async functions
(no Temporal server needed - these are ordinary async Python, per design
spec Determinism section) against a real temp-file AsyncSqliteStore.
"""

from __future__ import annotations

from uuid import uuid4


async def test_check_pending_approvals_returns_only_pending_status_threads(monkeypatch):
    from langgraph.store.sqlite.aio import AsyncSqliteStore

    from agentic_frameworks.temporal import activities

    # Point the activity's get_memory_store import at a real temp-file
    # store, so this test never touches sandbox-data/agent_memory.db.
    import tempfile
    from contextlib import asynccontextmanager
    from pathlib import Path

    tmp_dir = tempfile.mkdtemp()
    db_path = Path(tmp_dir) / "test_activities.db"

    @asynccontextmanager
    async def _tmp_store():
        async with AsyncSqliteStore.from_conn_string(str(db_path)) as store:
            await store.setup()
            yield store

    monkeypatch.setattr(activities, "get_memory_store", _tmp_store)

    async with _tmp_store() as store:
        await store.aput(
            activities.PENDING_APPROVALS_NAMESPACE,
            "advisor-daily-2026-09-18",
            {"status": "pending_approval", "thread_id": "advisor-daily-2026-09-18"},
        )
        await store.aput(
            activities.PENDING_APPROVALS_NAMESPACE,
            "advisor-daily-2026-09-17",
            {"status": "completed", "thread_id": "advisor-daily-2026-09-17"},
        )

    result = await activities.check_pending_approvals()

    assert result == ["advisor-daily-2026-09-18"]


async def test_notify_is_idempotent_on_thread_id_and_kind(monkeypatch, tmp_path):
    """Design spec Testing item 9: running notify twice with the same
    (thread_id, kind) leaves exactly one record - Temporal guarantees
    at-least-once Activity execution."""
    from contextlib import asynccontextmanager

    from langgraph.store.sqlite.aio import AsyncSqliteStore

    from agentic_frameworks.temporal import activities
    from agentic_frameworks.temporal.shared import Notification

    db_path = tmp_path / "test_notify_idempotent.db"

    @asynccontextmanager
    async def _tmp_store():
        async with AsyncSqliteStore.from_conn_string(str(db_path)) as store:
            await store.setup()
            yield store

    monkeypatch.setattr(activities, "get_memory_store", _tmp_store)

    thread_id = f"test-notify-{uuid4()}"
    notification = Notification(kind="needs_approval", thread_id=thread_id, summary="one pending item", payload={"a": 1})

    await activities.notify(notification)
    await activities.notify(notification)

    async with _tmp_store() as store:
        items = await store.asearch(activities.PENDING_APPROVALS_NAMESPACE, limit=1000)
        matching = [item for item in items if item.key == thread_id]

    assert len(matching) == 1
    assert matching[0].value["status"] == "pending_approval"
    assert matching[0].value["summary"] == "one pending item"


async def test_notify_completed_kind_maps_directly_to_status(monkeypatch, tmp_path):
    from contextlib import asynccontextmanager

    from langgraph.store.sqlite.aio import AsyncSqliteStore

    from agentic_frameworks.temporal import activities
    from agentic_frameworks.temporal.shared import Notification

    db_path = tmp_path / "test_notify_completed.db"

    @asynccontextmanager
    async def _tmp_store():
        async with AsyncSqliteStore.from_conn_string(str(db_path)) as store:
            await store.setup()
            yield store

    monkeypatch.setattr(activities, "get_memory_store", _tmp_store)

    thread_id = f"test-notify-completed-{uuid4()}"
    await activities.notify(Notification(kind="completed", thread_id=thread_id, summary="all good", payload=None))

    async with _tmp_store() as store:
        item = await store.aget(activities.PENDING_APPROVALS_NAMESPACE, thread_id)

    assert item is not None
    assert item.value["status"] == "completed"


async def test_run_advisor_graph_delegates_to_run_advisor_once(monkeypatch):
    """run_advisor_graph is a thin wrapper - it must call the same
    run_advisor_once a human calls from the command line (design spec
    Architecture 1), forwarding activity.heartbeat as progress."""
    from agentic_frameworks.temporal import activities
    from agentic_frameworks.temporal.shared import AdvisorRunOutcome, AdvisorRunRequest

    captured = {}

    async def _fake_run_advisor_once(*, thread_id, progress=None):
        captured["thread_id"] = thread_id
        captured["progress_is_heartbeat"] = progress is activities.activity.heartbeat
        return AdvisorRunOutcome(
            status="completed",
            thread_id=thread_id,
            summary="ok",
            pending_payload=None,
            new_buy_count=0,
            action_count=0,
        )

    monkeypatch.setattr(activities, "run_advisor_once", _fake_run_advisor_once)

    outcome = await activities.run_advisor_graph(AdvisorRunRequest(thread_id="test-thread"))

    assert captured["thread_id"] == "test-thread"
    assert outcome.status == "completed"
