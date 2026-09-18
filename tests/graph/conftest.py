"""Shared fixtures for graph tests."""

from __future__ import annotations

import pytest
from langgraph.store.sqlite.aio import AsyncSqliteStore


@pytest.fixture
async def memory_store(tmp_path):
    """A real AsyncSqliteStore on a temp SQLite file - no mocking.

    Function-scoped and on a fresh temp file per test, so rejection-
    memory tests never see another test's records.
    """
    db_path = tmp_path / "test_agent_memory.db"
    async with AsyncSqliteStore.from_conn_string(str(db_path)) as store:
        await store.setup()
        yield store
