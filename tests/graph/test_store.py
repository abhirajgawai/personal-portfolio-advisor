"""Tests for the graph's store wiring (long-term/cross-run memory)."""

from __future__ import annotations


async def test_get_memory_store_yields_working_store():
    """This deliberately exercises the REAL sandbox-data/agent_memory.db
    path (not a temp file) - that's the point of this test, confirming
    `get_memory_store()` actually points where the design spec says it
    does. The written probe row is therefore cleaned up in a `finally`,
    so running this test doesn't leave a stray `("test_ns",)`/`test_key`
    row behind in the real, shared memory store.
    """
    from langgraph.store.sqlite.aio import AsyncSqliteStore

    from agentic_frameworks.graph.store import AGENT_MEMORY_DB_PATH, get_memory_store

    async with get_memory_store() as store:
        assert isinstance(store, AsyncSqliteStore)

        try:
            await store.aput(("test_ns",), "test_key", {"hello": "world"})
            item = await store.aget(("test_ns",), "test_key")

            assert item is not None
            assert item.value == {"hello": "world"}
        finally:
            await store.adelete(("test_ns",), "test_key")

    assert AGENT_MEMORY_DB_PATH.exists()


async def test_get_memory_store_shares_path_constant_with_checkpointer():
    """Not re-derived - the same path constant, imported once."""
    from agentic_frameworks.graph.checkpointer import AGENT_MEMORY_DB_PATH as checkpointer_path
    from agentic_frameworks.graph.store import AGENT_MEMORY_DB_PATH as store_path

    assert store_path is checkpointer_path


async def test_checkpointer_and_store_tables_coexist_in_one_file():
    """Also exercises the REAL agent_memory.db path deliberately (see
    test_get_memory_store_yields_working_store's docstring) - the probe
    row is cleaned up the same way, so this test leaves no residue in
    the real, shared memory store either.
    """
    import aiosqlite

    from agentic_frameworks.graph.checkpointer import AGENT_MEMORY_DB_PATH, get_checkpointer
    from agentic_frameworks.graph.store import get_memory_store

    async with get_checkpointer() as _checkpointer, get_memory_store() as store:
        try:
            await store.aput(("test_ns",), "test_key", {"hello": "world"})
        finally:
            await store.adelete(("test_ns",), "test_key")

    async with aiosqlite.connect(str(AGENT_MEMORY_DB_PATH)) as conn:
        cursor = await conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in await cursor.fetchall()}

    assert {"checkpoints", "writes", "store", "store_migrations"} <= tables
