"""Store wiring for the graph's long-term/cross-run memory.

Mirrors checkpointer.py exactly: points at the same fresh
sandbox-data/agent_memory.db path (AGENT_MEMORY_DB_PATH, imported not
re-derived) the checkpointer already writes to. AsyncSqliteStore, not
the sync SqliteStore, for the same reason the checkpointer is
AsyncSqliteSaver - every node in this graph is async and the graph
runs via ainvoke.

Verified live (see docs/superpowers/specs/2026-09-18-langgraph-store-
design.md "Background/Constraints"): the checkpointer and this store
can share the one file concurrently under WAL, across a real
four-parallel-branch fan-out, with no `database is locked` errors -
they write to disjoint tables (checkpoints/writes vs. store/
store_migrations).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.store.sqlite.aio import AsyncSqliteStore

from agentic_frameworks.graph.checkpointer import AGENT_MEMORY_DB_PATH


@asynccontextmanager
async def get_memory_store() -> AsyncIterator[AsyncSqliteStore]:
    """Yield an AsyncSqliteStore pointed at sandbox-data/agent_memory.db.

    `await store.setup()` is not strictly required - migrations run
    lazily on first `aput` against a brand-new file (live-verified) -
    but it is called explicitly here so the schema exists
    deterministically before the first *read*, per the design spec.
    """
    AGENT_MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteStore.from_conn_string(str(AGENT_MEMORY_DB_PATH)) as store:
        await store.setup()
        yield store
