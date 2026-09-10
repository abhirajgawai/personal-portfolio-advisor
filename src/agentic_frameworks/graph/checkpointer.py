"""Checkpointer wiring for the graph's short-term/thread memory.

Points at a fresh agent_memory.db under sandbox-data/ - never
portfolio.db. AsyncSqliteSaver, not the sync SqliteSaver, since the
graph's nodes are async and the graph runs via ainvoke.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

AGENT_MEMORY_DB_PATH = Path(__file__).resolve().parents[3] / "sandbox-data" / "agent_memory.db"


@asynccontextmanager
async def get_checkpointer() -> AsyncIterator[AsyncSqliteSaver]:
    """Yield an AsyncSqliteSaver pointed at sandbox-data/agent_memory.db."""
    AGENT_MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(AGENT_MEMORY_DB_PATH)) as saver:
        yield saver
