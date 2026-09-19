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

from agentic_frameworks.db_lock import advisor_memory_lock

AGENT_MEMORY_DB_PATH = Path(__file__).resolve().parents[3] / "sandbox-data" / "agent_memory.db"


@asynccontextmanager
async def get_checkpointer() -> AsyncIterator[AsyncSqliteSaver]:
    """Yield an AsyncSqliteSaver pointed at sandbox-data/agent_memory.db.

    Acquires the process-wide OS advisory lock on this path first (see
    db_lock.py) and holds it for the whole time the saver is open - the
    structural half of the Temporal phase's concurrency guarantee
    (design spec Architecture 3): every path that can possibly open
    agent_memory.db goes through this function or get_memory_store(),
    so the lock is taken here rather than at any one caller's entry
    point.
    """
    AGENT_MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with advisor_memory_lock(AGENT_MEMORY_DB_PATH):
        async with AsyncSqliteSaver.from_conn_string(str(AGENT_MEMORY_DB_PATH)) as saver:
            yield saver
