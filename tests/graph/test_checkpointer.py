"""Tests for the graph's checkpointer wiring."""

from __future__ import annotations

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


async def test_get_checkpointer_yields_working_saver():
    from agentic_frameworks.graph.checkpointer import AGENT_MEMORY_DB_PATH, get_checkpointer

    async with get_checkpointer() as saver:
        assert isinstance(saver, AsyncSqliteSaver)

    assert AGENT_MEMORY_DB_PATH.exists()
