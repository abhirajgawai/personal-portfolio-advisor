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


@pytest.fixture
async def real_research_risk_state():
    """Real `research_node` + `risk_analyst_node` output against the real
    `portfolio.db` (via `portfolio_data_mcp`, read-only) - so `holdings`/
    `allocation`/`holdings_by_bucket` and the four risk views are real
    shapes with real values, per design spec Testing item 1.

    Function-scoped (not session-scoped as the spec's wording suggests):
    this project's pytest-asyncio config (`asyncio_mode = "auto"`,
    function-scoped default event loop) makes a session-scoped async
    fixture require a non-default loop-scope configuration this codebase
    doesn't otherwise use - every other real-data test in this suite
    (e.g. test_mutual_fund_specialist.py) already re-fetches per-test
    rather than fighting that, so this fixture follows the same,
    already-established pattern instead of introducing a new one for
    Phase 8 alone. The MCP round-trip is fast enough that this doesn't
    meaningfully slow the suite down.
    """
    from agentic_frameworks.graph.nodes.research import research_node
    from agentic_frameworks.graph.nodes.risk_analyst import risk_analyst_node

    research = await research_node({})
    risk = risk_analyst_node(research)
    return {**research, **risk}
