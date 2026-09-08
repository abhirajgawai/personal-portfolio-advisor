# Research Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the standalone Research node — an async LangGraph node that fetches real portfolio holdings via the `portfolio_data_mcp` server and returns them grouped by `asset_class × geography`.

**Architecture:** `langchain-mcp-adapters` launches `portfolio_data_mcp` as a stdio subprocess (using that sibling project's own venv Python). The node calls its `latest_snapshot` and `allocation_summary` tools, parses their JSON-text content, groups the holdings, and returns a `ResearchState` update. No graph or other nodes exist yet — this plan produces only the node function, tested standalone against the real, read-only `portfolio.db`.

**Tech Stack:** Python 3.14, `langgraph`, `langchain-mcp-adapters`, `pytest`, `pytest-asyncio`, `uv`.

**Spec:** `docs/superpowers/specs/2026-09-08-research-node-design.md`

## Global Constraints

- `portfolio.db` is read-only, always — this plan never writes to it and never bypasses `portfolio_data_mcp` to touch it directly.
- `portfolio_data_mcp` runs from its own project/venv (`E:\agentic-portfolio-dashboard\.venv\Scripts\python.exe`, `cwd=E:\agentic-portfolio-dashboard`) — it is not importable from this project's environment.
- Fail loud: no `try`/`except` around MCP calls; `MultiServerMCPClient` must be constructed with `handle_tool_errors=False`.
- All tests use the real `portfolio_data_mcp` server and real `portfolio.db` — no mocks (per `CLAUDE.md`'s phased-development testing rule).
- Confirmed real counts to assert against: equity/IN=36, equity/US=3, etf/IN=2, etf/US=3, mutual_fund/IN=9, real_estate/IN=1; `allocation["totalBase"]` ≈ 10,331,248.19.

---

## Task 1: Project test setup

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/__init__.py`
- Create: `tests/graph/__init__.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: a working `pytest` + `pytest-asyncio` setup any later task's tests rely on (async tests run without a `@pytest.mark.asyncio` decorator on every function, via `asyncio_mode = "auto"`).

- [ ] **Step 1: Add pytest and pytest-asyncio as dev dependencies**

Run: `uv add --dev pytest pytest-asyncio`

- [ ] **Step 2: Configure pytest for async-by-default tests**

Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 3: Create the test package markers**

Create `tests/__init__.py` (empty file).
Create `tests/graph/__init__.py` (empty file).

- [ ] **Step 4: Verify pytest collects (no tests yet, should report 0 collected, no errors)**

Run: `uv run pytest tests/ -v`
Expected: `collected 0 items`, exit code 0 (no errors — just nothing to run yet).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock tests/__init__.py tests/graph/__init__.py
git commit -m "test: set up pytest + pytest-asyncio for the graph package"
```

---

## Task 2: `ResearchState` schema

**Files:**
- Create: `src/agentic_frameworks/graph/__init__.py`
- Create: `src/agentic_frameworks/graph/state.py`
- Test: `tests/graph/test_state.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ResearchState` (a `TypedDict` with keys `holdings: list[dict]`, `allocation: dict`, `holdings_by_bucket: dict[str, dict[str, list[dict]]]`), imported by Task 4 (`research.py`) as the node's return type annotation.

- [ ] **Step 1: Write the failing test**

Create `tests/graph/test_state.py`:

```python
from typing import get_type_hints

from agentic_frameworks.graph.state import ResearchState


def test_research_state_has_expected_keys():
    hints = get_type_hints(ResearchState)
    assert set(hints.keys()) == {"holdings", "allocation", "holdings_by_bucket"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph/test_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentic_frameworks.graph'` (the package doesn't exist yet).

- [ ] **Step 3: Create the graph package and the state module**

Create `src/agentic_frameworks/graph/__init__.py` (empty file).

Create `src/agentic_frameworks/graph/state.py`:

```python
"""Shared state schema for the Portfolio Advisor Fleet graph."""

from __future__ import annotations

from typing import TypedDict


class ResearchState(TypedDict):
    """State the Research node produces: raw holdings plus a grouped view.

    holdings: the raw `latest_snapshot` rows (platform, geography, ticker,
        name, currency, sector, asset_class, quantity, price,
        market_value_base, updated_at).
    allocation: the raw `allocation_summary` output (totalBase,
        holdingsCount, byGeography, byPlatform, bySector, byAssetClass).
    holdings_by_bucket: `holdings` regrouped as
        {asset_class: {geography: [holding, ...]}}.
    """

    holdings: list[dict]
    allocation: dict
    holdings_by_bucket: dict[str, dict[str, list[dict]]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph/test_state.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/__init__.py src/agentic_frameworks/graph/state.py tests/graph/test_state.py
git commit -m "feat: add ResearchState schema"
```

---

## Task 3: MCP client wiring for `portfolio_data_mcp`

**Files:**
- Create: `src/agentic_frameworks/graph/mcp_clients.py`
- Test: `tests/graph/test_mcp_clients.py`

**Interfaces:**
- Consumes: nothing new (uses `langchain_mcp_adapters.client.MultiServerMCPClient` directly).
- Produces: `get_portfolio_data_client() -> MultiServerMCPClient`, consumed by Task 4's `research_node`.

- [ ] **Step 1: Write the failing test**

Create `tests/graph/test_mcp_clients.py`:

```python
from agentic_frameworks.graph.mcp_clients import get_portfolio_data_client


async def test_get_portfolio_data_client_exposes_expected_tools():
    client = get_portfolio_data_client()
    tools = await client.get_tools()
    names = {tool.name for tool in tools}
    assert {"latest_snapshot", "allocation_summary"}.issubset(names)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph/test_mcp_clients.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentic_frameworks.graph.mcp_clients'`.

- [ ] **Step 3: Write the MCP client helper**

Create `src/agentic_frameworks/graph/mcp_clients.py`:

```python
"""MCP client wiring for the read-only portfolio_data_mcp server.

portfolio_data_mcp lives in a separate project (E:\\agentic-portfolio-dashboard)
with its own venv - it is not importable from this project's environment, so
it must be launched as a subprocess speaking MCP over stdio.
"""

from __future__ import annotations

from langchain_mcp_adapters.client import MultiServerMCPClient

PORTFOLIO_DATA_MCP_ROOT = r"E:\agentic-portfolio-dashboard"
PORTFOLIO_DATA_MCP_PYTHON = r"E:\agentic-portfolio-dashboard\.venv\Scripts\python.exe"


def get_portfolio_data_client() -> MultiServerMCPClient:
    """Build the MCP client for the portfolio_data_mcp stdio server.

    handle_tool_errors=False so a failed tool call raises instead of being
    swallowed into error-content - Research fails loud, never on fabricated
    or partial data.
    """
    return MultiServerMCPClient(
        {
            "portfolio_data": {
                "transport": "stdio",
                "command": PORTFOLIO_DATA_MCP_PYTHON,
                "args": ["-m", "portfolio_data_mcp.server"],
                "cwd": PORTFOLIO_DATA_MCP_ROOT,
            }
        },
        handle_tool_errors=False,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph/test_mcp_clients.py -v`
Expected: PASS — this actually launches the real `portfolio_data_mcp` subprocess and lists its tools.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/mcp_clients.py tests/graph/test_mcp_clients.py
git commit -m "feat: add MCP client wiring for portfolio_data_mcp"
```

---

## Task 4: `research_node`

**Files:**
- Create: `src/agentic_frameworks/graph/nodes/__init__.py`
- Create: `src/agentic_frameworks/graph/nodes/research.py`
- Test: `tests/graph/test_research_node.py`

**Interfaces:**
- Consumes: `get_portfolio_data_client` (Task 3, `src/agentic_frameworks/graph/mcp_clients.py`), `ResearchState` (Task 2, `src/agentic_frameworks/graph/state.py`).
- Produces: `async def research_node(state: dict) -> ResearchState` — the deliverable of this whole plan; later graph-wiring work will import this function.

- [ ] **Step 1: Write the failing tests**

Create `tests/graph/test_research_node.py`:

```python
import pytest

from agentic_frameworks.graph.nodes.research import research_node


async def test_research_node_groups_real_holdings_by_asset_class_and_geography():
    result = await research_node({})

    buckets = result["holdings_by_bucket"]
    assert set(buckets.keys()) == {"equity", "etf", "mutual_fund", "real_estate"}

    assert len(buckets["real_estate"]["IN"]) == 1
    assert len(buckets["equity"]["IN"]) == 36
    assert len(buckets["equity"]["US"]) == 3
    assert len(buckets["etf"]["IN"]) == 2
    assert len(buckets["etf"]["US"]) == 3
    assert len(buckets["mutual_fund"]["IN"]) == 9

    assert result["allocation"]["totalBase"] > 0
    assert result["holdings"]


async def test_research_node_fails_loud_on_bad_server_config(monkeypatch):
    import agentic_frameworks.graph.mcp_clients as mcp_clients
    from langchain_mcp_adapters.client import MultiServerMCPClient

    def broken_client():
        return MultiServerMCPClient(
            {
                "portfolio_data": {
                    "transport": "stdio",
                    "command": mcp_clients.PORTFOLIO_DATA_MCP_PYTHON,
                    "args": ["-m", "nonexistent_module_xyz"],
                    "cwd": mcp_clients.PORTFOLIO_DATA_MCP_ROOT,
                }
            },
            handle_tool_errors=False,
        )

    monkeypatch.setattr(mcp_clients, "get_portfolio_data_client", broken_client)

    with pytest.raises(Exception):
        await research_node({})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/graph/test_research_node.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentic_frameworks.graph.nodes'`.

- [ ] **Step 3: Write the Research node**

Create `src/agentic_frameworks/graph/nodes/__init__.py` (empty file).

Create `src/agentic_frameworks/graph/nodes/research.py`:

```python
"""Research node: fetch current portfolio holdings via portfolio_data_mcp."""

from __future__ import annotations

import json

from agentic_frameworks.graph.mcp_clients import get_portfolio_data_client
from agentic_frameworks.graph.state import ResearchState


def _tool_result_to_dict(content: list[dict]) -> dict:
    """Parse an MCP tool's content-block list into the JSON dict it wraps.

    langchain-mcp-adapters returns tool output as
    [{"type": "text", "text": "<json string>"}], not a parsed dict.
    """
    return json.loads(content[0]["text"])


def _group_by_bucket(holdings: list[dict]) -> dict[str, dict[str, list[dict]]]:
    """Group holdings as {asset_class: {geography: [holding, ...]}}."""
    buckets: dict[str, dict[str, list[dict]]] = {}
    for holding in holdings:
        by_geography = buckets.setdefault(holding["asset_class"], {})
        by_geography.setdefault(holding["geography"], []).append(holding)
    return buckets


async def research_node(state: dict) -> ResearchState:
    """Fetch current holdings and allocation, grouped by asset_class x geography.

    Fails loud: any MCP/subprocess/DB error propagates uncaught, since no
    downstream node can produce a meaningful recommendation from partial or
    fabricated holdings data.
    """
    client = get_portfolio_data_client()
    tools = await client.get_tools()
    tools_by_name = {tool.name: tool for tool in tools}

    snapshot_content = await tools_by_name["latest_snapshot"].ainvoke({})
    allocation_content = await tools_by_name["allocation_summary"].ainvoke({})

    snapshot = _tool_result_to_dict(snapshot_content)
    allocation = _tool_result_to_dict(allocation_content)

    return {
        "holdings": snapshot["holdings"],
        "allocation": allocation,
        "holdings_by_bucket": _group_by_bucket(snapshot["holdings"]),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/graph/test_research_node.py -v`
Expected: PASS for both tests — the first makes a real round trip through `portfolio_data_mcp` to the real `portfolio.db`; the second confirms a broken MCP server config raises instead of returning partial state.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/nodes/__init__.py src/agentic_frameworks/graph/nodes/research.py tests/graph/test_research_node.py
git commit -m "feat: add research_node fetching real holdings via portfolio_data_mcp"
```

---

## Task 5: Full-suite verification (Phase 1 gate)

**Files:** none created or modified — verification only.

**Interfaces:** none — this task consumes everything from Tasks 1-4 to confirm the phase is genuinely done.

- [ ] **Step 1: Run the entire test suite**

Run: `uv run pytest tests/ -v`
Expected: all tests from Tasks 2-4 PASS (state schema, MCP client tool discovery, Research node happy path, Research node fail-loud path).

- [ ] **Step 2: Confirm no accidental writes to `portfolio.db`**

Run: `uv run python -c "import sqlite3, pathlib; p = pathlib.Path(r'E:\agentic-portfolio-dashboard\data\portfolio.db'); print('mtime unchanged check - compare against known state before/after test run')"`

Manually compare the file's modified timestamp from before Task 1 to now (e.g. via `git log`-independent `ls -la E:\agentic-portfolio-dashboard\data\portfolio.db` before/after) - it must be identical, since every access goes through the `mode=ro` connection in `portfolio_data_mcp`.

- [ ] **Step 3: Report phase completion**

This phase (Research node) is done once Steps 1-2 both pass. Per `CLAUDE.md`'s phased-development rule, do not begin planning Task/Phase 2 (the next graph node) until this step is confirmed complete and reported back.
