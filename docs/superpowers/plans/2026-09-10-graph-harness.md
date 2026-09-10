# Graph Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing `research_node` and `risk_analyst_node` into an actual compiled `StateGraph` (linear `Research → Risk Analyst`), backed by a real `AsyncSqliteSaver` checkpointer pointed at a fresh `agent_memory.db` — the first phase where this system runs end-to-end as a graph, not isolated tested functions.

**Architecture:** A new `GraphState` TypedDict (union of `ResearchState` + `RiskState`'s fields) in `state.py`; a new `checkpointer.py` providing `get_checkpointer()` (an async context manager yielding an `AsyncSqliteSaver`); a new `graph.py` providing `build_graph()` (an uncompiled `StateGraph` with both nodes wired linearly). Compilation happens at the call site, where the checkpointer's context manager is open.

**Tech Stack:** Python 3.14, `langgraph` (already a dependency), `langgraph-checkpoint-sqlite` (already a dependency, provides `AsyncSqliteSaver`), pytest/pytest-asyncio (already configured).

**Spec:** `docs/superpowers/specs/2026-09-10-graph-harness-design.md`

## Global Constraints

- Fail loud: no `try`/`except` anywhere in `checkpointer.py` or `graph.py`.
- `AsyncSqliteSaver`, not `SqliteSaver` — the graph's nodes are async and run via `ainvoke`.
- The checkpointer points at `sandbox-data/agent_memory.db` — never `portfolio.db`. `sandbox-data/` is already gitignored.
- No mocks: the end-to-end test hits the real `portfolio_data_mcp` server and real, read-only `portfolio.db`, same as the Research and Risk Analyst nodes' own tests.
- `ResearchState` and `RiskState` are unchanged and remain each node's own documented partial contribution — `GraphState` is additive, not a replacement.

---

### Task 1: `GraphState` schema

**Files:**
- Modify: `src/agentic_frameworks/graph/state.py`
- Test: `tests/graph/test_state.py`

**Interfaces:**
- Consumes: nothing new — the union of fields already defined by `ResearchState` and `RiskState` in this same file.
- Produces: `GraphState` TypedDict with keys `holdings: list[dict]`, `allocation: dict`, `holdings_by_bucket: dict[str, dict[str, list[dict]]]`, `concentration: dict`, `category_overlap: dict`, `currency_exposure: dict`, `liquidity: dict`. Task 3's `build_graph()` uses this as the graph's `state_schema`.

- [ ] **Step 1: Write the failing test**

Append to `tests/graph/test_state.py`:

```python
def test_graph_state_has_expected_keys():
    from agentic_frameworks.graph.state import GraphState

    hints = get_type_hints(GraphState)
    assert set(hints.keys()) == {
        "holdings",
        "allocation",
        "holdings_by_bucket",
        "concentration",
        "category_overlap",
        "currency_exposure",
        "liquidity",
    }
```

(`get_type_hints` is already imported at the top of this file from earlier work.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph/test_state.py::test_graph_state_has_expected_keys -v`
Expected: FAIL with `ImportError` (`GraphState` doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Append to `src/agentic_frameworks/graph/state.py`:

```python
class GraphState(TypedDict):
    """Combined state for the compiled graph - union of every node's output.

    Includes every field ResearchState and RiskState define individually.
    Those two schemas remain each node's own documented partial
    contribution and what their own tests assert against; this is what
    the compiled StateGraph itself uses as its state_schema, since
    StateGraph takes exactly one schema, not one per node.
    """

    holdings: list[dict]
    allocation: dict
    holdings_by_bucket: dict[str, dict[str, list[dict]]]
    concentration: dict
    category_overlap: dict
    currency_exposure: dict
    liquidity: dict
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph/test_state.py -v`
Expected: PASS (all tests in the file, including the pre-existing `ResearchState`/`RiskState` tests).

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/state.py tests/graph/test_state.py
git commit -m "feat: add GraphState schema"
```

---

### Task 2: Checkpointer wiring

**Files:**
- Create: `src/agentic_frameworks/graph/checkpointer.py`
- Create: `tests/graph/test_checkpointer.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `get_checkpointer()` — an async context manager (`@asynccontextmanager`) yielding a ready `AsyncSqliteSaver`. Also exports `AGENT_MEMORY_DB_PATH` (a `pathlib.Path`) as a module-level constant. Task 4's end-to-end test imports and uses `get_checkpointer()` directly.

- [ ] **Step 1: Write the failing test**

Create `tests/graph/test_checkpointer.py`:

```python
"""Tests for the graph's checkpointer wiring."""

from __future__ import annotations

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


async def test_get_checkpointer_yields_working_saver():
    from agentic_frameworks.graph.checkpointer import AGENT_MEMORY_DB_PATH, get_checkpointer

    async with get_checkpointer() as saver:
        assert isinstance(saver, AsyncSqliteSaver)

    assert AGENT_MEMORY_DB_PATH.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph/test_checkpointer.py -v`
Expected: FAIL with `ModuleNotFoundError` (`checkpointer.py` doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Create `src/agentic_frameworks/graph/checkpointer.py`:

```python
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
```

Note on the path: `__file__` resolves to `src/agentic_frameworks/graph/checkpointer.py`. `.parents[0]` is `graph/`, `[1]` is `agentic_frameworks/`, `[2]` is `src/`, `[3]` is the repo root (`E:\agentic-frameworks`) — so `AGENT_MEMORY_DB_PATH` ends up at `E:\agentic-frameworks\sandbox-data\agent_memory.db`, matching `CLAUDE.md`'s stated location exactly.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph/test_checkpointer.py -v`
Expected: PASS. This actually creates `sandbox-data/agent_memory.db` for the first time — confirm it exists on disk after the test run (it's gitignored, so this is expected and fine).

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/checkpointer.py tests/graph/test_checkpointer.py
git commit -m "feat: add checkpointer wiring for agent_memory.db"
```

---

### Task 3: Graph wiring (`build_graph`)

**Files:**
- Create: `src/agentic_frameworks/graph/graph.py`
- Create: `tests/graph/test_graph.py`

**Interfaces:**
- Consumes: `GraphState` (Task 1), `research_node` (from `agentic_frameworks.graph.nodes.research`, already exists), `risk_analyst_node` (from `agentic_frameworks.graph.nodes.risk_analyst`, already exists).
- Produces: `build_graph() -> StateGraph` — an uncompiled graph with nodes `"research"` and `"risk_analyst"` wired `START → research → risk_analyst → END`. Task 4's end-to-end test calls `build_graph().compile(checkpointer=...)`.

- [ ] **Step 1: Write the failing test**

Create `tests/graph/test_graph.py`:

```python
"""Tests for the graph's node/edge wiring and end-to-end execution."""

from __future__ import annotations

import pytest


def test_build_graph_wires_research_and_risk_analyst():
    from agentic_frameworks.graph.graph import build_graph

    graph = build_graph()

    assert set(graph.nodes.keys()) == {"research", "risk_analyst"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph/test_graph.py::test_build_graph_wires_research_and_risk_analyst -v`
Expected: FAIL with `ModuleNotFoundError` (`graph.py` doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Create `src/agentic_frameworks/graph/graph.py`:

```python
"""Builds the Portfolio Advisor Fleet graph: wires existing nodes together.

Linear for now (Research -> Risk Analyst) - parallel specialist
branches and the Synthesizer arrive in later phases.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agentic_frameworks.graph.nodes.research import research_node
from agentic_frameworks.graph.nodes.risk_analyst import risk_analyst_node
from agentic_frameworks.graph.state import GraphState


def build_graph() -> StateGraph:
    """Wire research_node and risk_analyst_node into an uncompiled StateGraph."""
    graph = StateGraph(GraphState)
    graph.add_node("research", research_node)
    graph.add_node("risk_analyst", risk_analyst_node)
    graph.add_edge(START, "research")
    graph.add_edge("research", "risk_analyst")
    graph.add_edge("risk_analyst", END)
    return graph
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph/test_graph.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/graph.py tests/graph/test_graph.py
git commit -m "feat: wire research and risk_analyst nodes into a StateGraph"
```

---

### Task 4: End-to-end execution test with checkpoint verification

**Files:**
- Modify: `tests/graph/test_graph.py`

**Interfaces:**
- Consumes: `get_checkpointer()` (Task 2), `build_graph()` (Task 3). No new production code — this task is test-only, proving the harness genuinely works end-to-end.
- Produces: nothing new for later tasks — this is the harness's own proof of correctness.

- [ ] **Step 1: Write the failing test**

Append to `tests/graph/test_graph.py`:

```python
async def test_graph_runs_end_to_end_with_real_data_and_persists_checkpoint():
    from agentic_frameworks.graph.checkpointer import get_checkpointer
    from agentic_frameworks.graph.graph import build_graph

    async with get_checkpointer() as checkpointer:
        compiled = build_graph().compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-graph-harness-e2e"}}

        result = await compiled.ainvoke({}, config=config)

        assert set(result.keys()) == {
            "holdings",
            "allocation",
            "holdings_by_bucket",
            "concentration",
            "category_overlap",
            "currency_exposure",
            "liquidity",
        }
        assert len(result["holdings_by_bucket"]["real_estate"]["IN"]) == 1
        assert result["concentration"]["by_holding"]
        assert result["currency_exposure"]
        assert result["liquidity"]["illiquid"]["value_base"] > 0

        state_snapshot = await compiled.aget_state(config)
        assert state_snapshot.values
        assert state_snapshot.config["configurable"]["thread_id"] == "test-graph-harness-e2e"
```

This test hits the real `portfolio_data_mcp` subprocess and real, read-only `portfolio.db` (via `research_node`, unchanged) — same as the existing Research node tests. `holdings_by_bucket["real_estate"]["IN"]` having exactly 1 entry is the same confirmed real-data assertion the Research node's own test already makes.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph/test_graph.py::test_graph_runs_end_to_end_with_real_data_and_persists_checkpoint -v`
Expected: FAIL — at this point `checkpointer.py` and `graph.py` both already exist (from Tasks 2-3), so this should fail only if there's a real bug in the wiring, not from a missing module. If it fails with an `ImportError` for `checkpointer` or `graph`, something is wrong with Tasks 2-3 — stop and report rather than guessing a fix.

- [ ] **Step 3: Confirm no implementation changes are needed**

This task adds no production code — `checkpointer.py` (Task 2) and `graph.py` (Task 3) already provide everything this test needs. If the test in Step 2 failed for a reason other than "not yet written," investigate whether Task 2 or Task 3's implementation has a real bug (e.g. wrong node names, wrong edge order) before proceeding — do not weaken this test to make it pass.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph/test_graph.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git add tests/graph/test_graph.py
git commit -m "test: add end-to-end graph execution and checkpoint persistence test"
```

---

### Task 5: Full-suite verification (Phase 3 gate)

**Files:** none created or modified — verification only.

**Interfaces:** none — this task consumes everything from Tasks 1-4 to confirm the phase is genuinely done.

- [ ] **Step 1: Run the entire test suite**

Run: `uv run pytest tests/ -v`
Expected: all tests from Tasks 1-4 PASS, plus all pre-existing Research and Risk Analyst node tests still PASS (15 tests total: 4 from Phase 1 + 7 from Phase 2 + 4 from this phase).

- [ ] **Step 2: Confirm `portfolio.db` is untouched, and `agent_memory.db` now exists**

`portfolio.db` must remain read-only and unchanged: check its modified timestamp is identical before/after the run (same check as Phases 1 and 2). Separately, confirm `sandbox-data/agent_memory.db` now exists on disk — this phase's first and only write target, created by Task 2/Task 4's tests.

- [ ] **Step 3: Report phase completion**

This phase (Graph Harness) is done once Steps 1-2 both pass. Per `CLAUDE.md`'s phased-development rule, do not begin planning Phase 4 (the next graph node, e.g. the first specialist) until this step is confirmed complete and reported back.
