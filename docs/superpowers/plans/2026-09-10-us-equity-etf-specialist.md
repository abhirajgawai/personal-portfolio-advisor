# US Equity/ETF Specialist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `us_equity_etf_specialist_node` — the first LLM-calling node in this graph. Proposes US equity/ETF new-buy candidates for real portfolio gaps, verifies each is a real instrument, and runs survivors through the New-Buy Conviction Gate, producing a strictly validated recommendation per candidate.

**Architecture:** Three-stage pipeline — a `pydantic-ai` candidate-generator `Agent` (capped in code, not trusted from the model), deterministic ticker verification against `yfinance-mcp`, and a `pydantic-ai` Conviction Gate evaluator `Agent` with `yfinance-mcp` tools attached. Model: `deepseek/deepseek-v4.1-flash` via OpenRouter, a named constant.

**Tech Stack:** Python 3.14, `pydantic-ai` (new dependency), `pydantic` (transitive, used directly for schemas), the `yfinance-mcp` package (launched via `uvx yfmcp@latest`, no new project dependency — it's an external tool, same pattern as `portfolio_data_mcp`).

**Spec:** `docs/superpowers/specs/2026-09-10-us-equity-etf-specialist-design.md`

## Global Constraints

- Fail loud: no `try`/`except` anywhere except `pydantic-ai`'s own bounded retry-on-validation-failure (via `Agent(retries=...)`), which is the one sanctioned exception per the spec.
- Model is a named constant, `SPECIALIST_MODEL_NAME = "deepseek/deepseek-v4.1-flash"` — never hardcoded inline at each call site.
- Hard caps enforced in code: max 3 candidates per sector, max 5 total per run. Never trust the model's own count.
- `pydantic-ai`'s `Agent(retries=...)` must be set explicitly on every agent — never left at the library default.
- Automated tests use `pydantic_ai.models.test.TestModel` for LLM-calling logic (free, proves wiring not judgment). Ticker verification is tested against the real `yfinance-mcp` server (free, deterministic, no LLM). No automated test calls the real paid model.
- **Phase-completion gate:** one real, manual run against the actual `deepseek/deepseek-v4.1-flash` model (via a real `OPENROUTER_API_KEY`) is required before this phase is done — output captured and reviewed by Abhi. This is Task 9's final step and cannot be automated away.
- Live-verified facts binding on implementation: `yfinance-mcp`'s `yfinance_get_ticker_info` tool takes a `symbol` argument (not `ticker`). A nonexistent ticker does **not** raise — it returns a near-empty JSON payload; `longName` being present (non-null) is the confirmed reliable signal a real instrument was found.

---

### Task 1: `CandidateEvaluation` and `ProposedCandidate` schemas

**Files:**
- Create: `src/agentic_frameworks/graph/schemas.py`
- Create: `tests/graph/test_schemas.py`
- Modify: `pyproject.toml` (add `pydantic-ai` dependency)

**Interfaces:**
- Consumes: nothing new.
- Produces: `ProposedCandidate(BaseModel)` with fields `ticker: str`, `sector: str` — Task 6's candidate generator uses this as its `output_type`. `CandidateEvaluation(BaseModel)` with fields `action: Literal["Buy", "Hold", "Sell", "Reduce"]`, `confidence: float`, `reasoning: str`, `risk_flags: list[str]`, `entry_approach: Literal["LumpSum", "Staggered"] | None`, `suggested_platform: str | None` — Task 7's Conviction Gate evaluator uses this as its `output_type`.

- [ ] **Step 1: Add the pydantic-ai dependency**

Run: `uv add pydantic-ai`
Expected: `pyproject.toml`'s `dependencies` list gains a `pydantic-ai>=...` entry, `uv.lock` updates.

- [ ] **Step 2: Write the failing test**

Create `tests/graph/test_schemas.py`:

```python
"""Tests for LLM output schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_candidate_evaluation_accepts_valid_data():
    from agentic_frameworks.graph.schemas import CandidateEvaluation

    evaluation = CandidateEvaluation(
        action="Buy",
        confidence=0.8,
        reasoning="Fills the Energy gap with strong fundamentals.",
        risk_flags=["sector_concentration"],
        entry_approach="Staggered",
        suggested_platform="moomoo",
    )

    assert evaluation.action == "Buy"
    assert evaluation.confidence == 0.8


def test_candidate_evaluation_rejects_invalid_action():
    from agentic_frameworks.graph.schemas import CandidateEvaluation

    with pytest.raises(ValidationError):
        CandidateEvaluation(
            action="Speculate",
            confidence=0.8,
            reasoning="not a real action",
            risk_flags=[],
            entry_approach=None,
            suggested_platform=None,
        )


def test_proposed_candidate_requires_ticker_and_sector():
    from agentic_frameworks.graph.schemas import ProposedCandidate

    candidate = ProposedCandidate(ticker="XOM", sector="Energy")

    assert candidate.ticker == "XOM"
    assert candidate.sector == "Energy"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/graph/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError` (`schemas.py` doesn't exist yet).

- [ ] **Step 4: Write minimal implementation**

Create `src/agentic_frameworks/graph/schemas.py`:

```python
"""Structured output schemas for LLM-driven nodes.

Every LLM call in this graph must produce a strictly validated
schema - no freeform text leaks through unvalidated, per plan.md.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ProposedCandidate(BaseModel):
    """One ticker the candidate generator proposes for a specific gap sector."""

    ticker: str
    sector: str


class CandidateEvaluation(BaseModel):
    """A new-buy specialist's verdict on one candidate, after the Conviction Gate."""

    action: Literal["Buy", "Hold", "Sell", "Reduce"]
    confidence: float
    reasoning: str
    risk_flags: list[str]
    entry_approach: Literal["LumpSum", "Staggered"] | None
    suggested_platform: str | None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/graph/test_schemas.py -v`
Expected: PASS (all 3 tests).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/agentic_frameworks/graph/schemas.py tests/graph/test_schemas.py
git commit -m "feat: add pydantic-ai dependency and specialist output schemas"
```

---

### Task 2: Specialist model wiring

**Files:**
- Create: `src/agentic_frameworks/graph/models.py`
- Create: `tests/graph/test_models.py`

**Interfaces:**
- Consumes: `pydantic-ai` (Task 1's dependency addition).
- Produces: `SPECIALIST_MODEL_NAME` (str constant) and `get_specialist_model() -> OpenRouterModel` — Tasks 6 and 7 both call this to build their `Agent`'s model.

- [ ] **Step 1: Write the failing test**

Create `tests/graph/test_models.py`:

```python
"""Tests for LLM model wiring."""

from __future__ import annotations


def test_get_specialist_model_uses_configured_model_name(monkeypatch):
    from agentic_frameworks.graph import models

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

    model = models.get_specialist_model()

    assert model.model_name == models.SPECIALIST_MODEL_NAME
    assert models.SPECIALIST_MODEL_NAME == "deepseek/deepseek-v4.1-flash"
```

This test does not make a network call — `OpenRouterModel`'s constructor only builds the object, it doesn't contact OpenRouter. `monkeypatch.setenv` supplies a fake key so `get_specialist_model()` doesn't raise `KeyError` on a missing real one.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError` (`models.py` doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Create `src/agentic_frameworks/graph/models.py`:

```python
"""LLM model wiring for specialist/Synthesizer nodes.

Single default model, no fallback list yet - kept as an easy-to-swap
named constant per CLAUDE.md's "LLM cost control" rule 1 (explicit
per-node choice, never a silent default).
"""

from __future__ import annotations

import os

from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

SPECIALIST_MODEL_NAME = "deepseek/deepseek-v4.1-flash"


def get_specialist_model() -> OpenRouterModel:
    """Build the OpenRouter-backed model used by specialist nodes."""
    api_key = os.environ["OPENROUTER_API_KEY"]
    return OpenRouterModel(SPECIALIST_MODEL_NAME, provider=OpenRouterProvider(api_key=api_key))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph/test_models.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/models.py tests/graph/test_models.py
git commit -m "feat: add specialist LLM model wiring"
```

---

### Task 3: `yfinance-mcp` client wiring

**Files:**
- Modify: `src/agentic_frameworks/graph/mcp_clients.py`
- Create: `tests/graph/test_yfinance_client.py`

**Interfaces:**
- Consumes: `pydantic-ai`'s `MCPToolset`/`FastMCPClient`/`StdioTransport` (already available via Task 1's dependency).
- Produces: `get_yfinance_toolset() -> MCPToolset` — Tasks 5 and 7 both use this to reach `yfinance-mcp`.

- [ ] **Step 1: Write the failing test**

Create `tests/graph/test_yfinance_client.py`:

```python
"""Tests for the yfinance-mcp client wiring."""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


async def test_get_yfinance_toolset_connects_and_exposes_tools():
    from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset

    toolset = get_yfinance_toolset()
    agent = Agent(TestModel(), toolsets=[toolset])

    result = await agent.run("what tools do you have?")

    assert "yfinance_get_ticker_info" in result.output
```

This launches the real `yfinance-mcp` subprocess (via `uvx yfmcp@latest`) and confirms the connection genuinely works — `TestModel` avoids the paid LLM call but still exercises the real MCP tool-calling path (confirmed live during design: `TestModel` automatically calls all available tools once).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph/test_yfinance_client.py -v`
Expected: FAIL with `ImportError` (`get_yfinance_toolset` doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Append to `src/agentic_frameworks/graph/mcp_clients.py`:

```python
from pydantic_ai.mcp import FastMCPClient, MCPToolset, StdioTransport


def get_yfinance_toolset() -> MCPToolset:
    """Build the MCPToolset for the yfinance-mcp server (US and India equity/ETF data).

    Launched via `uvx yfmcp@latest` - an external package, not a project
    dependency, same pattern as portfolio_data_mcp being a sibling
    subprocess rather than an import.
    """
    transport = StdioTransport(command="uvx", args=["yfmcp@latest"])
    client = FastMCPClient(transport)
    return MCPToolset(client)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph/test_yfinance_client.py -v`
Expected: PASS. This may take a few seconds the first time (`uvx` fetches `yfmcp` if not cached).

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/mcp_clients.py tests/graph/test_yfinance_client.py
git commit -m "feat: add yfinance-mcp client wiring"
```

---

### Task 4: `GraphState` extension

**Files:**
- Modify: `src/agentic_frameworks/graph/state.py`
- Test: `tests/graph/test_state.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `GraphState` gains a new key `us_equity_etf_candidates: list[dict]`. Task 8's node returns a dict with this key.

- [ ] **Step 1: Write the failing test**

Append to `tests/graph/test_state.py`:

```python
def test_graph_state_includes_us_equity_etf_candidates():
    from agentic_frameworks.graph.state import GraphState

    hints = get_type_hints(GraphState)
    assert "us_equity_etf_candidates" in hints
    assert hints["us_equity_etf_candidates"] == list[dict]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph/test_state.py::test_graph_state_includes_us_equity_etf_candidates -v`
Expected: FAIL with `AssertionError` (key doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Modify `src/agentic_frameworks/graph/state.py`'s `GraphState` class — add a field:

```python
    us_equity_etf_candidates: list[dict]
```

Add it as the last field in the class, and update the class's docstring to mention it: `us_equity_etf_candidates`: the US Equity/ETF Specialist's evaluated candidates, each a `CandidateEvaluation` dict.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph/test_state.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/state.py tests/graph/test_state.py
git commit -m "feat: add us_equity_etf_candidates to GraphState"
```

---

### Task 5: Ticker verification

**Files:**
- Create: `src/agentic_frameworks/graph/nodes/us_equity_etf_specialist.py`
- Create: `tests/graph/test_us_equity_etf_specialist.py`

**Interfaces:**
- Consumes: `get_yfinance_toolset()` (Task 3).
- Produces: `_verify_ticker(symbol: str, toolset: MCPToolset) -> bool` — Task 8's node composition calls this once per proposed candidate.

- [ ] **Step 1: Write the failing test**

Create `tests/graph/test_us_equity_etf_specialist.py`:

```python
"""Tests for the US Equity/ETF Specialist node and its stages."""

from __future__ import annotations


async def test_verify_ticker_confirms_real_instrument():
    from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset
    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import _verify_ticker

    toolset = get_yfinance_toolset()
    async with toolset:
        assert await _verify_ticker("AAPL", toolset) is True


async def test_verify_ticker_rejects_fake_ticker():
    from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset
    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import _verify_ticker

    toolset = get_yfinance_toolset()
    async with toolset:
        assert await _verify_ticker("NOTAREALTICKER999", toolset) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph/test_us_equity_etf_specialist.py -v`
Expected: FAIL with `ModuleNotFoundError` (the node file doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Create `src/agentic_frameworks/graph/nodes/us_equity_etf_specialist.py`:

```python
"""The US Equity/ETF Specialist node: proposes, verifies, and evaluates
US equity/ETF new-buy candidates against the New-Buy Conviction Gate.

First LLM-calling node in this graph. Fails loud, same as every prior
node - no try/except, except pydantic-ai's own bounded retry-on-
validation-failure (Agent(retries=...)), which validates output
correctness rather than swallowing a real error.
"""

from __future__ import annotations

import json

from pydantic_ai.mcp import MCPToolset


async def _verify_ticker(symbol: str, toolset: MCPToolset) -> bool:
    """Confirm `symbol` resolves to a real, current instrument via yfinance-mcp.

    A nonexistent/delisted ticker doesn't raise - yfinance-mcp returns a
    near-empty payload instead (confirmed live: {"trailingPegRatio": null}
    for a fake symbol). `longName` being present is the reliable signal
    a real instrument was found.
    """
    result = await toolset.direct_call_tool("yfinance_get_ticker_info", {"symbol": symbol})
    data = json.loads(result)
    return data.get("longName") is not None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph/test_us_equity_etf_specialist.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/nodes/us_equity_etf_specialist.py tests/graph/test_us_equity_etf_specialist.py
git commit -m "feat: add ticker verification against yfinance-mcp"
```

---

### Task 6: Candidate generator and cap enforcement

**Files:**
- Modify: `src/agentic_frameworks/graph/nodes/us_equity_etf_specialist.py`
- Modify: `tests/graph/test_us_equity_etf_specialist.py`

**Interfaces:**
- Consumes: `get_specialist_model()` (Task 2), `ProposedCandidate` (Task 1).
- Produces: `US_GAP_SECTORS` (list of sector name strings), `_build_candidate_generator(model) -> Agent`, `_apply_candidate_cap(candidates: list[ProposedCandidate]) -> list[ProposedCandidate]` (enforces max 3 per sector, max 5 total) — Task 8's node composition uses all three.

- [ ] **Step 1: Write the failing test**

Append to `tests/graph/test_us_equity_etf_specialist.py`:

```python
def test_apply_candidate_cap_enforces_per_sector_and_total_limits():
    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import (
        _apply_candidate_cap,
    )
    from agentic_frameworks.graph.schemas import ProposedCandidate

    candidates = [
        ProposedCandidate(ticker="A1", sector="Energy"),
        ProposedCandidate(ticker="A2", sector="Energy"),
        ProposedCandidate(ticker="A3", sector="Energy"),
        ProposedCandidate(ticker="A4", sector="Energy"),
        ProposedCandidate(ticker="B1", sector="Health Care"),
        ProposedCandidate(ticker="B2", sector="Health Care"),
        ProposedCandidate(ticker="C1", sector="Financials"),
    ]

    capped = _apply_candidate_cap(candidates)

    assert len(capped) == 5
    energy_tickers = [c.ticker for c in capped if c.sector == "Energy"]
    assert len(energy_tickers) == 3
    assert "A4" not in energy_tickers


async def test_candidate_generator_produces_proposed_candidates():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import (
        _build_candidate_generator,
    )

    generator = _build_candidate_generator(TestModel())
    result = await generator.run("propose candidates for: Energy")

    assert isinstance(result.output, list)
    assert all(hasattr(c, "ticker") and hasattr(c, "sector") for c in result.output)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph/test_us_equity_etf_specialist.py::test_apply_candidate_cap_enforces_per_sector_and_total_limits tests/graph/test_us_equity_etf_specialist.py::test_candidate_generator_produces_proposed_candidates -v`
Expected: both FAIL with `ImportError` (`_apply_candidate_cap`/`_build_candidate_generator` don't exist yet).

- [ ] **Step 3: Write minimal implementation**

Append to `src/agentic_frameworks/graph/nodes/us_equity_etf_specialist.py`:

```python
from pydantic_ai import Agent

from agentic_frameworks.graph.schemas import ProposedCandidate

US_GAP_SECTORS = [
    "Health Care",
    "Financials",
    "Energy",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Real Estate",
    "Utilities",
    "Communication Services",
    "Materials",
]

MAX_CANDIDATES_PER_SECTOR = 3
MAX_CANDIDATES_TOTAL = 5

_CANDIDATE_GENERATOR_SYSTEM_PROMPT = """You are a US equity/ETF research assistant for a personal \
long-term portfolio advisor. Given a list of sector gaps in the user's \
current US holdings, propose real, currently-traded US-listed stock or \
ETF tickers that could fill each gap - up to 3 tickers per gap sector.

Only propose tickers you are confident are real, currently-listed US \
securities. Do not propose delisted, speculative, or fictional tickers. \
For each candidate, name the ticker symbol and the gap sector it addresses."""


def _build_candidate_generator(model) -> Agent:
    """Build the candidate-generator Agent.

    retries=2: bounded per CLAUDE.md's LLM cost control rule 6 - never
    left at the library default.
    """
    return Agent(
        model,
        output_type=list[ProposedCandidate],
        system_prompt=_CANDIDATE_GENERATOR_SYSTEM_PROMPT,
        retries=2,
    )


def _apply_candidate_cap(candidates: list[ProposedCandidate]) -> list[ProposedCandidate]:
    """Enforce the hard cap in code - never trust the model's own count."""
    per_sector_counts: dict[str, int] = {}
    capped: list[ProposedCandidate] = []
    for candidate in candidates:
        per_sector_counts.setdefault(candidate.sector, 0)
        if per_sector_counts[candidate.sector] < MAX_CANDIDATES_PER_SECTOR:
            capped.append(candidate)
            per_sector_counts[candidate.sector] += 1
    return capped[:MAX_CANDIDATES_TOTAL]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph/test_us_equity_etf_specialist.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/nodes/us_equity_etf_specialist.py tests/graph/test_us_equity_etf_specialist.py
git commit -m "feat: add candidate generator and hard cap enforcement"
```

---

### Task 7: Conviction Gate evaluator

**Files:**
- Modify: `src/agentic_frameworks/graph/nodes/us_equity_etf_specialist.py`
- Modify: `tests/graph/test_us_equity_etf_specialist.py`

**Interfaces:**
- Consumes: `get_specialist_model()` (Task 2), `get_yfinance_toolset()` (Task 3), `CandidateEvaluation` (Task 1).
- Produces: `_build_conviction_gate_evaluator(model, toolset) -> Agent` — Task 8's node composition calls this once, then calls `.run(...)` per surviving candidate.

- [ ] **Step 1: Write the failing test**

Append to `tests/graph/test_us_equity_etf_specialist.py`:

```python
async def test_conviction_gate_evaluator_produces_candidate_evaluation():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset
    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import (
        _build_conviction_gate_evaluator,
    )

    toolset = get_yfinance_toolset()
    async with toolset:
        evaluator = _build_conviction_gate_evaluator(TestModel(), toolset)
        result = await evaluator.run("Evaluate ticker: XOM, gap sector: Energy")

        assert result.output.action in {"Buy", "Hold", "Sell", "Reduce"}
        assert isinstance(result.output.confidence, float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph/test_us_equity_etf_specialist.py::test_conviction_gate_evaluator_produces_candidate_evaluation -v`
Expected: FAIL with `ImportError` (`_build_conviction_gate_evaluator` doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Append to `src/agentic_frameworks/graph/nodes/us_equity_etf_specialist.py`:

```python
from agentic_frameworks.graph.schemas import CandidateEvaluation

_CONVICTION_GATE_SYSTEM_PROMPT = """You are evaluating one US equity/ETF new-buy candidate for a \
personal, long-term portfolio advisor, using the New-Buy Conviction \
Gate. Run these four checks in order, stopping at the first failure:

1. Portfolio-fit: does this candidate address a real, identified gap \
in the user's current holdings?
2. Fundamentals: use your tools to check real financials (revenue, \
earnings, debt, margins) - not assumptions.
3. Valuation: is it reasonably priced now versus its own history and \
peers? Richly-valued candidates should be flagged or rejected, not \
rationalized.
4. Style-fit: is it consistent with a long-term, buy-and-hold \
investing approach - no day-trading or speculative framing.

Avoid is the default, not buy: if any check fails, the action should \
reflect that (e.g. Hold or Reduce, not Buy), with the specific reason \
in risk_flags. This is advisory only - your output is never a trade \
instruction, only a recommendation for the user to act on manually."""


def _build_conviction_gate_evaluator(model, toolset: MCPToolset) -> Agent:
    """Build the Conviction Gate evaluator Agent.

    toolsets=[toolset] gives this agent real fundamentals/valuation
    data access via yfinance-mcp for the fundamentals/valuation checks.
    retries=2: bounded per CLAUDE.md's LLM cost control rule 6.
    """
    return Agent(
        model,
        output_type=CandidateEvaluation,
        system_prompt=_CONVICTION_GATE_SYSTEM_PROMPT,
        toolsets=[toolset],
        retries=2,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph/test_us_equity_etf_specialist.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/nodes/us_equity_etf_specialist.py tests/graph/test_us_equity_etf_specialist.py
git commit -m "feat: add Conviction Gate evaluator agent"
```

---

### Task 8: Node composition (`us_equity_etf_specialist_node`)

**Files:**
- Modify: `src/agentic_frameworks/graph/nodes/us_equity_etf_specialist.py`
- Modify: `tests/graph/test_us_equity_etf_specialist.py`

**Interfaces:**
- Consumes: everything from Tasks 1-7 in this same file: `get_specialist_model`, `get_yfinance_toolset`, `_build_candidate_generator`, `_apply_candidate_cap`, `_verify_ticker`, `_build_conviction_gate_evaluator`, `US_GAP_SECTORS`.
- Produces: `us_equity_etf_specialist_node(state: dict) -> dict` — the public entry point. No other task in this plan builds on this one; this completes the node. A future phase wires it into `build_graph()`.

- [ ] **Step 1: Write the failing test**

Append to `tests/graph/test_us_equity_etf_specialist.py`:

```python
async def test_us_equity_etf_specialist_node_returns_candidate_evaluations():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes import us_equity_etf_specialist as specialist_module

    def fake_get_specialist_model():
        return TestModel()

    original = specialist_module.get_specialist_model
    specialist_module.get_specialist_model = fake_get_specialist_model
    try:
        result = await specialist_module.us_equity_etf_specialist_node({})
    finally:
        specialist_module.get_specialist_model = original

    assert "us_equity_etf_candidates" in result
    assert isinstance(result["us_equity_etf_candidates"], list)
    for candidate in result["us_equity_etf_candidates"]:
        assert candidate["action"] in {"Buy", "Hold", "Sell", "Reduce"}
```

This monkeypatches the module-level `get_specialist_model` (module-attribute pattern, not a direct import into the node function, so the patch actually takes effect — same lesson learned in the Research node's fail-loud test) to substitute `TestModel()` for the real OpenRouter-backed model, keeping this test free while still exercising the real `yfinance-mcp` ticker verification and tool calls.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph/test_us_equity_etf_specialist.py::test_us_equity_etf_specialist_node_returns_candidate_evaluations -v`
Expected: FAIL with `AttributeError` (`us_equity_etf_specialist_node` doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Append to `src/agentic_frameworks/graph/nodes/us_equity_etf_specialist.py`:

```python
from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset
from agentic_frameworks.graph.models import get_specialist_model


async def us_equity_etf_specialist_node(state: dict) -> dict:
    """Propose, verify, and evaluate US equity/ETF new-buy candidates.

    Fails loud: any LLM/tool/verification error propagates uncaught.
    Cost control: reject-early (ticker verification before the Conviction
    Gate evaluator ever runs), hard caps enforced in code, minimal
    per-candidate context, bounded retries on both agents.
    """
    model = get_specialist_model()
    toolset = get_yfinance_toolset()

    async with toolset:
        generator = _build_candidate_generator(model)
        gap_sectors_text = ", ".join(US_GAP_SECTORS)
        generation_result = await generator.run(
            f"Propose candidates for these gap sectors: {gap_sectors_text}"
        )
        capped_candidates = _apply_candidate_cap(generation_result.output)

        verified_candidates = [
            candidate
            for candidate in capped_candidates
            if await _verify_ticker(candidate.ticker, toolset)
        ]

        evaluator = _build_conviction_gate_evaluator(model, toolset)
        evaluations = []
        for candidate in verified_candidates:
            eval_result = await evaluator.run(
                f"Evaluate ticker: {candidate.ticker}, gap sector: {candidate.sector}"
            )
            evaluations.append(eval_result.output.model_dump())

    return {"us_equity_etf_candidates": evaluations}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph/test_us_equity_etf_specialist.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/nodes/us_equity_etf_specialist.py tests/graph/test_us_equity_etf_specialist.py
git commit -m "feat: add us_equity_etf_specialist_node composing the full pipeline"
```

---

### Task 9: Full-suite verification and manual real-model phase gate

**Files:** none created or modified by the automated steps — the manual step produces no permanent file (a throwaway script or one-off command, not committed).

**Interfaces:** none — this task consumes everything from Tasks 1-8 to confirm the phase is genuinely done.

- [ ] **Step 1: Run the entire automated test suite**

Run: `uv run pytest tests/ -v`
Expected: all tests from Tasks 1-8 PASS, plus every pre-existing test from Phases 1-3 still PASS.

- [ ] **Step 2: Confirm `portfolio.db` is untouched**

This phase never touches `portfolio.db` (it reads `docs/portfolio-gap-analysis.md`, a static file, and calls `yfinance-mcp`, unrelated to the portfolio database). Confirm its modified timestamp is identical before/after Step 1, same check as every prior phase.

- [ ] **Step 3: Run the required manual real-model check**

This is the phase-completion gate named in the spec and cannot be skipped or automated away. With a real `OPENROUTER_API_KEY` set in the environment (not a fake test key), run:

```bash
uv run python -c "
import asyncio
from agentic_frameworks.graph.nodes.us_equity_etf_specialist import us_equity_etf_specialist_node

result = asyncio.run(us_equity_etf_specialist_node({}))
import json
print(json.dumps(result, indent=2))
"
```

Capture the full output. This calls the real `deepseek/deepseek-v4.1-flash` model via OpenRouter for real - review the output for: are the proposed tickers real, sensible candidates for the named gap sectors? Does each evaluation's `action`/`confidence`/`reasoning`/`risk_flags` look like genuine reasoning, not garbage or a schema-satisfying but content-free response? Report this output to Abhi for review before proceeding - this is a human sign-off step per the spec, not a pass/fail an implementer can self-certify.

- [ ] **Step 4: Report phase completion**

This phase (US Equity/ETF Specialist) is done once Steps 1-3 all pass **and** Abhi has reviewed and approved the real-model output from Step 3. Per `CLAUDE.md`'s phased-development rule, do not begin planning Phase 5 until this step is confirmed complete and reported back.
