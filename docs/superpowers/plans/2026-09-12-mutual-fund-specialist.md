# Mutual Fund Specialist Nodes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `mutual_fund_holdings_analyzer_node` (judges the real 9 existing IN mutual fund holdings) and `mutual_fund_specialist_node` (proposes and evaluates new-buy fund candidates), backed by a custom-built MCP server wrapping `MFapi.in`.

**Architecture:** A new, custom MCP server (`mfapi_mcp`) supplies raw scheme search + NAV history. A shared helpers module computes real, pure metrics from that raw data (scheme resolution, rolling-window returns, volatility, max drawdown, track-record length, benchmark difference, cross-fund redundancy correlation) — mirroring the Risk Analyst's measure-not-judge pattern. Two single-purpose LLM nodes then apply a redefined 7-check Conviction Gate to those real numbers: one judging existing holdings, one judging new-buy candidates sourced from live category-gap detection.

**Tech Stack:** `mcp.server.fastmcp.FastMCP` (official MCP SDK, already installed — no new dependency), `pydantic-ai` (`Agent`, `TestModel`), `yfinance-mcp` (benchmark index data, already integrated), stdlib `statistics` (volatility, correlation — no numpy/pandas needed), `pytest`/`pytest-asyncio`.

**Spec:** `docs/superpowers/specs/2026-09-12-mutual-fund-specialist-design.md`

## Global Constraints

- Fail loud everywhere — no `try`/`except` swallowing errors, except `pydantic-ai`'s bounded `Agent(retries=2)`.
- The new `mfapi` MCP toolset uses `tool_error_behavior="retry"` from the start (Phase 5's lesson applied proactively).
- No fabricated data: `_resolve_scheme_code` raises if no ISIN match is found — never guesses a "close enough" scheme. Benchmark-unavailable categories/funds return `None` with the reason preserved, never a fabricated number.
- Hard caps (3 candidates/category, 5/run) enforced in code for new-buy generation, reusing the existing shared cap logic (generalized, not duplicated).
- Automated tests never spend real money: `TestModel` for every LLM-calling agent; real, free calls to `MFapi.in` and `yfinance-mcp` are used exactly as in every prior phase.
- One real, deliberate, cost-bounded manual run against the actual model pair, reviewed by the user, is required before this phase counts as done.
- Reuse project venv for all commands: `.venv/Scripts/python.exe`.
- All real data referenced in tasks below (ISINs, scheme codes, benchmark tickers) was live-verified this session against the actual `portfolio.db`, `MFapi.in`, and `yfinance-mcp` — use these exact values, do not re-derive or guess alternates.

---

### Task 1: Generalize shared candidate-cap/filter helpers for a configurable grouping key

**Files:**
- Modify: `src/agentic_frameworks/graph/nodes/_specialist_helpers.py`
- Test: Modify `tests/graph/test_specialist_helpers.py`

**Interfaces:**
- Produces: `_apply_candidate_cap(candidates, key=lambda c: c.sector)`, `_filter_to_gap_sectors(candidates, gap_sectors, key=lambda c: c.sector)` — the default `key` preserves current behavior exactly (used unchanged by `us_equity_etf_specialist.py`/`india_equity_specialist.py`); Task 13's `mutual_fund_specialist_node` will call both with `key=lambda c: c.category`.

- [ ] **Step 1: Write failing tests for the new `key` parameter**

Add to `tests/graph/test_specialist_helpers.py`:

```python
def test_apply_candidate_cap_supports_custom_grouping_key():
    from agentic_frameworks.graph.nodes._specialist_helpers import (
        _apply_candidate_cap,
    )

    class FakeCandidate:
        def __init__(self, name, category):
            self.name = name
            self.category = category

    candidates = [
        FakeCandidate("A1", "Large Cap"),
        FakeCandidate("A2", "Large Cap"),
        FakeCandidate("A3", "Large Cap"),
        FakeCandidate("A4", "Large Cap"),
        FakeCandidate("B1", "Gilt"),
    ]

    capped = _apply_candidate_cap(candidates, key=lambda c: c.category)

    assert len(capped) == 4
    large_cap_names = [c.name for c in capped if c.category == "Large Cap"]
    assert len(large_cap_names) == 3
    assert "A4" not in large_cap_names


def test_filter_to_gap_sectors_supports_custom_grouping_key():
    from agentic_frameworks.graph.nodes._specialist_helpers import (
        _filter_to_gap_sectors,
    )

    class FakeCandidate:
        def __init__(self, name, category):
            self.name = name
            self.category = category

    candidates = [
        FakeCandidate("A1", "Gilt"),
        FakeCandidate("B1", "Large Cap"),
    ]

    filtered = _filter_to_gap_sectors(candidates, gap_sectors=["Gilt"], key=lambda c: c.category)

    assert len(filtered) == 1
    assert filtered[0].name == "A1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_specialist_helpers.py -k custom_grouping_key -v`
Expected: FAIL with `TypeError: _apply_candidate_cap() got an unexpected keyword argument 'key'` (and similarly for the filter test).

- [ ] **Step 3: Generalize both functions**

In `src/agentic_frameworks/graph/nodes/_specialist_helpers.py`, replace the two function definitions with:

```python
def _apply_candidate_cap(candidates: list[ProposedCandidate], key=lambda c: c.sector) -> list[ProposedCandidate]:
    """Enforce the hard cap in code - never trust the model's own count.

    key: extracts the grouping value from a candidate (default: .sector,
    used by the US/India equity specialists). The Mutual Fund Specialist
    passes key=lambda c: c.category, since its candidates group by fund
    category rather than GICS sector - the cap logic itself is identical
    either way, only the grouping key differs.
    """
    per_group_counts: dict[str, int] = {}
    capped: list[ProposedCandidate] = []
    for candidate in candidates:
        group = key(candidate)
        per_group_counts.setdefault(group, 0)
        if per_group_counts[group] < MAX_CANDIDATES_PER_SECTOR:
            capped.append(candidate)
            per_group_counts[group] += 1
    return capped[:MAX_CANDIDATES_TOTAL]


def _filter_to_gap_sectors(candidates: list[ProposedCandidate], gap_sectors: list[str], key=lambda c: c.sector) -> list[ProposedCandidate]:
    """Drop any candidate whose group isn't one of this run's live-computed gaps.

    Necessary because ProposedCandidate.sector (and ProposedFundCandidate.category)
    validate against a full taxonomy, not just this run's actual gaps - a
    model can legally propose a candidate for a group the portfolio
    already covers. That candidate must never reach a real verification
    or Conviction Gate call. key: see _apply_candidate_cap.
    """
    gap_set = set(gap_sectors)
    return [candidate for candidate in candidates if key(candidate) in gap_set]
```

- [ ] **Step 4: Run tests to verify they pass, and confirm no regression on existing callers**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_specialist_helpers.py tests/graph/test_us_equity_etf_specialist.py tests/graph/test_india_equity_specialist.py -v`
Expected: all PASS (existing tests call both functions with no `key` argument, exercising the default `lambda c: c.sector` path unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/nodes/_specialist_helpers.py tests/graph/test_specialist_helpers.py
git commit -m "refactor: generalize candidate-cap/filter helpers to a configurable grouping key

Backward-compatible (default key=lambda c: c.sector preserves current
behavior exactly). The upcoming Mutual Fund Specialist needs the same
cap/filter logic grouped by fund category instead of GICS sector -
this avoids a third, duplicated copy of the same logic."
```

---

### Task 2: Custom MFapi MCP server

**Files:**
- Create: `src/agentic_frameworks/mfapi_mcp/__init__.py`
- Create: `src/agentic_frameworks/mfapi_mcp/server.py`
- Modify: `src/agentic_frameworks/graph/mcp_clients.py`
- Test: Create `tests/graph/test_mfapi_mcp.py`

**Interfaces:**
- Produces: `search_scheme(query: str) -> list[dict]` and `get_scheme_nav(scheme_code: int) -> dict` as MCP tools; `get_mfapi_toolset() -> MCPToolset` in `mcp_clients.py` — consumed by Task 4's `_resolve_scheme_code` and Task 5's `_fetch_nav_history`.

- [ ] **Step 1: Write a failing test against the real MFapi.in API through the new server**

Create `tests/graph/test_mfapi_mcp.py`:

```python
"""Tests for the custom mfapi_mcp server, launched as a real subprocess."""

from __future__ import annotations

import json


async def test_search_scheme_finds_a_real_known_fund():
    from agentic_frameworks.graph.mcp_clients import get_mfapi_toolset

    toolset = get_mfapi_toolset()
    async with toolset:
        result = await toolset.wrapped.direct_call_tool("search_scheme", {"query": "Parag Parikh Flexi Cap"})
        candidates = json.loads(result)

    assert isinstance(candidates, list)
    assert any(c["scheme_code"] == 122639 for c in candidates)


async def test_get_scheme_nav_returns_real_meta_and_history():
    from agentic_frameworks.graph.mcp_clients import get_mfapi_toolset

    toolset = get_mfapi_toolset()
    async with toolset:
        result = await toolset.wrapped.direct_call_tool("get_scheme_nav", {"scheme_code": 122639})
        data = json.loads(result)

    assert data["meta"]["isin_growth"] == "INF879O01027"
    assert data["meta"]["scheme_category"] == "Equity Scheme - Flexi Cap Fund"
    assert len(data["data"]) > 1000
    assert "date" in data["data"][0]
    assert "nav" in data["data"][0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_mfapi_mcp.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `get_mfapi_toolset` (doesn't exist yet).

- [ ] **Step 3: Create the server package**

Create `src/agentic_frameworks/mfapi_mcp/__init__.py` (empty file):

```python
```

Create `src/agentic_frameworks/mfapi_mcp/server.py`:

```python
"""A thin, custom MCP server wrapping MFapi.in's search and NAV-history
endpoints for Indian mutual funds.

MFapi.in is a plain REST API with no existing MCP server that met this
project's actual requirements (see docs/superpowers/specs/2026-09-12-
mutual-fund-specialist-design.md's Background/Constraints for the
pre-built-server evaluation). This server exposes exactly two thin,
real operations - the same role yfinance-mcp plays for equities: raw
data only, no analysis. All metrics/scheme-resolution logic lives in
graph/nodes/_mutual_fund_helpers.py, not here.
"""

from __future__ import annotations

import json
import urllib.request

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mfapi")

_BASE_URL = "https://api.mfapi.in"


@mcp.tool()
def search_scheme(query: str) -> str:
    """Search MFapi.in for mutual fund schemes matching a name query.

    Returns a JSON string: a list of {"scheme_code": int, "scheme_name": str}.
    """
    url = f"{_BASE_URL}/mf/search?q={urllib.parse.quote(query)}"
    with urllib.request.urlopen(url, timeout=15) as response:
        raw = json.loads(response.read())
    results = [{"scheme_code": item["schemeCode"], "scheme_name": item["schemeName"]} for item in raw]
    return json.dumps(results)


@mcp.tool()
def get_scheme_nav(scheme_code: int) -> str:
    """Fetch a mutual fund scheme's metadata and full NAV history from MFapi.in.

    Returns a JSON string: {"meta": {...7 fields...}, "data": [{"date": "DD-MM-YYYY", "nav": "..."}, ...]}
    (newest-first, exactly as MFapi.in returns it).
    """
    url = f"{_BASE_URL}/mf/{scheme_code}"
    with urllib.request.urlopen(url, timeout=15) as response:
        raw = json.loads(response.read())
    return json.dumps(raw)


if __name__ == "__main__":
    mcp.run()
```

Add the missing `urllib.parse` import - fix the imports at the top of the file to:

```python
import json
import urllib.parse
import urllib.request
```

- [ ] **Step 4: Wire the client in `mcp_clients.py`**

Add to `src/agentic_frameworks/graph/mcp_clients.py`, after the existing `get_yfinance_toolset` function:

```python
def get_mfapi_toolset() -> MCPToolset:
    """Build the toolset for this project's own custom mfapi_mcp server.

    Launched via `sys.executable -m agentic_frameworks.mfapi_mcp.server` -
    in this project's own venv, since this server is code we wrote
    ourselves (unlike portfolio_data_mcp, a sibling project, or
    yfinance-mcp, an external package launched via uvx).

    tool_error_behavior="retry" from the start - Phase 5's lesson
    (a malformed tool call should let the model self-correct via
    ModelRetry, not crash the node with a raw ToolError) applied
    proactively here rather than re-learned.
    """
    transport = StdioTransport(command=sys.executable, args=["-m", "agentic_frameworks.mfapi_mcp.server"])
    client = FastMCPClient(transport)
    return MCPToolset(client, tool_error_behavior="retry")
```

Add `import sys` to the top of `mcp_clients.py`'s imports if not already present.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_mfapi_mcp.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agentic_frameworks/mfapi_mcp/ src/agentic_frameworks/graph/mcp_clients.py tests/graph/test_mfapi_mcp.py
git commit -m "feat: add custom mfapi_mcp server wrapping MFapi.in

Thin, real wrapper (search_scheme, get_scheme_nav) - no pre-built MFapi
MCP server covered this project's actual requirements (ISIN-matched
resolution, category-gap detection, cross-fund redundancy), and this
picks up the 'build an MCP server yourself' learning goal deferred
from Phase 4's yfinance-mcp reuse decision."
```

---

### Task 3: MF category taxonomy, real-category mapping, and benchmark ticker tables

**Files:**
- Modify: `src/agentic_frameworks/graph/schemas.py`
- Create: `src/agentic_frameworks/graph/nodes/_mutual_fund_helpers.py`
- Test: Modify `tests/graph/test_schemas.py`
- Test: Create `tests/graph/test_mutual_fund_helpers.py`

**Interfaces:**
- Produces: `MF_CATEGORY_TAXONOMY: list[str]` (in `schemas.py`, mirroring where `GICS_SECTORS` already lives — node modules import taxonomy constants FROM `schemas.py`, never the reverse, per this project's established layering), `MF_CATEGORY_TO_TAXONOMY: dict[str, str]`, `MF_CATEGORY_BENCHMARK_TICKERS: dict[str, str | None]`, `INDEX_FUND_BENCHMARK_TICKERS_BY_ISIN: dict[str, str | None]` (all three in `_mutual_fund_helpers.py`) — consumed by Task 4 onward, Task 10's `ProposedFundCandidate`, and both node tasks.

- [ ] **Step 0: Add `MF_CATEGORY_TAXONOMY` to `schemas.py`**

Add to `tests/graph/test_schemas.py`:

```python
def test_mf_category_taxonomy_has_expected_entries():
    from agentic_frameworks.graph.schemas import MF_CATEGORY_TAXONOMY

    assert "Large Cap" in MF_CATEGORY_TAXONOMY
    assert "Gilt" in MF_CATEGORY_TAXONOMY
    assert "Index Fund" in MF_CATEGORY_TAXONOMY
    assert len(MF_CATEGORY_TAXONOMY) == 13
```

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_schemas.py -k mf_category_taxonomy -v`
Expected: FAIL with `ImportError: cannot import name 'MF_CATEGORY_TAXONOMY'`.

Add to `src/agentic_frameworks/graph/schemas.py`, after the existing `GICS_SECTORS` list:

```python
MF_CATEGORY_TAXONOMY = [
    "Large Cap",
    "Mid Cap",
    "Small Cap",
    "Flexi Cap",
    "Multi Cap",
    "ELSS",
    "Aggressive Hybrid",
    "Conservative Hybrid",
    "Liquid",
    "Short Duration",
    "Corporate Bond",
    "Gilt",
    "Index Fund",
]
```

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_schemas.py -k mf_category_taxonomy -v`
Expected: PASS.

- [ ] **Step 1: Write failing tests**

Create `tests/graph/test_mutual_fund_helpers.py`:

```python
"""Tests for the Mutual Fund Specialist nodes' shared helpers."""

from __future__ import annotations


def test_mf_category_to_taxonomy_maps_all_five_real_categories():
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import MF_CATEGORY_TO_TAXONOMY

    assert MF_CATEGORY_TO_TAXONOMY == {
        "Flexi Cap": "Flexi Cap",
        "Hybrid": "Aggressive Hybrid",
        "Index Fund": "Index Fund",
        "Small Cap": "Small Cap",
        "Mid Cap": "Mid Cap",
    }


def test_benchmark_tickers_reflect_live_verification():
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import MF_CATEGORY_BENCHMARK_TICKERS

    assert MF_CATEGORY_BENCHMARK_TICKERS["Large Cap"] == "^NSEI"
    assert MF_CATEGORY_BENCHMARK_TICKERS["Small Cap"] == "NIFTYSMLCAP250.NS"
    assert MF_CATEGORY_BENCHMARK_TICKERS["Mid Cap"] == "NIFTYMIDCAP150.NS"
    assert MF_CATEGORY_BENCHMARK_TICKERS["Liquid"] == "LIQUIDBEES.NS"
    assert MF_CATEGORY_BENCHMARK_TICKERS["Gilt"] is None
    assert MF_CATEGORY_BENCHMARK_TICKERS["Corporate Bond"] is None
    assert MF_CATEGORY_BENCHMARK_TICKERS["Short Duration"] is None
    assert MF_CATEGORY_BENCHMARK_TICKERS["Aggressive Hybrid"] is None
    assert MF_CATEGORY_BENCHMARK_TICKERS["Conservative Hybrid"] is None
    assert "Index Fund" not in MF_CATEGORY_BENCHMARK_TICKERS


def test_index_fund_benchmark_tickers_by_isin_reflect_live_verification():
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
        INDEX_FUND_BENCHMARK_TICKERS_BY_ISIN,
    )

    assert INDEX_FUND_BENCHMARK_TICKERS_BY_ISIN["INF179K01WN9"] == "^BSESN"
    assert INDEX_FUND_BENCHMARK_TICKERS_BY_ISIN["INF789F1AUT5"] == "NIFTY200MOMENTM30.NS"
    assert INDEX_FUND_BENCHMARK_TICKERS_BY_ISIN["INF754K01QD2"] is None
    assert INDEX_FUND_BENCHMARK_TICKERS_BY_ISIN["INF789F1AYN0"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_mutual_fund_helpers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_frameworks.graph.nodes._mutual_fund_helpers'`.

- [ ] **Step 3: Create the module with the taxonomy and mappings**

Create `src/agentic_frameworks/graph/nodes/_mutual_fund_helpers.py`:

```python
"""Shared helpers for the Mutual Fund Specialist nodes
(mutual_fund_holdings_analyzer.py, mutual_fund_specialist.py).

Curated taxonomy, real-category mapping, and live-verified benchmark
ticker tables. Scheme resolution, NAV fetching, and metrics computation
are added in later tasks in this same file.
"""

from __future__ import annotations

from agentic_frameworks.graph.schemas import MF_CATEGORY_TAXONOMY

MF_CATEGORY_TO_TAXONOMY = {
    "Flexi Cap": "Flexi Cap",
    "Hybrid": "Aggressive Hybrid",
    "Index Fund": "Index Fund",
    "Small Cap": "Small Cap",
    "Mid Cap": "Mid Cap",
}

MF_CATEGORY_BENCHMARK_TICKERS = {
    "Large Cap": "^NSEI",
    "Flexi Cap": "^CRSLDX",
    "Multi Cap": "^CRSLDX",
    "ELSS": "^CRSLDX",
    "Small Cap": "NIFTYSMLCAP250.NS",
    "Mid Cap": "NIFTYMIDCAP150.NS",
    "Liquid": "LIQUIDBEES.NS",
    "Short Duration": None,
    "Corporate Bond": None,
    "Gilt": None,
    "Aggressive Hybrid": None,
    "Conservative Hybrid": None,
}

INDEX_FUND_BENCHMARK_TICKERS_BY_ISIN = {
    "INF179K01WN9": "^BSESN",
    "INF789F1AUT5": "NIFTY200MOMENTM30.NS",
    "INF754K01QD2": None,
    "INF789F1AYN0": None,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_mutual_fund_helpers.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/schemas.py src/agentic_frameworks/graph/nodes/_mutual_fund_helpers.py tests/graph/test_schemas.py tests/graph/test_mutual_fund_helpers.py
git commit -m "feat: add curated MF category taxonomy and live-verified benchmark tickers

MF_CATEGORY_TAXONOMY lives in schemas.py alongside GICS_SECTORS,
following this project's established layering (node modules import
taxonomy constants from schemas.py, never the reverse). Benchmark
tickers live-verified via yfinance-mcp this session: exact-index
matches for 2 of 4 existing index funds, category-proxy matches for
the equity categories, confirmed unavailable for the hybrid and debt
categories. Index funds get their own per-ISIN mapping since their
exact tracked index varies per fund, not per category."
```

---

### Task 4: ISIN-matched scheme-code resolution

**Files:**
- Modify: `src/agentic_frameworks/graph/nodes/_mutual_fund_helpers.py`
- Test: Modify `tests/graph/test_mutual_fund_helpers.py`

**Interfaces:**
- Consumes: `get_mfapi_toolset` (`agentic_frameworks.graph.mcp_clients`, Task 2).
- Produces: `async def _resolve_scheme_code(name: str, isin: str, toolset) -> int` (raises `ValueError` if no match found) — consumed by Task 12/13's node pipelines.

- [ ] **Step 1: Write failing tests against real holdings**

Add to `tests/graph/test_mutual_fund_helpers.py`:

```python
async def test_resolve_scheme_code_matches_real_fund_by_isin():
    from agentic_frameworks.graph.mcp_clients import get_mfapi_toolset
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import _resolve_scheme_code

    toolset = get_mfapi_toolset()
    async with toolset:
        code = await _resolve_scheme_code(
            "Parag Parikh Flexi Cap Fund - Direct Plan Growth", "INF879O01027", toolset
        )

    assert code == 122639


async def test_resolve_scheme_code_handles_letter_digit_spacing_mismatch():
    """Real edge case: the portfolio's stored name has 'Nifty200' (no
    space) while MFapi.in stores 'Nifty 200' (with space) for this
    specific fund - confirmed live this session. The resolver must
    normalize this, not just try the raw name.
    """
    from agentic_frameworks.graph.mcp_clients import get_mfapi_toolset
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import _resolve_scheme_code

    toolset = get_mfapi_toolset()
    async with toolset:
        code = await _resolve_scheme_code(
            "UTI Nifty200 Momentum 30 Index Fund - Direct Plan (Demat)", "INF789F1AUT5", toolset
        )

    assert code == 148703


async def test_resolve_scheme_code_raises_on_no_match():
    from agentic_frameworks.graph.mcp_clients import get_mfapi_toolset
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import _resolve_scheme_code

    toolset = get_mfapi_toolset()
    async with toolset:
        with __import__("pytest").raises(ValueError):
            await _resolve_scheme_code(
                "Completely Fictional Nonexistent Mutual Fund XYZ", "INF000000000", toolset
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_mutual_fund_helpers.py -k resolve_scheme_code -v`
Expected: FAIL with `ImportError: cannot import name '_resolve_scheme_code'`.

- [ ] **Step 3: Implement the resolver**

Add to `src/agentic_frameworks/graph/nodes/_mutual_fund_helpers.py`:

```python
import json
import re


def _normalize_letter_digit_spacing(name: str) -> str:
    """Insert a space between a letter and an immediately-following digit.

    Real edge case, live-verified this session: the portfolio's stored
    name for one fund is 'Nifty200' (no space) while MFapi.in stores
    'Nifty 200' (with space) - a single data-entry quirk, not a
    systematic pattern (every other fund's name already has correct
    spacing). This normalization is tried as a fallback, not the
    primary query, since the primary (raw) name resolves 8 of 9 real
    holdings correctly on its own.
    """
    return re.sub(r"([a-zA-Z])(\d)", r"\1 \2", name)


async def _search_scheme(query: str, toolset) -> list[dict]:
    """toolset here is the plain MCPToolset returned by get_mfapi_toolset()
    (not a FilteredToolset like the yfinance toolset), so it exposes
    direct_call_tool directly - no .wrapped indirection needed, unlike
    calls against the yfinance toolset elsewhere in this file."""
    result = await toolset.direct_call_tool("search_scheme", {"query": query})
    return json.loads(result)


async def _get_scheme_isin(scheme_code: int, toolset) -> str | None:
    result = await toolset.direct_call_tool("get_scheme_nav", {"scheme_code": scheme_code})
    data = json.loads(result)
    return data["meta"].get("isin_growth")


async def _resolve_scheme_code(name: str, isin: str, toolset) -> int:
    """Deterministically resolve a fund name + ISIN to its AMFI scheme code.

    MFapi.in rejects direct ISIN lookups and its name search returns
    multiple near-identical variants (Direct/Regular plan, Growth/IDCW
    option) - so this never trusts a name match alone. It tries the
    raw name, then progressively shorter prefixes, then the same
    progression on a letter-digit-normalized name, and only accepts a
    candidate whose own meta.isin_growth matches our known ISIN exactly.
    Raises ValueError (fail loud) if no candidate matches after every
    attempt - never falls back to "the first result" or a fuzzy guess.
    """
    for candidate_name in (name, _normalize_letter_digit_spacing(name)):
        words = candidate_name.split()
        for n_words in (len(words), 4, 3, 2):
            query = " ".join(words[:n_words])
            candidates = await _search_scheme(query, toolset)
            for candidate in candidates:
                candidate_isin = await _get_scheme_isin(candidate["scheme_code"], toolset)
                if candidate_isin == isin:
                    return candidate["scheme_code"]
            if n_words <= 2:
                break
    raise ValueError(f"No MFapi.in scheme found matching name={name!r} isin={isin!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_mutual_fund_helpers.py -k resolve_scheme_code -v`
Expected: all PASS. (Real, unmocked calls to the live `MFapi.in` API via the `mfapi_mcp` server.)

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/nodes/_mutual_fund_helpers.py tests/graph/test_mutual_fund_helpers.py
git commit -m "feat: add ISIN-matched mutual fund scheme resolution

Deterministic, not fuzzy: always verifies a candidate's own ISIN
before accepting it, with progressively-shortened and letter-digit-
normalized query fallbacks to handle real naming mismatches between
the portfolio's stored fund names and MFapi.in's scheme names -
confirmed live against all 9 real holdings this session, including
one genuine spacing-mismatch edge case."
```

---

### Task 5: NAV history fetching and parsing

**Files:**
- Modify: `src/agentic_frameworks/graph/nodes/_mutual_fund_helpers.py`
- Test: Modify `tests/graph/test_mutual_fund_helpers.py`

**Interfaces:**
- Produces: `async def _fetch_nav_history(scheme_code: int, toolset) -> list[dict]` (each entry `{"date": date, "nav": float}`, chronological ascending) — consumed by Task 6's `_compute_fund_metrics` and Task 7/8.

- [ ] **Step 1: Write failing test**

Add to `tests/graph/test_mutual_fund_helpers.py`:

```python
async def test_fetch_nav_history_returns_chronological_real_data():
    from datetime import date

    from agentic_frameworks.graph.mcp_clients import get_mfapi_toolset
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import _fetch_nav_history

    toolset = get_mfapi_toolset()
    async with toolset:
        history = await _fetch_nav_history(122639, toolset)

    assert len(history) > 1000
    assert all(isinstance(h["date"], date) for h in history)
    assert all(isinstance(h["nav"], float) for h in history)
    assert history[0]["date"] < history[-1]["date"]
    assert history[0]["date"] == date(2013, 5, 28)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_mutual_fund_helpers.py -k fetch_nav_history -v`
Expected: FAIL with `ImportError: cannot import name '_fetch_nav_history'`.

- [ ] **Step 3: Implement the fetcher**

Add to `src/agentic_frameworks/graph/nodes/_mutual_fund_helpers.py`:

```python
from datetime import datetime


def _parse_mfapi_date(date_str: str) -> "date":
    return datetime.strptime(date_str, "%d-%m-%Y").date()


async def _fetch_nav_history(scheme_code: int, toolset) -> list[dict]:
    """Fetch a scheme's full NAV history, parsed and sorted chronologically.

    MFapi.in returns newest-first; this re-sorts ascending since every
    downstream computation (CAGR, resampling, drawdown) reasons forward
    in time.
    """
    result = await toolset.direct_call_tool("get_scheme_nav", {"scheme_code": scheme_code})
    raw = json.loads(result)
    history = [
        {"date": _parse_mfapi_date(entry["date"]), "nav": float(entry["nav"])}
        for entry in raw["data"]
    ]
    return sorted(history, key=lambda h: h["date"])
```

Add `from datetime import date` to the top-level imports (needed for the type hint used elsewhere in this file going forward) - update the existing `from datetime import datetime` line to `from datetime import date, datetime`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_mutual_fund_helpers.py -k fetch_nav_history -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/nodes/_mutual_fund_helpers.py tests/graph/test_mutual_fund_helpers.py
git commit -m "feat: add chronological NAV history fetching"
```

---

### Task 6: Pure fund metrics computation (returns, volatility, drawdown, track record)

**Files:**
- Modify: `src/agentic_frameworks/graph/nodes/_mutual_fund_helpers.py`
- Test: Modify `tests/graph/test_mutual_fund_helpers.py`

**Interfaces:**
- Consumes: NAV history shape from Task 5 (`list[{"date": date, "nav": float}]`).
- Produces: `_resample_month_end(history: list[dict]) -> list[dict]`, `_compute_fund_metrics(history: list[dict]) -> dict` (keys: `track_record_years`, `cagr_1y`, `cagr_3y`, `cagr_since_inception`, `volatility`, `max_drawdown`) — pure, no LLM, no MCP call. Consumed by Task 12/13's nodes and Task 8's redundancy computation.

- [ ] **Step 1: Write failing tests with a synthetic, hand-computable NAV series**

Add to `tests/graph/test_mutual_fund_helpers.py`:

```python
def test_compute_fund_metrics_on_synthetic_series():
    from datetime import date

    from agentic_frameworks.graph.nodes._mutual_fund_helpers import _compute_fund_metrics

    # Exactly 2 years, NAV doubles from 100 to 200 - CAGR should be
    # sqrt(2) - 1 ~= 41.42%. No drawdown (monotonic increase).
    history = [
        {"date": date(2024, 1, 1), "nav": 100.0},
        {"date": date(2024, 7, 1), "nav": 130.0},
        {"date": date(2025, 1, 1), "nav": 150.0},
        {"date": date(2025, 7, 1), "nav": 170.0},
        {"date": date(2026, 1, 1), "nav": 200.0},
    ]

    metrics = _compute_fund_metrics(history)

    assert metrics["track_record_years"] == 2.0
    assert abs(metrics["cagr_since_inception"] - (2 ** 0.5 - 1)) < 0.01
    assert metrics["cagr_1y"] is not None
    assert metrics["max_drawdown"] == 0.0


def test_compute_fund_metrics_detects_a_real_drawdown():
    from datetime import date

    from agentic_frameworks.graph.nodes._mutual_fund_helpers import _compute_fund_metrics

    history = [
        {"date": date(2024, 1, 1), "nav": 100.0},
        {"date": date(2024, 6, 1), "nav": 150.0},
        {"date": date(2024, 9, 1), "nav": 90.0},
        {"date": date(2025, 1, 1), "nav": 120.0},
    ]

    metrics = _compute_fund_metrics(history)

    assert abs(metrics["max_drawdown"] - (90.0 / 150.0 - 1)) < 0.001


def test_compute_fund_metrics_returns_none_cagr_when_period_exceeds_history():
    from datetime import date

    from agentic_frameworks.graph.nodes._mutual_fund_helpers import _compute_fund_metrics

    history = [
        {"date": date(2025, 1, 1), "nav": 100.0},
        {"date": date(2025, 6, 1), "nav": 110.0},
    ]

    metrics = _compute_fund_metrics(history)

    assert metrics["cagr_3y"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_mutual_fund_helpers.py -k compute_fund_metrics -v`
Expected: FAIL with `ImportError: cannot import name '_compute_fund_metrics'`.

- [ ] **Step 3: Implement the metrics functions**

Add to `src/agentic_frameworks/graph/nodes/_mutual_fund_helpers.py`:

```python
import statistics
from datetime import timedelta


def _resample_month_end(history: list[dict]) -> list[dict]:
    """Pick the last NAV entry on/before each month boundary present in the series.

    history must already be sorted ascending by date (as _fetch_nav_history
    returns it). Used for volatility and cross-fund correlation, so both
    are computed on a consistent, lower-noise monthly cadence rather than
    raw daily NAV entries.
    """
    by_month: dict[tuple[int, int], dict] = {}
    for entry in history:
        key = (entry["date"].year, entry["date"].month)
        by_month[key] = entry
    return [by_month[key] for key in sorted(by_month.keys())]


def _find_nav_on_or_before(history: list[dict], target_date) -> dict | None:
    candidates = [h for h in history if h["date"] <= target_date]
    return candidates[-1] if candidates else None


def _compute_fund_metrics(history: list[dict]) -> dict:
    """Pure computation from real NAV history - no LLM, no fabrication.

    Returns raw numbers only, mirroring the Risk Analyst's measure-not-
    judge pattern: track_record_years, cagr_1y/cagr_3y (None if the
    fund's history doesn't span that period), cagr_since_inception,
    volatility (annualized stdev of monthly returns), max_drawdown
    (a non-positive fraction, e.g. -0.4 for a 40% peak-to-trough decline).
    """
    latest = history[-1]
    oldest = history[0]
    track_record_years = (latest["date"] - oldest["date"]).days / 365.25

    def cagr_over_years(years: float) -> float | None:
        target_date = latest["date"] - timedelta(days=int(years * 365.25))
        if target_date < oldest["date"]:
            return None
        start = _find_nav_on_or_before(history, target_date)
        if start is None or start["nav"] <= 0:
            return None
        return (latest["nav"] / start["nav"]) ** (1 / years) - 1

    cagr_since_inception = (
        (latest["nav"] / oldest["nav"]) ** (1 / track_record_years) - 1
        if track_record_years > 0
        else 0.0
    )

    monthly = _resample_month_end(history)
    monthly_returns = [
        (monthly[i]["nav"] / monthly[i - 1]["nav"]) - 1
        for i in range(1, len(monthly))
        if monthly[i - 1]["nav"] > 0
    ]
    volatility = statistics.stdev(monthly_returns) * (12 ** 0.5) if len(monthly_returns) >= 2 else 0.0

    peak = history[0]["nav"]
    max_drawdown = 0.0
    for entry in history:
        peak = max(peak, entry["nav"])
        drawdown = (entry["nav"] - peak) / peak
        max_drawdown = min(max_drawdown, drawdown)

    return {
        "track_record_years": round(track_record_years, 2),
        "cagr_1y": cagr_over_years(1),
        "cagr_3y": cagr_over_years(3),
        "cagr_since_inception": cagr_since_inception,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_mutual_fund_helpers.py -k compute_fund_metrics -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/nodes/_mutual_fund_helpers.py tests/graph/test_mutual_fund_helpers.py
git commit -m "feat: add pure fund metrics computation (returns, volatility, drawdown)

No numpy/pandas needed - stdlib statistics.stdev covers volatility.
Pure function, no LLM/MCP call, mirrors the Risk Analyst's
measure-not-judge pattern."
```

---

### Task 7: Benchmark price-history parsing and benchmark-difference computation

**Files:**
- Modify: `src/agentic_frameworks/graph/nodes/_mutual_fund_helpers.py`
- Test: Modify `tests/graph/test_mutual_fund_helpers.py`

**Interfaces:**
- Consumes: `get_yfinance_toolset` (`agentic_frameworks.graph.mcp_clients`), `MF_CATEGORY_BENCHMARK_TICKERS`/`INDEX_FUND_BENCHMARK_TICKERS_BY_ISIN` (Task 3), NAV history shape (Task 5).
- Produces: `_parse_yfinance_price_history(text: str) -> list[dict]`, `async def _compute_benchmark_difference(nav_history: list[dict], benchmark_ticker: str | None, yfinance_toolset) -> dict | None` — consumed by Task 12/13's nodes.

- [ ] **Step 1: Write failing tests**

Add to `tests/graph/test_mutual_fund_helpers.py`:

```python
def test_parse_yfinance_price_history_extracts_date_and_close():
    from datetime import date

    from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
        _parse_yfinance_price_history,
    )

    raw = (
        "| Date                      |    Open |    High |     Low |   Close |   Volume |   Dividends |   Stock Splits |\n"
        "|:--------------------------|--------:|--------:|--------:|--------:|---------:|------------:|---------------:|\n"
        "| 2026-07-01 00:00:00+05:30 | 76545.2 | 78664.9 | 75474.4 | 78094.6 |   430400 |           0 |              0 |\n"
        "| 2026-08-01 00:00:00+05:30 | 78883.3 | 79143.1 | 76751.3 | 76957.3 |   319600 |           0 |              0 |"
    )

    parsed = _parse_yfinance_price_history(raw)

    assert parsed == [
        {"date": date(2026, 7, 1), "close": 78094.6},
        {"date": date(2026, 8, 1), "close": 76957.3},
    ]


async def test_compute_benchmark_difference_returns_none_when_ticker_unavailable():
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
        _compute_benchmark_difference,
    )

    result = await _compute_benchmark_difference([], None, None)

    assert result is None


async def test_compute_benchmark_difference_computes_real_comparison():
    from datetime import date

    from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
        _compute_benchmark_difference,
    )

    fund_history = [
        {"date": date(2023, 1, 1), "nav": 100.0},
        {"date": date(2026, 1, 1), "nav": 130.0},
    ]

    toolset = get_yfinance_toolset()
    async with toolset:
        result = await _compute_benchmark_difference(fund_history, "^NSEI", toolset)

    assert result is not None
    assert "fund_cagr" in result
    assert "benchmark_cagr" in result
    assert "difference" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_mutual_fund_helpers.py -k "parse_yfinance or benchmark_difference" -v`
Expected: FAIL with `ImportError: cannot import name '_parse_yfinance_price_history'`.

- [ ] **Step 3: Implement the parser and benchmark-difference computation**

Add to `src/agentic_frameworks/graph/nodes/_mutual_fund_helpers.py`:

```python
def _parse_yfinance_price_history(text: str) -> list[dict]:
    """Parse yfinance_get_price_history's markdown table output.

    Confirmed live this session: this tool returns a markdown table
    (Date|Open|High|Low|Close|Volume|Dividends|Stock Splits), not JSON
    like yfinance_get_ticker_info - a real API-shape difference this
    parser exists specifically to handle.
    """
    lines = [line for line in text.strip().split("\n") if line.strip()]
    data_lines = lines[2:]  # skip header + markdown separator row
    parsed = []
    for line in data_lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        date_str = cells[0].split(" ")[0]
        parsed.append({"date": date.fromisoformat(date_str), "close": float(cells[4])})
    return parsed


async def _compute_benchmark_difference(nav_history: list[dict], benchmark_ticker: str | None, yfinance_toolset) -> dict | None:
    """Compare a fund's own CAGR against its benchmark's CAGR over the same period.

    Returns None (not a fabricated number) when no verified benchmark
    ticker exists for this fund's category - a real, disclosed
    limitation, not a silently skipped check.
    """
    if benchmark_ticker is None:
        return None

    fund_metrics = _compute_fund_metrics(nav_history)
    years = fund_metrics["track_record_years"]

    raw = await yfinance_toolset.wrapped.direct_call_tool(
        "yfinance_get_price_history", {"symbol": benchmark_ticker, "period": "max", "interval": "1mo"}
    )
    benchmark_history = _parse_yfinance_price_history(raw)

    start_date = nav_history[0]["date"]
    end_date = nav_history[-1]["date"]
    benchmark_start = _find_nav_on_or_before(
        [{"date": h["date"], "nav": h["close"]} for h in benchmark_history], start_date
    )
    benchmark_end = _find_nav_on_or_before(
        [{"date": h["date"], "nav": h["close"]} for h in benchmark_history], end_date
    )
    if benchmark_start is None or benchmark_end is None or years <= 0:
        return None

    benchmark_cagr = (benchmark_end["nav"] / benchmark_start["nav"]) ** (1 / years) - 1
    fund_cagr = fund_metrics["cagr_since_inception"]

    return {
        "fund_cagr": fund_cagr,
        "benchmark_cagr": benchmark_cagr,
        "difference": fund_cagr - benchmark_cagr,
    }
```

Add `from datetime import date` if not already present from Task 5 (it should already be there — verify, don't duplicate the import).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_mutual_fund_helpers.py -k "parse_yfinance or benchmark_difference" -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/nodes/_mutual_fund_helpers.py tests/graph/test_mutual_fund_helpers.py
git commit -m "feat: add benchmark-difference computation via yfinance-mcp

Includes a markdown-table parser for yfinance_get_price_history's
output - confirmed live this session that this tool returns markdown,
not JSON, unlike yfinance_get_ticker_info. Returns None (not a
fabricated number) when no verified benchmark ticker exists."
```

---

### Task 8: Cross-fund redundancy correlation

**Files:**
- Modify: `src/agentic_frameworks/graph/nodes/_mutual_fund_helpers.py`
- Test: Modify `tests/graph/test_mutual_fund_helpers.py`

**Interfaces:**
- Consumes: NAV history shape (Task 5), `_resample_month_end` (Task 6).
- Produces: `_compute_redundancy_correlations(fund_histories: dict[str, list[dict]]) -> dict[str, dict[str, float]]` (keyed by ISIN, symmetric pairwise Pearson correlation of monthly returns) — consumed by Task 12's `mutual_fund_holdings_analyzer_node`.

- [ ] **Step 1: Write failing test with synthetic, hand-verifiable data**

Add to `tests/graph/test_mutual_fund_helpers.py`:

```python
def test_compute_redundancy_correlations_identifies_perfectly_correlated_funds():
    from datetime import date

    from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
        _compute_redundancy_correlations,
    )

    # Fund B's NAV is always exactly 2x Fund A's - perfectly correlated (1.0).
    fund_a = [
        {"date": date(2024, 1, 1), "nav": 100.0},
        {"date": date(2024, 2, 1), "nav": 110.0},
        {"date": date(2024, 3, 1), "nav": 105.0},
        {"date": date(2024, 4, 1), "nav": 120.0},
    ]
    fund_b = [{"date": h["date"], "nav": h["nav"] * 2} for h in fund_a]

    correlations = _compute_redundancy_correlations({"ISIN_A": fund_a, "ISIN_B": fund_b})

    assert abs(correlations["ISIN_A"]["ISIN_B"] - 1.0) < 0.001
    assert abs(correlations["ISIN_B"]["ISIN_A"] - 1.0) < 0.001
    assert "ISIN_A" not in correlations["ISIN_A"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_mutual_fund_helpers.py -k redundancy_correlations -v`
Expected: FAIL with `ImportError: cannot import name '_compute_redundancy_correlations'`.

- [ ] **Step 3: Implement redundancy computation**

Add to `src/agentic_frameworks/graph/nodes/_mutual_fund_helpers.py`:

```python
def _compute_redundancy_correlations(fund_histories: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    """Pairwise Pearson correlation of monthly returns across funds, keyed by ISIN.

    A statistical proxy for portfolio-holdings overlap (which MFapi.in
    doesn't expose) - two funds whose NAV moves nearly identically are
    functionally redundant even if their category labels differ, and
    two funds with the same category label can still genuinely
    diversify if their correlation is low. Symmetric: result[a][b] ==
    result[b][a]. A fund never appears as its own key in its own row.
    """
    monthly_returns: dict[str, dict[tuple[int, int], float]] = {}
    for isin, history in fund_histories.items():
        monthly = _resample_month_end(history)
        returns_by_month = {}
        for i in range(1, len(monthly)):
            if monthly[i - 1]["nav"] > 0:
                key = (monthly[i]["date"].year, monthly[i]["date"].month)
                returns_by_month[key] = (monthly[i]["nav"] / monthly[i - 1]["nav"]) - 1
        monthly_returns[isin] = returns_by_month

    result: dict[str, dict[str, float]] = {isin: {} for isin in fund_histories}
    isins = list(fund_histories.keys())
    for i, isin_a in enumerate(isins):
        for isin_b in isins[i + 1:]:
            common_months = sorted(set(monthly_returns[isin_a]) & set(monthly_returns[isin_b]))
            if len(common_months) < 2:
                continue
            series_a = [monthly_returns[isin_a][m] for m in common_months]
            series_b = [monthly_returns[isin_b][m] for m in common_months]
            correlation = statistics.correlation(series_a, series_b)
            result[isin_a][isin_b] = correlation
            result[isin_b][isin_a] = correlation
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_mutual_fund_helpers.py -k redundancy_correlations -v`
Expected: PASS.

- [ ] **Step 5: Run the full helpers test file to confirm no regressions**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_mutual_fund_helpers.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agentic_frameworks/graph/nodes/_mutual_fund_helpers.py tests/graph/test_mutual_fund_helpers.py
git commit -m "feat: add cross-fund redundancy correlation computation

Statistical proxy for portfolio-holdings overlap, which MFapi.in
doesn't expose. Pure function using stdlib statistics.correlation -
no numpy/pandas needed."
```

---

### Task 9: Live category-gap detection for new-buy candidates

**Files:**
- Modify: `src/agentic_frameworks/graph/nodes/_mutual_fund_helpers.py`
- Test: Modify `tests/graph/test_mutual_fund_helpers.py`

**Interfaces:**
- Consumes: `MF_CATEGORY_TAXONOMY`, `MF_CATEGORY_TO_TAXONOMY` (Task 3).
- Produces: `_compute_mf_category_gaps(holdings_by_bucket: dict) -> list[str]` — consumed by Task 13's `mutual_fund_specialist_node`.

- [ ] **Step 1: Write failing tests against real holdings**

Add to `tests/graph/test_mutual_fund_helpers.py`:

```python
async def test_compute_mf_category_gaps_finds_expected_gaps_given_real_holdings():
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
        _compute_mf_category_gaps,
    )
    from agentic_frameworks.graph.nodes.research import research_node

    research_result = await research_node({})

    gaps = _compute_mf_category_gaps(research_result["holdings_by_bucket"])

    assert set(gaps) == {
        "Large Cap",
        "Multi Cap",
        "ELSS",
        "Conservative Hybrid",
        "Liquid",
        "Short Duration",
        "Corporate Bond",
        "Gilt",
    }


def test_compute_mf_category_gaps_raises_on_unmapped_category():
    import pytest

    from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
        _compute_mf_category_gaps,
    )

    holdings_by_bucket = {
        "mutual_fund": {
            "IN": [{"ticker": "INF000000000", "name": "Fictional Fund", "sector": "Not A Real Category"}],
        }
    }

    with pytest.raises(KeyError):
        _compute_mf_category_gaps(holdings_by_bucket)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_mutual_fund_helpers.py -k mf_category_gaps -v`
Expected: FAIL with `ImportError: cannot import name '_compute_mf_category_gaps'`.

- [ ] **Step 3: Implement gap detection**

Add to `src/agentic_frameworks/graph/nodes/_mutual_fund_helpers.py`:

```python
def _compute_mf_category_gaps(holdings_by_bucket: dict[str, dict[str, list[dict]]]) -> list[str]:
    """Live-compute curated MF categories with zero coverage in real IN mutual fund holdings.

    Mirrors _compute_us_gaps/_compute_india_equity_gaps: any category
    string not in MF_CATEGORY_TO_TAXONOMY raises - fail loud rather
    than silently miscounting.
    """
    covered: set[str] = set()
    for holding in holdings_by_bucket.get("mutual_fund", {}).get("IN", []):
        category = holding["sector"]
        covered.add(MF_CATEGORY_TO_TAXONOMY[category])
    return [category for category in MF_CATEGORY_TAXONOMY if category not in covered]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_mutual_fund_helpers.py -k mf_category_gaps -v`
Expected: all PASS (the real-holdings test hits the live `portfolio_data_mcp` server via `research_node`, free and unmocked, consistent with every other gap-detection test in this project).

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/nodes/_mutual_fund_helpers.py tests/graph/test_mutual_fund_helpers.py
git commit -m "feat: add live MF category-gap detection

Confirms the real gaps given the current 9 holdings: Large Cap, Multi
Cap, ELSS, Conservative Hybrid, and all 4 debt categories (Liquid,
Short Duration, Corporate Bond, Gilt) - zero debt exposure across the
whole mutual fund book."
```

---

### Task 10: `ProposedFundCandidate` schema

**Files:**
- Modify: `src/agentic_frameworks/graph/schemas.py`
- Test: Modify `tests/graph/test_schemas.py`

**Interfaces:**
- Consumes: `MF_CATEGORY_TAXONOMY` (already added to `schemas.py` by Task 3 — no new import needed, it's already in this same file).
- Produces: `ProposedFundCandidate(BaseModel)`: `scheme_name: str`, `category: Literal[*MF_CATEGORY_TAXONOMY]` — consumed by Task 13's candidate generator.

- [ ] **Step 1: Write failing tests**

Add to `tests/graph/test_schemas.py`:

```python
def test_proposed_fund_candidate_requires_scheme_name_and_category():
    from agentic_frameworks.graph.schemas import ProposedFundCandidate

    candidate = ProposedFundCandidate(scheme_name="Some Large Cap Fund Direct Growth", category="Large Cap")

    assert candidate.scheme_name == "Some Large Cap Fund Direct Growth"
    assert candidate.category == "Large Cap"


def test_proposed_fund_candidate_rejects_unknown_category():
    from agentic_frameworks.graph.schemas import ProposedFundCandidate

    with pytest.raises(ValidationError):
        ProposedFundCandidate(scheme_name="Some Fund", category="Not A Real Category")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_schemas.py -k proposed_fund_candidate -v`
Expected: FAIL with `ImportError: cannot import name 'ProposedFundCandidate'`.

- [ ] **Step 3: Add the schema**

In `src/agentic_frameworks/graph/schemas.py`, add after the existing `ProposedCandidate` class (no new import needed — `MF_CATEGORY_TAXONOMY` already lives in this file, added by Task 3):

```python
class ProposedFundCandidate(BaseModel):
    """One mutual fund the candidate generator proposes for a specific gap category.

    category is validated against the full curated MF_CATEGORY_TAXONOMY,
    not a hardcoded gap-subset - same reasoning as ProposedCandidate.sector:
    which categories are actual gaps is computed live at runtime
    (see _compute_mf_category_gaps), never baked into this schema.
    """

    scheme_name: str
    category: Literal[*MF_CATEGORY_TAXONOMY]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_schemas.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/schemas.py tests/graph/test_schemas.py
git commit -m "feat: add ProposedFundCandidate schema for MF new-buy candidates"
```

---

### Task 11: Wire new state fields

**Files:**
- Modify: `src/agentic_frameworks/graph/state.py`
- Test: Modify `tests/graph/test_state.py`

**Interfaces:**
- Produces: `SpecialistState.existing_fund_evaluations: list[dict]`, `SpecialistState.mutual_fund_candidates: list[dict]`, and the same two fields on `GraphState` — consumed by Task 14's `graph.py` wiring.

- [ ] **Step 1: Write failing tests**

Modify `tests/graph/test_state.py`'s `test_specialist_state_has_expected_keys` and `test_graph_state_has_expected_keys`:

```python
def test_specialist_state_has_expected_keys():
    from agentic_frameworks.graph.state import SpecialistState

    hints = get_type_hints(SpecialistState)
    assert set(hints.keys()) == {
        "us_equity_etf_candidates",
        "india_equity_candidates",
        "existing_fund_evaluations",
        "mutual_fund_candidates",
    }
    assert hints["existing_fund_evaluations"] == list[dict]
    assert hints["mutual_fund_candidates"] == list[dict]


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
        "us_equity_etf_candidates",
        "india_equity_candidates",
        "existing_fund_evaluations",
        "mutual_fund_candidates",
    }
```

Add two new tests mirroring the existing `test_graph_state_includes_india_equity_candidates` pattern:

```python
def test_graph_state_includes_existing_fund_evaluations():
    from agentic_frameworks.graph.state import GraphState

    hints = get_type_hints(GraphState)
    assert "existing_fund_evaluations" in hints
    assert hints["existing_fund_evaluations"] == list[dict]


def test_graph_state_includes_mutual_fund_candidates():
    from agentic_frameworks.graph.state import GraphState

    hints = get_type_hints(GraphState)
    assert "mutual_fund_candidates" in hints
    assert hints["mutual_fund_candidates"] == list[dict]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_state.py -v`
Expected: the modified/new tests FAIL (missing keys).

- [ ] **Step 3: Update `state.py`**

In `src/agentic_frameworks/graph/state.py`, modify `SpecialistState`:

```python
class SpecialistState(TypedDict):
    """State the new-buy specialist and holdings-analyzer nodes produce.

    us_equity_etf_candidates: the US Equity/ETF Specialist's evaluated
        candidates, each a `CandidateEvaluation` dict.
    india_equity_candidates: the India Equity Specialist's evaluated
        candidates, each a `CandidateEvaluation` dict.
    existing_fund_evaluations: the Mutual Fund Holdings Analyzer's
        verdicts on the real 9 existing IN mutual fund holdings, each
        a `CandidateEvaluation` dict.
    mutual_fund_candidates: the Mutual Fund Specialist's evaluated
        new-buy candidates, each a `CandidateEvaluation` dict.
    """

    us_equity_etf_candidates: list[dict]
    india_equity_candidates: list[dict]
    existing_fund_evaluations: list[dict]
    mutual_fund_candidates: list[dict]
```

Add the same two fields to `GraphState`, with corresponding docstring lines, after the existing `india_equity_candidates: list[dict]` line:

```python
    existing_fund_evaluations: list[dict]
    mutual_fund_candidates: list[dict]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_state.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/state.py tests/graph/test_state.py
git commit -m "feat: add existing_fund_evaluations and mutual_fund_candidates to shared graph state"
```

---

### Task 12: `mutual_fund_holdings_analyzer_node`

**Files:**
- Create: `src/agentic_frameworks/graph/nodes/mutual_fund_holdings_analyzer.py`
- Test: Create `tests/graph/test_mutual_fund_holdings_analyzer.py`

**Interfaces:**
- Consumes: `get_mfapi_toolset`, `get_yfinance_toolset` (`agentic_frameworks.graph.mcp_clients`), `get_specialist_model` (`agentic_frameworks.graph.models`), `_resolve_scheme_code`, `_fetch_nav_history`, `_compute_fund_metrics`, `_compute_benchmark_difference`, `_compute_redundancy_correlations`, `MF_CATEGORY_TO_TAXONOMY`, `MF_CATEGORY_BENCHMARK_TICKERS`, `INDEX_FUND_BENCHMARK_TICKERS_BY_ISIN` (all from `agentic_frameworks.graph.nodes._mutual_fund_helpers`), `CandidateEvaluation` (`agentic_frameworks.graph.schemas`), `GraphState`/`SpecialistState` (`agentic_frameworks.graph.state`).
- Produces: `async def mutual_fund_holdings_analyzer_node(state: GraphState) -> SpecialistState` — consumed by Task 14's `graph.py`.

- [ ] **Step 1: Write failing tests**

Create `tests/graph/test_mutual_fund_holdings_analyzer.py`:

```python
"""Tests for the Mutual Fund Holdings Analyzer node."""

from __future__ import annotations


def test_lookup_benchmark_ticker_uses_index_fund_isin_mapping_first():
    from agentic_frameworks.graph.nodes.mutual_fund_holdings_analyzer import (
        _lookup_benchmark_ticker,
    )

    ticker = _lookup_benchmark_ticker(isin="INF179K01WN9", category="Index Fund")

    assert ticker == "^BSESN"


def test_lookup_benchmark_ticker_falls_back_to_category_table():
    from agentic_frameworks.graph.nodes.mutual_fund_holdings_analyzer import (
        _lookup_benchmark_ticker,
    )

    ticker = _lookup_benchmark_ticker(isin="INF879O01027", category="Flexi Cap")

    assert ticker == "^CRSLDX"


async def test_conviction_gate_evaluator_produces_candidate_evaluation():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset
    from agentic_frameworks.graph.nodes.mutual_fund_holdings_analyzer import (
        _build_conviction_gate_evaluator,
    )

    toolset = get_yfinance_toolset()
    async with toolset:
        evaluator = _build_conviction_gate_evaluator(TestModel(), toolset)
        result = await evaluator.run(
            "Evaluate existing holding: Parag Parikh Flexi Cap Fund (category: Flexi Cap). "
            "Metrics: 3yr CAGR 18%, volatility 15%, max drawdown -22%, track record 12 years. "
            "Benchmark (Nifty 500) difference: +4% annualized. Redundancy: low correlation with other holdings."
        )

        assert result.output.action in {"Buy", "Hold", "Sell", "Reduce"}
        assert isinstance(result.output.confidence, float)


async def test_mutual_fund_holdings_analyzer_node_returns_one_evaluation_per_real_fund():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes import mutual_fund_holdings_analyzer as node_module
    from agentic_frameworks.graph.nodes.research import research_node

    def fake_get_specialist_model():
        return TestModel()

    research_result = await research_node({})
    state = {"holdings_by_bucket": research_result["holdings_by_bucket"]}

    original = node_module.get_specialist_model
    node_module.get_specialist_model = fake_get_specialist_model
    try:
        result = await node_module.mutual_fund_holdings_analyzer_node(state)
    finally:
        node_module.get_specialist_model = original

    assert "existing_fund_evaluations" in result
    assert len(result["existing_fund_evaluations"]) == 9
    for evaluation in result["existing_fund_evaluations"]:
        assert evaluation["action"] in {"Buy", "Hold", "Sell", "Reduce"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_mutual_fund_holdings_analyzer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_frameworks.graph.nodes.mutual_fund_holdings_analyzer'`.

- [ ] **Step 3: Implement the node**

Create `src/agentic_frameworks/graph/nodes/mutual_fund_holdings_analyzer.py`:

```python
"""The Mutual Fund Holdings Analyzer node: judges the real 9 existing IN
mutual fund holdings against a 7-check Conviction Gate, redefined for
funds (see docs/superpowers/specs/2026-09-12-mutual-fund-specialist-
design.md). Measures first (pure functions, no LLM), judges second
(one LLM evaluator call per fund) - mirrors the Risk Analyst's
measure-not-judge separation. Fails loud, same as every other node.
"""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.toolsets.filtered import FilteredToolset

from agentic_frameworks.graph.mcp_clients import get_mfapi_toolset, get_yfinance_toolset
from agentic_frameworks.graph.models import get_specialist_model
from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
    INDEX_FUND_BENCHMARK_TICKERS_BY_ISIN,
    MF_CATEGORY_BENCHMARK_TICKERS,
    MF_CATEGORY_TO_TAXONOMY,
    _compute_benchmark_difference,
    _compute_fund_metrics,
    _compute_redundancy_correlations,
    _fetch_nav_history,
    _resolve_scheme_code,
)
from agentic_frameworks.graph.schemas import CandidateEvaluation
from agentic_frameworks.graph.state import GraphState, SpecialistState


def _lookup_benchmark_ticker(isin: str, category: str) -> str | None:
    """Index funds resolve via their own ISIN (exact tracked index varies
    per fund, not per category); every other category resolves via the
    shared category-benchmark table."""
    if category == "Index Fund":
        return INDEX_FUND_BENCHMARK_TICKERS_BY_ISIN.get(isin)
    taxonomy_category = MF_CATEGORY_TO_TAXONOMY[category]
    return MF_CATEGORY_BENCHMARK_TICKERS.get(taxonomy_category)


def _format_metrics_for_prompt(metrics: dict, benchmark: dict | None, redundancy: dict[str, float]) -> str:
    lines = [
        f"Track record: {metrics['track_record_years']} years",
        f"CAGR since inception: {metrics['cagr_since_inception']:.1%}",
        f"CAGR 1yr: {metrics['cagr_1y']:.1%}" if metrics["cagr_1y"] is not None else "CAGR 1yr: insufficient history",
        f"CAGR 3yr: {metrics['cagr_3y']:.1%}" if metrics["cagr_3y"] is not None else "CAGR 3yr: insufficient history",
        f"Annualized volatility: {metrics['volatility']:.1%}",
        f"Max drawdown: {metrics['max_drawdown']:.1%}",
    ]
    if benchmark is not None:
        lines.append(
            f"Benchmark comparison: fund CAGR {benchmark['fund_cagr']:.1%} vs "
            f"benchmark CAGR {benchmark['benchmark_cagr']:.1%} "
            f"(difference: {benchmark['difference']:+.1%})"
        )
    else:
        lines.append("Benchmark comparison: unavailable for this fund's category")
    if redundancy:
        top = sorted(redundancy.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        lines.append("Highest correlation with other held funds: " + ", ".join(f"{isin}={corr:.2f}" for isin, corr in top))
    else:
        lines.append("Redundancy: no comparable overlap data")
    return "\n".join(lines)


_CONVICTION_GATE_SYSTEM_PROMPT = """You are evaluating one existing India mutual fund holding for a \
personal, long-term portfolio advisor, using a 7-check Conviction Gate \
adapted for mutual funds (not equities - no price/valuation checks apply). \
Run these checks in order, weighing all of them in your final verdict:

1. Fit/redundancy: does this fund still serve a distinct purpose, or is it \
largely redundant with another holding (see the redundancy correlation data)?
2. Cost: funds held via a Direct plan avoid distributor commission drag - \
treat this as a qualitative positive, not a numeric expense-ratio check \
(exact expense ratios are not available from this project's data sources).
3. Performance vs benchmark: did it beat its benchmark over its own history, \
where a verified benchmark exists? If no benchmark is available for this \
fund's category, say so explicitly and weigh the other checks more heavily \
rather than guessing at benchmark performance.
4. Risk-adjusted performance: consider volatility and max drawdown, not just \
raw returns - a fund beating its benchmark by taking on much more risk isn't \
straightforwardly better for a long-term investor.
5. Cross-fund redundancy: real correlation data with the user's other funds \
is provided - flag genuine overlap.
6. Track-record length: a short track record (under ~3 years) means \
performance-consistency claims should carry less weight in your reasoning.
7. Style-fit: is this fund's mandate consistent with a long-term, \
buy-and-hold approach?

Tax context (India, current rules, for informing hold/reduce/sell timing \
advice only - never invent numbers beyond this): equity-oriented mutual \
funds (>=65% equity) are taxed as equity capital gains; other funds \
(debt-oriented, most hybrid funds) are taxed at the investor's income slab \
rate regardless of holding period; ELSS funds carry a mandatory 3-year \
lock-in from each investment date.

Your output is advisory only - never a trade instruction, only a \
recommendation for the user to act on manually."""


def _build_conviction_gate_evaluator(model, toolset: FilteredToolset) -> Agent:
    """retries=2: bounded per CLAUDE.md's LLM cost control rule 6."""
    return Agent(
        model,
        output_type=CandidateEvaluation,
        system_prompt=_CONVICTION_GATE_SYSTEM_PROMPT,
        toolsets=[toolset],
        retries=2,
    )


async def mutual_fund_holdings_analyzer_node(state: GraphState) -> SpecialistState:
    """Judge the real 9 existing IN mutual fund holdings.

    Fails loud: any scheme-resolution/NAV-fetch/LLM error propagates
    uncaught. Cost control: metrics computed once per fund (pure, no
    LLM), reused across the evaluator call; toolsets shared across all
    9 evaluations within one run.
    """
    model = get_specialist_model()
    mfapi_toolset = get_mfapi_toolset()
    yfinance_toolset = get_yfinance_toolset()
    holdings = state["holdings_by_bucket"].get("mutual_fund", {}).get("IN", [])

    async with mfapi_toolset, yfinance_toolset:
        fund_histories: dict[str, list[dict]] = {}
        for holding in holdings:
            scheme_code = await _resolve_scheme_code(holding["name"], holding["ticker"], mfapi_toolset)
            fund_histories[holding["ticker"]] = await _fetch_nav_history(scheme_code, mfapi_toolset)

        redundancy = _compute_redundancy_correlations(fund_histories)

        evaluator = _build_conviction_gate_evaluator(model, yfinance_toolset)
        evaluations = []
        for holding in holdings:
            isin = holding["ticker"]
            category = holding["sector"]
            history = fund_histories[isin]
            metrics = _compute_fund_metrics(history)
            benchmark_ticker = _lookup_benchmark_ticker(isin, category)
            benchmark = await _compute_benchmark_difference(history, benchmark_ticker, yfinance_toolset)
            metrics_text = _format_metrics_for_prompt(metrics, benchmark, redundancy.get(isin, {}))

            eval_result = await evaluator.run(
                f"Evaluate existing holding: {holding['name']} (category: {category}).\n{metrics_text}"
            )
            evaluations.append(eval_result.output.model_dump())

    return {"existing_fund_evaluations": evaluations}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_mutual_fund_holdings_analyzer.py -v`
Expected: all PASS. (This test makes real calls to `portfolio_data_mcp`, `mfapi_mcp`, and `yfinance-mcp` for all 9 real funds - free, unmocked, consistent with this project's testing philosophy. `TestModel` covers the LLM call.)

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/nodes/mutual_fund_holdings_analyzer.py tests/graph/test_mutual_fund_holdings_analyzer.py
git commit -m "feat: add Mutual Fund Holdings Analyzer node

Judges the real 9 existing IN mutual fund holdings via the redefined
7-check Conviction Gate. Measures first (pure metrics/benchmark/
redundancy computation), judges second (one LLM evaluator call per
fund) - mirrors the Risk Analyst's measure-not-judge separation."
```

---

### Task 13: `mutual_fund_specialist_node`

**Files:**
- Modify: `src/agentic_frameworks/graph/nodes/_mutual_fund_helpers.py`
- Create: `src/agentic_frameworks/graph/nodes/mutual_fund_specialist.py`
- Test: Modify `tests/graph/test_mutual_fund_helpers.py`
- Test: Create `tests/graph/test_mutual_fund_specialist.py`

**Interfaces:**
- Consumes: `get_mfapi_toolset`, `get_yfinance_toolset` (`agentic_frameworks.graph.mcp_clients`), `get_specialist_model` (`agentic_frameworks.graph.models`), `_apply_candidate_cap`, `_filter_to_gap_sectors` (`agentic_frameworks.graph.nodes._specialist_helpers`, Task 1's generalized versions), `_compute_mf_category_gaps`, `_fetch_nav_history`, `_compute_fund_metrics`, `_compute_benchmark_difference`, `MF_CATEGORY_BENCHMARK_TICKERS` (`agentic_frameworks.graph.nodes._mutual_fund_helpers`), `ProposedFundCandidate`, `CandidateEvaluation` (`agentic_frameworks.graph.schemas`), `_format_metrics_for_prompt` from Task 12's `mutual_fund_holdings_analyzer.py` (imported, not reimplemented — the numeric formatting logic is identical; only the system prompts differ between the two nodes since one evaluates an existing holding and the other a new-buy candidate), `GraphState`/`SpecialistState` (`agentic_frameworks.graph.state`).
- Produces: `async def _resolve_scheme_by_name_and_direct_plan(scheme_name: str, toolset) -> tuple[int, str]` (new, in `_mutual_fund_helpers.py`), `async def mutual_fund_specialist_node(state: GraphState) -> SpecialistState` — consumed by Task 14's `graph.py`.

**Design note carried from the spec:** a new-buy candidate has no known ISIN yet (unlike an existing holding), so it cannot be verified with Task 4's `_resolve_scheme_code` (which requires a known ISIN to match against). Verification here instead confirms the LLM-proposed fund name resolves to a real, currently-listed **Direct-plan** scheme via `MFapi.in` search — the anti-hallucination safeguard, mirroring `_verify_ticker`'s role in the equity specialists: a fabricated or delisted fund name simply has no real search match.

- [ ] **Step 1: Write failing tests for the new verification helper**

Add to `tests/graph/test_mutual_fund_helpers.py`:

```python
async def test_resolve_scheme_by_name_and_direct_plan_finds_real_fund():
    from agentic_frameworks.graph.mcp_clients import get_mfapi_toolset
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
        _resolve_scheme_by_name_and_direct_plan,
    )

    toolset = get_mfapi_toolset()
    async with toolset:
        scheme_code, category = await _resolve_scheme_by_name_and_direct_plan(
            "ICICI Prudential Gilt Fund", toolset
        )

    assert scheme_code is not None
    assert "gilt" in category.lower()


async def test_resolve_scheme_by_name_and_direct_plan_raises_on_fabricated_name():
    import pytest

    from agentic_frameworks.graph.mcp_clients import get_mfapi_toolset
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
        _resolve_scheme_by_name_and_direct_plan,
    )

    toolset = get_mfapi_toolset()
    async with toolset:
        with pytest.raises(ValueError):
            await _resolve_scheme_by_name_and_direct_plan(
                "Completely Fictional Nonexistent Fund XYZ123", toolset
            )
```

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_mutual_fund_helpers.py -k resolve_scheme_by_name_and_direct_plan -v`
Expected: FAIL with `ImportError: cannot import name '_resolve_scheme_by_name_and_direct_plan'`.

- [ ] **Step 2: Implement the verification helper**

Add to `src/agentic_frameworks/graph/nodes/_mutual_fund_helpers.py`:

```python
async def _resolve_scheme_by_name_and_direct_plan(scheme_name: str, toolset) -> tuple[int, str]:
    """Verify a proposed (not-yet-held) fund name resolves to a real,
    currently-listed Direct-plan scheme.

    Unlike _resolve_scheme_code (which verifies against a KNOWN ISIN for
    an existing holding), a new-buy candidate has no ISIN yet - this is
    the anti-hallucination safeguard instead: a fabricated or delisted
    fund name returns no real search results at all. Returns
    (scheme_code, scheme_category) for the first Direct-plan match found.
    Raises ValueError (fail loud) if no genuine match exists.
    """
    candidates = await _search_scheme(scheme_name, toolset)
    for candidate in candidates:
        if "direct" in candidate["scheme_name"].lower():
            result = await toolset.direct_call_tool(
                "get_scheme_nav", {"scheme_code": candidate["scheme_code"]}
            )
            detail = json.loads(result)
            return candidate["scheme_code"], detail["meta"]["scheme_category"]
    raise ValueError(f"No real Direct-plan scheme found matching name={scheme_name!r}")
```

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_mutual_fund_helpers.py -k resolve_scheme_by_name_and_direct_plan -v`
Expected: PASS.

- [ ] **Step 3: Write failing tests for the node**

Create `tests/graph/test_mutual_fund_specialist.py`:

```python
"""Tests for the Mutual Fund Specialist (new-buy) node."""

from __future__ import annotations


async def test_candidate_generator_produces_proposed_fund_candidates():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes.mutual_fund_specialist import (
        _build_candidate_generator,
    )

    generator = _build_candidate_generator(TestModel())
    result = await generator.run("propose candidates for: Gilt")

    assert isinstance(result.output, list)
    assert all(hasattr(c, "scheme_name") and hasattr(c, "category") for c in result.output)


async def test_mutual_fund_specialist_node_returns_candidate_evaluations():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes import mutual_fund_specialist as node_module
    from agentic_frameworks.graph.nodes.research import research_node

    def fake_get_specialist_model():
        return TestModel()

    research_result = await research_node({})
    state = {"holdings_by_bucket": research_result["holdings_by_bucket"]}

    original = node_module.get_specialist_model
    node_module.get_specialist_model = fake_get_specialist_model
    try:
        result = await node_module.mutual_fund_specialist_node(state)
    finally:
        node_module.get_specialist_model = original

    assert "mutual_fund_candidates" in result
    assert isinstance(result["mutual_fund_candidates"], list)
    for candidate in result["mutual_fund_candidates"]:
        assert candidate["action"] in {"Buy", "Hold", "Sell", "Reduce"}
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_mutual_fund_specialist.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_frameworks.graph.nodes.mutual_fund_specialist'`.

- [ ] **Step 5: Implement the node**

Create `src/agentic_frameworks/graph/nodes/mutual_fund_specialist.py`:

```python
"""The Mutual Fund Specialist node: proposes and evaluates new-buy
mutual fund candidates for live-computed category gaps, mirroring the
US/India equity specialists' three-stage pipeline (propose -> verify ->
evaluate). Fails loud, same as every other node.
"""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.toolsets.filtered import FilteredToolset

from agentic_frameworks.graph.mcp_clients import get_mfapi_toolset, get_yfinance_toolset
from agentic_frameworks.graph.models import get_specialist_model
from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
    MF_CATEGORY_BENCHMARK_TICKERS,
    _compute_benchmark_difference,
    _compute_fund_metrics,
    _compute_mf_category_gaps,
    _fetch_nav_history,
    _resolve_scheme_by_name_and_direct_plan,
)
from agentic_frameworks.graph.nodes._specialist_helpers import (
    _apply_candidate_cap,
    _filter_to_gap_sectors,
)
from agentic_frameworks.graph.nodes.mutual_fund_holdings_analyzer import (
    _format_metrics_for_prompt,
)
from agentic_frameworks.graph.schemas import CandidateEvaluation, ProposedFundCandidate
from agentic_frameworks.graph.state import GraphState, SpecialistState

_CANDIDATE_GENERATOR_SYSTEM_PROMPT = """You are an India mutual fund research assistant for a personal \
long-term portfolio advisor. Given a list of category gaps in the \
user's current mutual fund holdings, propose real, currently-active, \
Direct-plan Indian mutual fund schemes that could fill each gap - up \
to 3 schemes per gap category.

Always propose the Direct plan variant by name (not Regular plan) - \
Direct plans avoid distributor commission and are the correct \
long-term choice. Only propose funds you are confident are real, \
currently-active schemes. Do not propose discontinued or fictional \
funds. For each candidate, name the scheme (including "Direct Plan" \
in the name) and the gap category it addresses."""


def _build_candidate_generator(model) -> Agent:
    """retries=2: bounded per CLAUDE.md's LLM cost control rule 6."""
    return Agent(
        model,
        output_type=list[ProposedFundCandidate],
        system_prompt=_CANDIDATE_GENERATOR_SYSTEM_PROMPT,
        retries=2,
    )


_NEW_BUY_CONVICTION_GATE_SYSTEM_PROMPT = """You are evaluating one new-buy India mutual fund candidate for a \
personal, long-term portfolio advisor, using a 7-check Conviction Gate \
adapted for mutual funds (not equities - no price/valuation checks apply). \
Run these checks in order:

1. Portfolio-fit: does this fund address a real, identified category gap \
in the user's current holdings?
2. Cost: only propose/favor Direct plan funds - Direct plans avoid \
distributor commission drag (exact expense ratios are not available from \
this project's data sources, so this stays a qualitative check).
3. Performance vs benchmark: did it beat its benchmark over its own \
history, where a verified benchmark exists? If unavailable, say so \
explicitly rather than guessing.
4. Risk-adjusted performance: weigh volatility and max drawdown, not just \
raw returns.
5. Track-record length: a short track record (under ~3 years) means \
performance-consistency claims should carry less weight.
6. Style-fit: is this fund's mandate consistent with a long-term, \
buy-and-hold approach?

Avoid is the default, not buy: if checks are weak or data is missing, the \
action should reflect that (Hold, not Buy), with the specific reason in \
risk_flags. Tax context (India, current rules): equity-oriented mutual \
funds (>=65% equity) are taxed as equity capital gains; other funds are \
taxed at the investor's income slab rate regardless of holding period; \
ELSS funds carry a mandatory 3-year lock-in. Your output is advisory only \
- never a trade instruction, only a recommendation for the user to act on \
manually."""


def _build_conviction_gate_evaluator(model, toolset: FilteredToolset) -> Agent:
    """retries=2: bounded per CLAUDE.md's LLM cost control rule 6."""
    return Agent(
        model,
        output_type=CandidateEvaluation,
        system_prompt=_NEW_BUY_CONVICTION_GATE_SYSTEM_PROMPT,
        toolsets=[toolset],
        retries=2,
    )


async def mutual_fund_specialist_node(state: GraphState) -> SpecialistState:
    """Propose, verify, and evaluate new-buy mutual fund candidates.

    Fails loud: any LLM/tool/verification error propagates uncaught,
    except a candidate whose proposed name has no real Direct-plan
    match (ValueError from _resolve_scheme_by_name_and_direct_plan),
    which is dropped - the same reject-early role _verify_ticker plays
    for the equity specialists. Cost control: reject-early (empty gap
    list short-circuits before any model/toolset setup; candidates are
    filtered to real gap categories and capped before the expensive
    resolve/metrics/evaluate steps run).
    """
    gap_categories = _compute_mf_category_gaps(state["holdings_by_bucket"])
    if not gap_categories:
        return {"mutual_fund_candidates": []}

    model = get_specialist_model()
    mfapi_toolset = get_mfapi_toolset()
    yfinance_toolset = get_yfinance_toolset()

    async with mfapi_toolset, yfinance_toolset:
        generator = _build_candidate_generator(model)
        gap_categories_text = ", ".join(gap_categories)
        generation_result = await generator.run(
            f"Propose candidates for these gap categories: {gap_categories_text}"
        )
        on_gap_candidates = _filter_to_gap_sectors(
            generation_result.output, gap_categories, key=lambda c: c.category
        )
        capped_candidates = _apply_candidate_cap(on_gap_candidates, key=lambda c: c.category)

        evaluator = _build_conviction_gate_evaluator(model, yfinance_toolset)
        evaluations = []
        for candidate in capped_candidates:
            try:
                scheme_code, _resolved_category = await _resolve_scheme_by_name_and_direct_plan(
                    candidate.scheme_name, mfapi_toolset
                )
            except ValueError:
                continue

            history = await _fetch_nav_history(scheme_code, mfapi_toolset)
            metrics = _compute_fund_metrics(history)
            benchmark_ticker = MF_CATEGORY_BENCHMARK_TICKERS.get(candidate.category)
            benchmark = await _compute_benchmark_difference(history, benchmark_ticker, yfinance_toolset)
            metrics_text = _format_metrics_for_prompt(metrics, benchmark, {})

            eval_result = await evaluator.run(
                f"Evaluate candidate: {candidate.scheme_name} (category: {candidate.category}).\n{metrics_text}"
            )
            evaluations.append(eval_result.output.model_dump())

    return {"mutual_fund_candidates": evaluations}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_mutual_fund_helpers.py tests/graph/test_mutual_fund_specialist.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/agentic_frameworks/graph/nodes/_mutual_fund_helpers.py src/agentic_frameworks/graph/nodes/mutual_fund_specialist.py tests/graph/test_mutual_fund_helpers.py tests/graph/test_mutual_fund_specialist.py
git commit -m "feat: add Mutual Fund Specialist node for new-buy candidates

Mirrors the equity specialists' propose -> verify -> evaluate pipeline.
Verification uses a name+Direct-plan real-scheme check (not ISIN
matching, since new-buy candidates have no known ISIN yet) as the
anti-hallucination safeguard - a fabricated fund name has no real
search match at all."
```

---

### Task 14: Wire both new nodes into the graph

**Files:**
- Modify: `src/agentic_frameworks/graph/graph.py`
- Test: Modify `tests/graph/test_graph.py`

**Interfaces:**
- Consumes: `mutual_fund_holdings_analyzer_node` (Task 12), `mutual_fund_specialist_node` (Task 13).

- [ ] **Step 1: Write failing tests**

Modify `tests/graph/test_graph.py`'s node-set test:

```python
def test_build_graph_wires_all_five_nodes():
    from agentic_frameworks.graph.graph import build_graph

    graph = build_graph()

    assert set(graph.nodes.keys()) == {
        "research",
        "risk_analyst",
        "us_equity_etf_specialist",
        "india_equity_specialist",
        "mutual_fund_holdings_analyzer",
        "mutual_fund_specialist",
    }
```

(replacing the old `test_build_graph_wires_research_risk_analyst_and_both_specialists`)

Update the e2e test to monkeypatch all four LLM-calling node modules and expect all four new candidate/evaluation keys:

```python
async def test_graph_runs_end_to_end_with_real_data_and_persists_checkpoint():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.checkpointer import get_checkpointer
    from agentic_frameworks.graph.graph import build_graph
    from agentic_frameworks.graph.nodes import india_equity_specialist as india_module
    from agentic_frameworks.graph.nodes import mutual_fund_holdings_analyzer as mf_holdings_module
    from agentic_frameworks.graph.nodes import mutual_fund_specialist as mf_specialist_module
    from agentic_frameworks.graph.nodes import us_equity_etf_specialist as us_module

    def fake_get_specialist_model():
        return TestModel()

    modules = [india_module, mf_holdings_module, mf_specialist_module, us_module]
    originals = [m.get_specialist_model for m in modules]
    for m in modules:
        m.get_specialist_model = fake_get_specialist_model
    try:
        async with get_checkpointer() as checkpointer:
            compiled = build_graph().compile(checkpointer=checkpointer)
            thread_id = f"test-graph-harness-e2e-{uuid4()}"
            config = {"configurable": {"thread_id": thread_id}}

            result = await compiled.ainvoke({}, config=config)

            assert set(result.keys()) == {
                "holdings",
                "allocation",
                "holdings_by_bucket",
                "concentration",
                "category_overlap",
                "currency_exposure",
                "liquidity",
                "us_equity_etf_candidates",
                "india_equity_candidates",
                "existing_fund_evaluations",
                "mutual_fund_candidates",
            }
            assert len(result["holdings_by_bucket"]["real_estate"]["IN"]) == 1
            assert result["concentration"]["by_holding"]
            assert result["currency_exposure"]
            assert result["liquidity"]["illiquid"]["value_base"] > 0
            assert result["us_equity_etf_candidates"]
            assert result["india_equity_candidates"]
            assert len(result["existing_fund_evaluations"]) == 9

            state_snapshot = await compiled.aget_state(config)
            assert state_snapshot.values
            assert state_snapshot.config["configurable"]["thread_id"] == thread_id
    finally:
        for m, original in zip(modules, originals):
            m.get_specialist_model = original
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_graph.py -v`
Expected: FAIL (node set mismatch; missing result keys).

- [ ] **Step 3: Wire both nodes into `build_graph()`**

Replace the contents of `src/agentic_frameworks/graph/graph.py`:

```python
"""Builds the Portfolio Advisor Fleet graph: wires existing nodes together.

Research -> Risk Analyst -> {US Equity/ETF Specialist, India Equity
Specialist, Mutual Fund Holdings Analyzer, Mutual Fund Specialist} in
parallel. All four branches currently dead-end at END - fan-in to a
Synthesizer arrives in a later phase.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agentic_frameworks.graph.nodes.india_equity_specialist import (
    india_equity_specialist_node,
)
from agentic_frameworks.graph.nodes.mutual_fund_holdings_analyzer import (
    mutual_fund_holdings_analyzer_node,
)
from agentic_frameworks.graph.nodes.mutual_fund_specialist import (
    mutual_fund_specialist_node,
)
from agentic_frameworks.graph.nodes.research import research_node
from agentic_frameworks.graph.nodes.risk_analyst import risk_analyst_node
from agentic_frameworks.graph.nodes.us_equity_etf_specialist import (
    us_equity_etf_specialist_node,
)
from agentic_frameworks.graph.state import GraphState


def build_graph() -> StateGraph:
    """Wire research, risk_analyst, and all four new-buy/holdings-analysis nodes into an uncompiled StateGraph."""
    graph = StateGraph(GraphState)
    graph.add_node("research", research_node)
    graph.add_node("risk_analyst", risk_analyst_node)
    graph.add_node("us_equity_etf_specialist", us_equity_etf_specialist_node)
    graph.add_node("india_equity_specialist", india_equity_specialist_node)
    graph.add_node("mutual_fund_holdings_analyzer", mutual_fund_holdings_analyzer_node)
    graph.add_node("mutual_fund_specialist", mutual_fund_specialist_node)
    graph.add_edge(START, "research")
    graph.add_edge("research", "risk_analyst")
    graph.add_edge("risk_analyst", "us_equity_etf_specialist")
    graph.add_edge("risk_analyst", "india_equity_specialist")
    graph.add_edge("risk_analyst", "mutual_fund_holdings_analyzer")
    graph.add_edge("risk_analyst", "mutual_fund_specialist")
    graph.add_edge("us_equity_etf_specialist", END)
    graph.add_edge("india_equity_specialist", END)
    graph.add_edge("mutual_fund_holdings_analyzer", END)
    graph.add_edge("mutual_fund_specialist", END)
    return graph
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_graph.py -v`
Expected: all PASS. (Real `portfolio_data_mcp`, `mfapi_mcp`, and `yfinance-mcp` calls are expected and free; only LLM calls are mocked via `TestModel`.)

- [ ] **Step 5: Run the entire test suite to confirm no regressions**

Run: `.venv/Scripts/python.exe -m pytest tests/graph -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agentic_frameworks/graph/graph.py tests/graph/test_graph.py
git commit -m "feat: wire both mutual fund nodes into the graph as parallel branches

Four parallel branches now run off risk_analyst: US Equity/ETF,
India Equity, Mutual Fund Holdings Analyzer, and Mutual Fund
Specialist. All four still dead-end at END until the Synthesizer
(fan-in) exists in a later phase."
```

---

### Task 15: Manual real-model verification run (phase-completion gate)

This task has no automated test — it's the same manual, cost-bounded, human-reviewed gate Phases 4 and 5 required before being marked done.

- [ ] **Step 1: Run the real graph end-to-end against the actual model pair**

Run (no monkeypatching — this uses real `OPENROUTER_API_KEY` spend, deliberately):

```bash
.venv/Scripts/python.exe -c "
import asyncio, json
from agentic_frameworks.graph.checkpointer import get_checkpointer
from agentic_frameworks.graph.graph import build_graph

async def main():
    async with get_checkpointer() as checkpointer:
        compiled = build_graph().compile(checkpointer=checkpointer)
        result = await compiled.ainvoke({}, config={'configurable': {'thread_id': 'manual-phase6-verification'}})
        with open('mf_verification_output.json', 'w', encoding='utf-8') as f:
            json.dump({
                'existing_fund_evaluations': result['existing_fund_evaluations'],
                'mutual_fund_candidates': result['mutual_fund_candidates'],
            }, f, indent=2, ensure_ascii=False)
        print('existing_fund_evaluations count:', len(result['existing_fund_evaluations']))
        print('mutual_fund_candidates count:', len(result['mutual_fund_candidates']))

asyncio.run(main())
"
```

(Write to a file rather than printing directly, to avoid the Windows console Unicode-encoding crash encountered in Phase 5's verification run — read the file back with `Read` afterward.)

- [ ] **Step 2: Review output with the user**

Present all 9 existing-fund evaluations and any new-buy candidates. Confirm: fund names/categories are real, `confidence` values are within `[0.0, 1.0]`, benchmark comparisons (where present) reference real numbers, benchmark-unavailable cases are explicitly stated (not silently omitted or guessed), redundancy findings make sense given the fund categories (e.g. the two Small Cap funds should show meaningfully higher correlation with each other than with the Index Funds), reasoning references real holdings and real metrics, no fabricated data, genuinely advisory language throughout.

- [ ] **Step 3: Update `HANDOFF.md`**

Once the user confirms the manual run looks correct, update `HANDOFF.md`'s "What's actually done" and "What's explicitly NOT done yet" sections to reflect Phase 6's completion, following the same structure used for Phases 1-5. Commit alongside no code changes:

```bash
git add HANDOFF.md
git commit -m "docs: update handoff for Phase 6 (Mutual Fund Specialist) completion"
```

- [ ] **Step 4: Push `main`**

Per `CLAUDE.md`'s phased development rule:

```bash
git push origin main
```
