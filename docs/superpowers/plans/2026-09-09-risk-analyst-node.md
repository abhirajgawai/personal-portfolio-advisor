# Risk Analyst Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `risk_analyst_node`, a pure function that consumes the Research node's `ResearchState` and computes concentration, category-overlap, currency/FX exposure, and liquidity views of the existing portfolio.

**Architecture:** One `RiskState` TypedDict extending `state.py`, and one new file `nodes/risk_analyst.py` containing `risk_analyst_node` composed of four private helper functions, each independently unit-tested. No MCP client, no I/O — pure data transformation over `ResearchState["holdings"]`.

**Tech Stack:** Python 3.14, pytest/pytest-asyncio (already configured), no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-09-risk-analyst-node-design.md`

## Global Constraints

- Fail loud: no `try`/`except` anywhere in `risk_analyst.py` — a holding missing a required field must raise `KeyError`, never be silently skipped or defaulted.
- No verdicts/thresholds: every computed value is a raw number (weight, %, total) — no "OVERWEIGHT"/"OK"/flag logic anywhere in this node.
- No new MCP client, no subprocess, no network I/O in this node — it only transforms the `holdings` list already present in `ResearchState`.
- Test fixtures use real field names and real-world-realistic values (real tickers/names/sectors seen in the actual portfolio), not gibberish — but exact market values in fixtures may be chosen for clean test math (this is a fixture for a pure-function unit test, unlike the Research node's live-server integration tests).
- `category_overlap.by_sector` includes only sectors with **more than one** holding — a sector with exactly one holding is not "overlap" and must not appear in the output.

## Shared Test Fixture

Every task below that writes a test uses this exact same holding list (so results are comparable across tasks). Total portfolio value is 100,000 with clean round splits:

```python
HOLDINGS = [
    {
        "platform": "zerodha", "geography": "IN", "ticker": "INF846K01K35",
        "name": "Axis Small Cap Fund Direct Growth", "currency": "INR",
        "sector": "Small Cap", "asset_class": "mutual_fund",
        "quantity": 1000.0, "price": 10.0, "market_value_base": 10000.0,
        "updated_at": "2026-09-08T00:00:00Z",
    },
    {
        "platform": "zerodha", "geography": "IN", "ticker": "INF179K01WN9",
        "name": "HDFC BSE Sensex Index Fund - Direct Plan - Growth Option",
        "currency": "INR", "sector": "Index Fund", "asset_class": "mutual_fund",
        "quantity": 1000.0, "price": 10.0, "market_value_base": 10000.0,
        "updated_at": "2026-09-08T00:00:00Z",
    },
    {
        "platform": "zerodha", "geography": "IN", "ticker": "INF769K01LG4",
        "name": "Mirae Asset Nifty Smallcap 250 Momentum Quality 100 ETF Fund of Fund Direct Plan",
        "currency": "INR", "sector": "Small Cap", "asset_class": "etf",
        "quantity": 1000.0, "price": 10.0, "market_value_base": 10000.0,
        "updated_at": "2026-09-08T00:00:00Z",
    },
    {
        "platform": "moomoo", "geography": "US", "ticker": "VOO",
        "name": "Vanguard S&P 500 ETF", "currency": "USD",
        "sector": "US Large Cap", "asset_class": "etf",
        "quantity": 20.0, "price": 500.0, "market_value_base": 10000.0,
        "updated_at": "2026-09-08T00:00:00Z",
    },
    {
        "platform": "zerodha", "geography": "IN", "ticker": "HDFCBANK",
        "name": "HDFC Bank Ltd", "currency": "INR",
        "sector": "Banking", "asset_class": "equity",
        "quantity": 100.0, "price": 200.0, "market_value_base": 20000.0,
        "updated_at": "2026-09-08T00:00:00Z",
    },
    {
        "platform": "groww", "geography": "IN", "ticker": "TALOJA_PROPERTY",
        "name": "Asavari Taloja", "currency": "INR",
        "sector": "Residential", "asset_class": "real_estate",
        "quantity": 1.0, "price": 40000.0, "market_value_base": 40000.0,
        "updated_at": "2026-09-08T00:00:00Z",
    },
]
```

Known correct results for this fixture (used throughout the tasks below):
- Total portfolio value: 100,000.
- By asset class: `mutual_fund` = 20,000 (20%), `etf` = 20,000 (20%), `equity` = 20,000 (20%), `real_estate` = 40,000 (40%).
- Within `mutual_fund`: the two holdings are 50%/50% of that bucket. Same for `etf`. `equity` and `real_estate` are each 100% of their own bucket (one holding each).
- Sector `"Small Cap"` appears on 2 holdings (`INF846K01K35`, a `mutual_fund`, and `INF769K01LG4`, an `etf`) — this is the cross-asset-class overlap case. `"Index Fund"`, `"US Large Cap"`, `"Banking"`, `"Residential"` each appear on exactly 1 holding — no overlap.
- Currency: `INR` = 10000+10000+10000+20000+40000 = 90,000 (90%). `USD` = 10,000 (10%).
- Liquidity: `illiquid` (real_estate only) = 40,000 (40%). `liquid` = 60,000 (60%).

---

### Task 1: `RiskState` schema

**Files:**
- Modify: `src/agentic_frameworks/graph/state.py`
- Test: `tests/graph/test_state.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `RiskState` TypedDict with keys `concentration: dict`, `category_overlap: dict`, `currency_exposure: dict`, `liquidity: dict` — Task 6's `risk_analyst_node` return type is annotated with this.

- [ ] **Step 1: Write the failing test**

Append to `tests/graph/test_state.py`:

```python
def test_risk_state_has_expected_keys():
    from agentic_frameworks.graph.state import RiskState

    hints = get_type_hints(RiskState)
    assert set(hints.keys()) == {
        "concentration",
        "category_overlap",
        "currency_exposure",
        "liquidity",
    }
    assert hints["concentration"] is dict
    assert hints["category_overlap"] is dict
    assert hints["currency_exposure"] is dict
    assert hints["liquidity"] is dict
```

(`get_type_hints` is already imported at the top of this file from the Research node work — if it isn't, add `from typing import get_type_hints`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph/test_state.py::test_risk_state_has_expected_keys -v`
Expected: FAIL with `ImportError` or `AttributeError` (`RiskState` doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Append to `src/agentic_frameworks/graph/state.py`:

```python
class RiskState(TypedDict):
    """State the Risk Analyst node produces: raw risk measurements, no verdicts.

    concentration: {"by_holding": [{"ticker", "name", "asset_class",
        "pct_of_portfolio", "pct_of_asset_class"}], "by_asset_class":
        [{"asset_class", "pct_of_portfolio"}]}.
    category_overlap: {"by_sector": {"<sector>": [{"ticker", "name",
        "asset_class"}]}} — only sectors with more than one holding.
    currency_exposure: {"<currency>": {"value_base", "pct_of_portfolio"}}.
    liquidity: {"illiquid": {"value_base", "pct_of_portfolio"},
        "liquid": {"value_base", "pct_of_portfolio"}}.
    """

    concentration: dict
    category_overlap: dict
    currency_exposure: dict
    liquidity: dict
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph/test_state.py -v`
Expected: PASS (both the new test and the existing `test_research_state_has_expected_keys`).

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/state.py tests/graph/test_state.py
git commit -m "feat: add RiskState schema"
```

---

### Task 2: `_concentration` helper

**Files:**
- Create: `src/agentic_frameworks/graph/nodes/risk_analyst.py`
- Create: `tests/graph/test_risk_analyst.py`

**Interfaces:**
- Consumes: a `list[dict]` of holdings, each with at least `ticker`, `name`, `asset_class`, `market_value_base`.
- Produces: `_concentration(holdings: list[dict]) -> dict` with keys `by_holding` (list) and `by_asset_class` (list) — Task 6's `risk_analyst_node` calls this directly.

- [ ] **Step 1: Write the failing test**

Create `tests/graph/test_risk_analyst.py`:

```python
"""Tests for the Risk Analyst node and its helper functions."""

from __future__ import annotations

import pytest

HOLDINGS = [
    {
        "platform": "zerodha", "geography": "IN", "ticker": "INF846K01K35",
        "name": "Axis Small Cap Fund Direct Growth", "currency": "INR",
        "sector": "Small Cap", "asset_class": "mutual_fund",
        "quantity": 1000.0, "price": 10.0, "market_value_base": 10000.0,
        "updated_at": "2026-09-08T00:00:00Z",
    },
    {
        "platform": "zerodha", "geography": "IN", "ticker": "INF179K01WN9",
        "name": "HDFC BSE Sensex Index Fund - Direct Plan - Growth Option",
        "currency": "INR", "sector": "Index Fund", "asset_class": "mutual_fund",
        "quantity": 1000.0, "price": 10.0, "market_value_base": 10000.0,
        "updated_at": "2026-09-08T00:00:00Z",
    },
    {
        "platform": "zerodha", "geography": "IN", "ticker": "INF769K01LG4",
        "name": "Mirae Asset Nifty Smallcap 250 Momentum Quality 100 ETF Fund of Fund Direct Plan",
        "currency": "INR", "sector": "Small Cap", "asset_class": "etf",
        "quantity": 1000.0, "price": 10.0, "market_value_base": 10000.0,
        "updated_at": "2026-09-08T00:00:00Z",
    },
    {
        "platform": "moomoo", "geography": "US", "ticker": "VOO",
        "name": "Vanguard S&P 500 ETF", "currency": "USD",
        "sector": "US Large Cap", "asset_class": "etf",
        "quantity": 20.0, "price": 500.0, "market_value_base": 10000.0,
        "updated_at": "2026-09-08T00:00:00Z",
    },
    {
        "platform": "zerodha", "geography": "IN", "ticker": "HDFCBANK",
        "name": "HDFC Bank Ltd", "currency": "INR",
        "sector": "Banking", "asset_class": "equity",
        "quantity": 100.0, "price": 200.0, "market_value_base": 20000.0,
        "updated_at": "2026-09-08T00:00:00Z",
    },
    {
        "platform": "groww", "geography": "IN", "ticker": "TALOJA_PROPERTY",
        "name": "Asavari Taloja", "currency": "INR",
        "sector": "Residential", "asset_class": "real_estate",
        "quantity": 1.0, "price": 40000.0, "market_value_base": 40000.0,
        "updated_at": "2026-09-08T00:00:00Z",
    },
]


def test_concentration_computes_correct_weights():
    from agentic_frameworks.graph.nodes.risk_analyst import _concentration

    result = _concentration(HOLDINGS)

    by_holding = {h["ticker"]: h for h in result["by_holding"]}
    assert by_holding["INF846K01K35"]["pct_of_portfolio"] == pytest.approx(10.0)
    assert by_holding["INF846K01K35"]["pct_of_asset_class"] == pytest.approx(50.0)
    assert by_holding["HDFCBANK"]["pct_of_portfolio"] == pytest.approx(20.0)
    assert by_holding["HDFCBANK"]["pct_of_asset_class"] == pytest.approx(100.0)
    assert by_holding["TALOJA_PROPERTY"]["pct_of_portfolio"] == pytest.approx(40.0)

    by_asset_class = {a["asset_class"]: a["pct_of_portfolio"] for a in result["by_asset_class"]}
    assert by_asset_class["mutual_fund"] == pytest.approx(20.0)
    assert by_asset_class["etf"] == pytest.approx(20.0)
    assert by_asset_class["equity"] == pytest.approx(20.0)
    assert by_asset_class["real_estate"] == pytest.approx(40.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph/test_risk_analyst.py::test_concentration_computes_correct_weights -v`
Expected: FAIL with `ModuleNotFoundError` (`risk_analyst.py` doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Create `src/agentic_frameworks/graph/nodes/risk_analyst.py`:

```python
"""The Risk Analyst node: pure risk measurements over existing holdings.

No MCP client, no I/O - every function here is a synchronous
transformation of the holdings list the Research node already fetched.
Reports raw numbers only; no thresholds or verdicts.
"""

from __future__ import annotations


def _concentration(holdings: list[dict]) -> dict:
    total = sum(h["market_value_base"] for h in holdings)

    by_asset_class_totals: dict[str, float] = {}
    for h in holdings:
        by_asset_class_totals.setdefault(h["asset_class"], 0.0)
        by_asset_class_totals[h["asset_class"]] += h["market_value_base"]

    by_holding = [
        {
            "ticker": h["ticker"],
            "name": h["name"],
            "asset_class": h["asset_class"],
            "pct_of_portfolio": h["market_value_base"] / total * 100,
            "pct_of_asset_class": (
                h["market_value_base"] / by_asset_class_totals[h["asset_class"]] * 100
            ),
        }
        for h in holdings
    ]

    by_asset_class = [
        {"asset_class": asset_class, "pct_of_portfolio": value / total * 100}
        for asset_class, value in by_asset_class_totals.items()
    ]

    return {"by_holding": by_holding, "by_asset_class": by_asset_class}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph/test_risk_analyst.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/nodes/risk_analyst.py tests/graph/test_risk_analyst.py
git commit -m "feat: add concentration risk helper"
```

---

### Task 3: `_category_overlap` helper

**Files:**
- Modify: `src/agentic_frameworks/graph/nodes/risk_analyst.py`
- Modify: `tests/graph/test_risk_analyst.py`

**Interfaces:**
- Consumes: same `HOLDINGS` fixture and holding shape as Task 2 (each holding has `ticker`, `name`, `asset_class`, `sector`).
- Produces: `_category_overlap(holdings: list[dict]) -> dict` with key `by_sector` (dict) — Task 6's `risk_analyst_node` calls this directly.

- [ ] **Step 1: Write the failing test**

Append to `tests/graph/test_risk_analyst.py` (reuses the `HOLDINGS` constant already defined at the top of this file from Task 2):

```python
def test_category_overlap_includes_only_shared_sectors():
    from agentic_frameworks.graph.nodes.risk_analyst import _category_overlap

    result = _category_overlap(HOLDINGS)

    assert set(result["by_sector"].keys()) == {"Small Cap"}

    small_cap_tickers = {h["ticker"] for h in result["by_sector"]["Small Cap"]}
    assert small_cap_tickers == {"INF846K01K35", "INF769K01LG4"}

    small_cap_asset_classes = {h["asset_class"] for h in result["by_sector"]["Small Cap"]}
    assert small_cap_asset_classes == {"mutual_fund", "etf"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph/test_risk_analyst.py::test_category_overlap_includes_only_shared_sectors -v`
Expected: FAIL with `ImportError` (`_category_overlap` doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Append to `src/agentic_frameworks/graph/nodes/risk_analyst.py`:

```python
def _category_overlap(holdings: list[dict]) -> dict:
    by_sector: dict[str, list[dict]] = {}
    for h in holdings:
        by_sector.setdefault(h["sector"], []).append(
            {"ticker": h["ticker"], "name": h["name"], "asset_class": h["asset_class"]}
        )

    overlapping = {sector: entries for sector, entries in by_sector.items() if len(entries) > 1}
    return {"by_sector": overlapping}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph/test_risk_analyst.py -v`
Expected: PASS (both tests in the file).

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/nodes/risk_analyst.py tests/graph/test_risk_analyst.py
git commit -m "feat: add category-overlap risk helper"
```

---

### Task 4: `_currency_exposure` helper

**Files:**
- Modify: `src/agentic_frameworks/graph/nodes/risk_analyst.py`
- Modify: `tests/graph/test_risk_analyst.py`

**Interfaces:**
- Consumes: same `HOLDINGS` fixture (each holding has `currency`, `market_value_base`).
- Produces: `_currency_exposure(holdings: list[dict]) -> dict` keyed by currency code — Task 6's `risk_analyst_node` calls this directly.

- [ ] **Step 1: Write the failing test**

Append to `tests/graph/test_risk_analyst.py`:

```python
def test_currency_exposure_computes_correct_totals():
    from agentic_frameworks.graph.nodes.risk_analyst import _currency_exposure

    result = _currency_exposure(HOLDINGS)

    assert result["INR"]["value_base"] == pytest.approx(90000.0)
    assert result["INR"]["pct_of_portfolio"] == pytest.approx(90.0)
    assert result["USD"]["value_base"] == pytest.approx(10000.0)
    assert result["USD"]["pct_of_portfolio"] == pytest.approx(10.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph/test_risk_analyst.py::test_currency_exposure_computes_correct_totals -v`
Expected: FAIL with `ImportError` (`_currency_exposure` doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Append to `src/agentic_frameworks/graph/nodes/risk_analyst.py`:

```python
def _currency_exposure(holdings: list[dict]) -> dict:
    total = sum(h["market_value_base"] for h in holdings)

    totals: dict[str, float] = {}
    for h in holdings:
        totals.setdefault(h["currency"], 0.0)
        totals[h["currency"]] += h["market_value_base"]

    return {
        currency: {"value_base": value, "pct_of_portfolio": value / total * 100}
        for currency, value in totals.items()
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph/test_risk_analyst.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/nodes/risk_analyst.py tests/graph/test_risk_analyst.py
git commit -m "feat: add currency exposure risk helper"
```

---

### Task 5: `_liquidity_split` helper

**Files:**
- Modify: `src/agentic_frameworks/graph/nodes/risk_analyst.py`
- Modify: `tests/graph/test_risk_analyst.py`

**Interfaces:**
- Consumes: same `HOLDINGS` fixture (each holding has `asset_class`, `market_value_base`).
- Produces: `_liquidity_split(holdings: list[dict]) -> dict` with keys `illiquid` and `liquid` — Task 6's `risk_analyst_node` calls this directly.

- [ ] **Step 1: Write the failing test**

Append to `tests/graph/test_risk_analyst.py`:

```python
def test_liquidity_split_separates_real_estate():
    from agentic_frameworks.graph.nodes.risk_analyst import _liquidity_split

    result = _liquidity_split(HOLDINGS)

    assert result["illiquid"]["value_base"] == pytest.approx(40000.0)
    assert result["illiquid"]["pct_of_portfolio"] == pytest.approx(40.0)
    assert result["liquid"]["value_base"] == pytest.approx(60000.0)
    assert result["liquid"]["pct_of_portfolio"] == pytest.approx(60.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph/test_risk_analyst.py::test_liquidity_split_separates_real_estate -v`
Expected: FAIL with `ImportError` (`_liquidity_split` doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Append to `src/agentic_frameworks/graph/nodes/risk_analyst.py`:

```python
def _liquidity_split(holdings: list[dict]) -> dict:
    total = sum(h["market_value_base"] for h in holdings)
    illiquid = sum(
        h["market_value_base"] for h in holdings if h["asset_class"] == "real_estate"
    )
    liquid = total - illiquid

    return {
        "illiquid": {"value_base": illiquid, "pct_of_portfolio": illiquid / total * 100},
        "liquid": {"value_base": liquid, "pct_of_portfolio": liquid / total * 100},
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph/test_risk_analyst.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/nodes/risk_analyst.py tests/graph/test_risk_analyst.py
git commit -m "feat: add liquidity split risk helper"
```

---

### Task 6: `risk_analyst_node` — compose the helpers, fail loud on malformed input

**Files:**
- Modify: `src/agentic_frameworks/graph/nodes/risk_analyst.py`
- Modify: `tests/graph/test_risk_analyst.py`

**Interfaces:**
- Consumes: `ResearchState` (from `agentic_frameworks.graph.state`, already defined), and the four helpers `_concentration`, `_category_overlap`, `_currency_exposure`, `_liquidity_split` defined in Tasks 2-5 in this same file.
- Produces: `risk_analyst_node(state: ResearchState) -> RiskState` — the public entry point of this node. No other task builds on this one; this completes the node.

- [ ] **Step 1: Write the failing tests**

Append to `tests/graph/test_risk_analyst.py`:

```python
def test_risk_analyst_node_returns_all_four_risk_views():
    from agentic_frameworks.graph.nodes.risk_analyst import risk_analyst_node

    state = {"holdings": HOLDINGS, "allocation": {}, "holdings_by_bucket": {}}
    result = risk_analyst_node(state)

    assert set(result.keys()) == {
        "concentration",
        "category_overlap",
        "currency_exposure",
        "liquidity",
    }
    assert result["currency_exposure"]["USD"]["pct_of_portfolio"] == pytest.approx(10.0)
    assert result["liquidity"]["illiquid"]["pct_of_portfolio"] == pytest.approx(40.0)
    assert set(result["category_overlap"]["by_sector"].keys()) == {"Small Cap"}


def test_risk_analyst_node_fails_loud_on_holding_missing_sector():
    from agentic_frameworks.graph.nodes.risk_analyst import risk_analyst_node

    broken_holdings = [dict(h) for h in HOLDINGS]
    del broken_holdings[0]["sector"]
    state = {"holdings": broken_holdings, "allocation": {}, "holdings_by_bucket": {}}

    with pytest.raises(KeyError):
        risk_analyst_node(state)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/graph/test_risk_analyst.py::test_risk_analyst_node_returns_all_four_risk_views tests/graph/test_risk_analyst.py::test_risk_analyst_node_fails_loud_on_holding_missing_sector -v`
Expected: both FAIL with `ImportError` (`risk_analyst_node` doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Append to `src/agentic_frameworks/graph/nodes/risk_analyst.py` (this file already has the four private helpers from Tasks 2-5 above it):

```python
def risk_analyst_node(state: dict) -> dict:
    holdings = state["holdings"]
    return {
        "concentration": _concentration(holdings),
        "category_overlap": _category_overlap(holdings),
        "currency_exposure": _currency_exposure(holdings),
        "liquidity": _liquidity_split(holdings),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/graph/test_risk_analyst.py -v`
Expected: PASS (all 6 tests in the file: concentration, category-overlap, currency-exposure, liquidity, node happy-path, node fail-loud).

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/nodes/risk_analyst.py tests/graph/test_risk_analyst.py
git commit -m "feat: add risk_analyst_node composing all risk helpers"
```

---

### Task 7: Full-suite verification (Phase 2 gate)

**Files:** none created or modified — verification only.

**Interfaces:** none — this task consumes everything from Tasks 1-6 to confirm the phase is genuinely done.

- [ ] **Step 1: Run the entire test suite**

Run: `uv run pytest tests/ -v`
Expected: all tests from Tasks 1-6 PASS (RiskState schema, all four helpers, node happy path, node fail-loud), plus all pre-existing Research node tests from Phase 1 still PASS (10 tests total: 4 from Phase 1 + 6 from this phase).

- [ ] **Step 2: Confirm no accidental writes to `portfolio.db`**

This phase's tests use only hand-built fixtures — no MCP client, no subprocess, no database connection at all. Confirm by checking `portfolio.db`'s modified timestamp is unchanged before/after the run (same check as Phase 1): `ls -la E:\agentic-portfolio-dashboard\data\portfolio.db` (or equivalent) before and after Step 1 — it must be identical.

- [ ] **Step 3: Report phase completion**

This phase (Risk Analyst node) is done once Steps 1-2 both pass. Per `CLAUDE.md`'s phased-development rule, do not begin planning Phase 3 (the next graph node) until this step is confirmed complete and reported back.
