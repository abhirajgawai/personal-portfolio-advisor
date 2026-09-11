# India Equity Specialist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `india_equity_specialist_node` (propose → verify → Conviction Gate, mirroring Phase 4's US Equity/ETF Specialist), with gap detection computed live from real holdings for both the new India node and a retrofit of the existing US node (fixing a real bug where the hardcoded `US_GAP_SECTORS` list omits Information Technology).

**Architecture:** A pure, no-LLM gap-computation function per specialist (`_compute_india_equity_gaps`, `_compute_us_gaps`) replaces static/hardcoded gap lists, rolling up real per-holding `sector` strings into the 11-bucket GICS taxonomy and returning zero-coverage buckets. Both specialists then run the same three-stage pipeline already proven in Phase 4: LLM candidate generation (capped in code) → deterministic ticker verification via `yfinance-mcp` → LLM Conviction Gate evaluation per surviving candidate. A new shared helpers module removes the now-duplicated cap-enforcement logic. Both specialist nodes are wired into `build_graph()` as parallel branches off `risk_analyst`.

**Tech Stack:** `pydantic-ai` (`Agent`, `TestModel`, `FallbackModel`), `yfinance-mcp` (via existing `get_yfinance_toolset()`), `langgraph` (`StateGraph`), `pytest` + `pytest-asyncio` (auto mode).

**Spec:** `docs/superpowers/specs/2026-09-11-india-equity-specialist-design.md`

## Global Constraints

- Fail loud everywhere — no `try`/`except` swallowing errors, except `pydantic-ai`'s bounded `Agent(retries=2)` validation-retry mechanism (per spec's Error Handling section).
- Both gap-computation functions raise on an unmapped, non-null sector string — never silently drop or miscount.
- Rows with `sector=None` are skipped, not treated as unmapped (known `MANUAL_TOTAL` data-quality caveat, not a real gap).
- Automated tests never spend real money: `pydantic_ai.models.test.TestModel` for every LLM-calling agent test; real `yfinance-mcp` and real `portfolio_data_mcp` calls are free and used as in every prior phase.
- Hard caps (3 candidates/sector, 5/run) enforced in code, never trusted from model output.
- One real, deliberate, cost-bounded manual run against the actual model pair (`deepseek/deepseek-v4.1-flash` / `z-ai/glm-5.3-flash`) is required before this phase counts as done — separate from the automated suite, reviewed by the user.
- Reuse project venv for all commands: `.venv/Scripts/python.exe`.

---

### Task 1: Generalize `ProposedCandidate` to the full GICS taxonomy

**Files:**
- Modify: `src/agentic_frameworks/graph/schemas.py`
- Test: `tests/graph/test_schemas.py`

**Interfaces:**
- Produces: `GICS_SECTORS: list[str]` (11 entries), `ProposedCandidate.sector: Literal[*GICS_SECTORS]` — consumed by Task 3 (US gap fix) and Task 4 (India gap function) and both specialists' candidate generators.

- [ ] **Step 1: Write failing test for the widened sector validation**

Add to `tests/graph/test_schemas.py`:

```python
def test_proposed_candidate_accepts_information_technology():
    from agentic_frameworks.graph.schemas import ProposedCandidate

    candidate = ProposedCandidate(ticker="TCS.NS", sector="Information Technology")

    assert candidate.sector == "Information Technology"


def test_gics_sectors_has_eleven_standard_buckets():
    from agentic_frameworks.graph.schemas import GICS_SECTORS

    assert len(GICS_SECTORS) == 11
    assert "Information Technology" in GICS_SECTORS
    assert "Energy" in GICS_SECTORS
    assert "Real Estate" in GICS_SECTORS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_schemas.py -v`
Expected: `test_gics_sectors_has_eleven_standard_buckets` fails with `ImportError: cannot import name 'GICS_SECTORS'`; `test_proposed_candidate_accepts_information_technology` fails with `ValidationError` (since `"Information Technology"` isn't in the current `US_GAP_SECTORS`-based `Literal`).

- [ ] **Step 3: Implement the schema change**

In `src/agentic_frameworks/graph/schemas.py`, replace:

```python
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


class ProposedCandidate(BaseModel):
    """One ticker the candidate generator proposes for a specific gap sector."""

    ticker: str
    sector: Literal[*US_GAP_SECTORS]
```

with:

```python
GICS_SECTORS = [
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
    "Information Technology",
]


class ProposedCandidate(BaseModel):
    """One ticker the candidate generator proposes for a specific gap sector.

    sector is validated against the full 11-bucket GICS taxonomy
    (GICS_SECTORS), not a hardcoded subset - which sectors are actual
    gaps for a given specialist/portfolio is computed live at runtime
    (see _compute_us_gaps, _compute_india_equity_gaps), never baked
    into this schema.
    """

    ticker: str
    sector: Literal[*GICS_SECTORS]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_schemas.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/schemas.py tests/graph/test_schemas.py
git commit -m "feat: generalize ProposedCandidate to full GICS taxonomy

Fixes a real bug: the old US_GAP_SECTORS-based Literal omitted
Information Technology, since it was hardcoded at a point in time
when the portfolio already held IT. sector validation now covers all
11 standard GICS buckets; which sectors are gaps is computed live
downstream instead."
```

---

### Task 2: Extract shared candidate-cap helper

**Files:**
- Create: `src/agentic_frameworks/graph/nodes/_specialist_helpers.py`
- Modify: `src/agentic_frameworks/graph/nodes/us_equity_etf_specialist.py`
- Test: Create `tests/graph/test_specialist_helpers.py`
- Test: Modify `tests/graph/test_us_equity_etf_specialist.py` (remove the moved test)

**Interfaces:**
- Produces: `MAX_CANDIDATES_PER_SECTOR: int`, `MAX_CANDIDATES_TOTAL: int`, `_apply_candidate_cap(candidates: list[ProposedCandidate]) -> list[ProposedCandidate]` — consumed by Task 3's `us_equity_etf_specialist_node` (already does, via updated import) and Task 6's `india_equity_specialist_node`.

- [ ] **Step 1: Write failing test in the new test file**

Create `tests/graph/test_specialist_helpers.py`:

```python
"""Tests for cap-enforcement logic shared across specialist nodes."""

from __future__ import annotations


def test_apply_candidate_cap_enforces_per_sector_and_total_limits():
    from agentic_frameworks.graph.nodes._specialist_helpers import (
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_specialist_helpers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_frameworks.graph.nodes._specialist_helpers'`

- [ ] **Step 3: Create the shared helpers module**

Create `src/agentic_frameworks/graph/nodes/_specialist_helpers.py`:

```python
"""Cap-enforcement logic shared across all new-buy specialist nodes.

Moved out of us_equity_etf_specialist.py once the India Equity
Specialist needed the identical logic - kept in one place so the two
specialists' cap behavior can never silently drift apart.
"""

from __future__ import annotations

from agentic_frameworks.graph.schemas import ProposedCandidate

MAX_CANDIDATES_PER_SECTOR = 3
MAX_CANDIDATES_TOTAL = 5


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

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_specialist_helpers.py -v`
Expected: PASS

- [ ] **Step 5: Remove the duplicated definition from us_equity_etf_specialist.py and import from the shared module instead**

In `src/agentic_frameworks/graph/nodes/us_equity_etf_specialist.py`, remove:

```python
MAX_CANDIDATES_PER_SECTOR = 3
MAX_CANDIDATES_TOTAL = 5
```

and the `_apply_candidate_cap` function definition (currently right below `_verify_ticker`). Add to the imports at the top of the file:

```python
from agentic_frameworks.graph.nodes._specialist_helpers import _apply_candidate_cap
```

- [ ] **Step 6: Remove the now-duplicated test from test_us_equity_etf_specialist.py**

In `tests/graph/test_us_equity_etf_specialist.py`, delete `test_apply_candidate_cap_enforces_per_sector_and_total_limits` (it now lives in `test_specialist_helpers.py`).

- [ ] **Step 7: Run the full existing specialist test file to confirm nothing broke**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_us_equity_etf_specialist.py tests/graph/test_specialist_helpers.py -v`
Expected: all PASS (the moved cap test passes from its new location; every other Phase 4 test still passes using the imported `_apply_candidate_cap`).

- [ ] **Step 8: Commit**

```bash
git add src/agentic_frameworks/graph/nodes/_specialist_helpers.py src/agentic_frameworks/graph/nodes/us_equity_etf_specialist.py tests/graph/test_specialist_helpers.py tests/graph/test_us_equity_etf_specialist.py
git commit -m "refactor: extract candidate-cap enforcement into shared helper

Both the US and (upcoming) India specialists need identical
3-per-sector/5-total cap logic. Sharing one implementation means it
can't silently drift between the two."
```

---

### Task 3: Fix US gap detection to be live-computed (retrofit)

**Files:**
- Modify: `src/agentic_frameworks/graph/nodes/us_equity_etf_specialist.py`
- Test: Modify `tests/graph/test_us_equity_etf_specialist.py`

**Interfaces:**
- Consumes: `GICS_SECTORS` from Task 1.
- Produces: `US_SECTOR_TO_GICS: dict[str, str]`, `_compute_us_gaps(holdings_by_bucket: dict) -> list[str]` — consumed by `us_equity_etf_specialist_node` (replacing the old `US_GAP_SECTORS` join).

- [ ] **Step 1: Write failing tests against real holdings data**

Add to `tests/graph/test_us_equity_etf_specialist.py`:

```python
async def test_compute_us_gaps_excludes_information_technology_given_real_holdings():
    from agentic_frameworks.graph.nodes.research import research_node
    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import (
        _compute_us_gaps,
    )

    research_result = await research_node({})

    gaps = _compute_us_gaps(research_result["holdings_by_bucket"])

    assert "Information Technology" not in gaps
    assert "Energy" in gaps
    assert "Health Care" in gaps
    assert len(gaps) == 10


def test_compute_us_gaps_raises_on_unmapped_sector():
    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import (
        _compute_us_gaps,
    )

    holdings_by_bucket = {
        "equity": {
            "US": [{"ticker": "ZZZ", "name": "Fictional Co", "sector": "Not A Real Sector"}],
        }
    }

    with __import__("pytest").raises(KeyError):
        _compute_us_gaps(holdings_by_bucket)


def test_compute_us_gaps_skips_null_sector_rows():
    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import (
        _compute_us_gaps,
    )

    holdings_by_bucket = {
        "equity": {
            "US": [{"ticker": "MANUAL_TOTAL", "name": "Manual Total", "sector": None}],
        }
    }

    gaps = _compute_us_gaps(holdings_by_bucket)

    assert len(gaps) == 11
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_us_equity_etf_specialist.py -k compute_us_gaps -v`
Expected: FAIL with `ImportError: cannot import name '_compute_us_gaps'`

- [ ] **Step 3: Implement `US_SECTOR_TO_GICS` and `_compute_us_gaps`**

In `src/agentic_frameworks/graph/nodes/us_equity_etf_specialist.py`, add (after the imports, before `_verify_ticker`):

```python
US_SECTOR_TO_GICS = {
    "Technology": "Information Technology",
    "Semiconductors": "Information Technology",
}


def _compute_us_gaps(holdings_by_bucket: dict[str, dict[str, list[dict]]]) -> list[str]:
    """Live-compute GICS sectors with zero coverage in real US equity holdings.

    Equity only - US ETF `sector` values (e.g. "US Large Cap", "Gold")
    are asset-classification labels, not real GICS sectors, and
    including them would mean fabricating sector coverage that isn't
    actually known. Rows with sector=None are skipped (manual-total
    platform rows, a known data-quality case), not treated as
    unmapped. Any other sector string not in US_SECTOR_TO_GICS raises -
    fail loud rather than silently miscounting.
    """
    covered: set[str] = set()
    for holding in holdings_by_bucket.get("equity", {}).get("US", []):
        sector = holding["sector"]
        if sector is None:
            continue
        covered.add(US_SECTOR_TO_GICS[sector])
    return [bucket for bucket in GICS_SECTORS if bucket not in covered]
```

Add `GICS_SECTORS` to this file's import from `schemas`:

```python
from agentic_frameworks.graph.schemas import (
    GICS_SECTORS,
    CandidateEvaluation,
    ProposedCandidate,
)
```

(replacing the old `US_GAP_SECTORS` import.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_us_equity_etf_specialist.py -k compute_us_gaps -v`
Expected: all PASS.

- [ ] **Step 5: Wire `_compute_us_gaps` into the node, replacing the old hardcoded join**

In `src/agentic_frameworks/graph/nodes/us_equity_etf_specialist.py`, inside `us_equity_etf_specialist_node`, replace:

```python
        generator = _build_candidate_generator(model)
        gap_sectors_text = ", ".join(US_GAP_SECTORS)
        generation_result = await generator.run(
            f"Propose candidates for these gap sectors: {gap_sectors_text}"
        )
```

with:

```python
        generator = _build_candidate_generator(model)
        gap_sectors = _compute_us_gaps(state["holdings_by_bucket"])
        gap_sectors_text = ", ".join(gap_sectors)
        generation_result = await generator.run(
            f"Propose candidates for these gap sectors: {gap_sectors_text}"
        )
```

- [ ] **Step 6: Update the existing node-level test's fixture to exercise the live gap computation**

In `tests/graph/test_us_equity_etf_specialist.py`, the existing `test_us_equity_etf_specialist_node_returns_candidate_evaluations` test already passes a `holdings_by_bucket` fixture with `equity.US` and `etf.US` — no change needed to the fixture itself, since `_compute_us_gaps` reads the same `holdings_by_bucket` shape the node already receives. Run the full file to confirm the node-level test still passes with the live computation wired in.

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_us_equity_etf_specialist.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/agentic_frameworks/graph/nodes/us_equity_etf_specialist.py tests/graph/test_us_equity_etf_specialist.py
git commit -m "fix: compute US specialist gap sectors live instead of hardcoded list

US_GAP_SECTORS was frozen at spike time and omitted Information
Technology, since the portfolio already held it then - a stale
constant, not a stable rule. _compute_us_gaps derives gaps fresh from
real holdings_by_bucket on every node call."
```

---

### Task 4: India equity gap computation (new)

**Files:**
- Create: `src/agentic_frameworks/graph/nodes/india_equity_specialist.py`
- Test: Create `tests/graph/test_india_equity_specialist.py`

**Interfaces:**
- Consumes: `GICS_SECTORS` from Task 1.
- Produces: `IN_SECTOR_TO_GICS: dict[str, str]`, `_compute_india_equity_gaps(holdings_by_bucket: dict) -> list[str]` — consumed by Task 6's `india_equity_specialist_node`.

- [ ] **Step 1: Write failing tests against real holdings data**

Create `tests/graph/test_india_equity_specialist.py`:

```python
"""Tests for the India Equity Specialist node and its stages."""

from __future__ import annotations


async def test_compute_india_equity_gaps_finds_energy_and_real_estate_given_real_holdings():
    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _compute_india_equity_gaps,
    )
    from agentic_frameworks.graph.nodes.research import research_node

    research_result = await research_node({})

    gaps = _compute_india_equity_gaps(research_result["holdings_by_bucket"])

    assert set(gaps) == {"Energy", "Real Estate"}


def test_compute_india_equity_gaps_raises_on_unmapped_sector():
    import pytest

    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _compute_india_equity_gaps,
    )

    holdings_by_bucket = {
        "equity": {
            "IN": [{"ticker": "ZZZ.NS", "name": "Fictional Co", "sector": "Not A Real Sector"}],
        }
    }

    with pytest.raises(KeyError):
        _compute_india_equity_gaps(holdings_by_bucket)


def test_compute_india_equity_gaps_skips_null_sector_rows():
    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _compute_india_equity_gaps,
    )

    holdings_by_bucket = {
        "equity": {
            "IN": [{"ticker": "MANUAL_TOTAL", "name": "Manual Total", "sector": None}],
        }
    }

    gaps = _compute_india_equity_gaps(holdings_by_bucket)

    assert len(gaps) == 11
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_india_equity_specialist.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_frameworks.graph.nodes.india_equity_specialist'`

- [ ] **Step 3: Create the module with the sector mapping and gap function**

Create `src/agentic_frameworks/graph/nodes/india_equity_specialist.py`:

```python
"""The India Equity Specialist node: proposes, verifies, and evaluates
India equity new-buy candidates against the New-Buy Conviction Gate.

Same three-stage pipeline as the US Equity/ETF Specialist
(nodes/us_equity_etf_specialist.py): live gap detection -> candidate
generation -> ticker verification -> Conviction Gate evaluation.
Fails loud, same as every other node in this graph.
"""

from __future__ import annotations

from agentic_frameworks.graph.schemas import GICS_SECTORS

IN_SECTOR_TO_GICS = {
    "Industrial Equipment": "Industrials",
    "Financial Services": "Financials",
    "Pharmaceuticals": "Health Care",
    "Industrials": "Industrials",
    "Steel": "Materials",
    "Water & Environment": "Utilities",
    "Consumer Packaging": "Consumer Staples",
    "Renewables": "Utilities",
    "Power": "Utilities",
    "Auto Components": "Consumer Discretionary",
    "Information Technology": "Information Technology",
    "Banking": "Financials",
    "Agri & Sugar": "Consumer Staples",
    "Chemicals": "Materials",
    "Electrical Equipment": "Industrials",
    "Infrastructure": "Industrials",
    "Media & Entertainment": "Communication Services",
    "Travel & Tourism": "Consumer Discretionary",
}


def _compute_india_equity_gaps(holdings_by_bucket: dict[str, dict[str, list[dict]]]) -> list[str]:
    """Live-compute GICS sectors with zero coverage in real IN equity holdings.

    IN equity only - mutual funds and real estate are excluded (fund
    `sector` values are fund category, not GICS sector, and real
    estate isn't sector-tagged at all; including either would mean
    fabricating coverage that isn't actually known). Rows with
    sector=None are skipped (the two MANUAL_TOTAL platform-total rows,
    a known data-quality case, not a real gap). Any other sector
    string not in IN_SECTOR_TO_GICS raises - fail loud rather than
    silently miscounting.
    """
    covered: set[str] = set()
    for holding in holdings_by_bucket.get("equity", {}).get("IN", []):
        sector = holding["sector"]
        if sector is None:
            continue
        covered.add(IN_SECTOR_TO_GICS[sector])
    return [bucket for bucket in GICS_SECTORS if bucket not in covered]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_india_equity_specialist.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/nodes/india_equity_specialist.py tests/graph/test_india_equity_specialist.py
git commit -m "feat: add live IN equity gap detection

Rolls up real IN equity holdings' sector strings into the 11-bucket
GICS taxonomy and returns zero-coverage buckets, computed fresh from
holdings_by_bucket every call - confirmed against the real portfolio
to correctly identify Energy and Real Estate as the current gaps."
```

---

### Task 5: India holdings summary (portfolio-fit input)

**Files:**
- Modify: `src/agentic_frameworks/graph/nodes/india_equity_specialist.py`
- Test: Modify `tests/graph/test_india_equity_specialist.py`

**Interfaces:**
- Produces: `_summarize_india_equity_holdings(holdings_by_bucket: dict) -> str` — consumed by Task 6's `_build_conviction_gate_evaluator`.

- [ ] **Step 1: Write failing tests**

Add to `tests/graph/test_india_equity_specialist.py`:

```python
def test_summarize_india_equity_holdings_lists_real_in_equity():
    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _summarize_india_equity_holdings,
    )

    holdings_by_bucket = {
        "equity": {
            "IN": [{"ticker": "TCS", "name": "Tata Consultancy Services", "sector": "Information Technology"}],
            "US": [{"ticker": "GOOGL", "name": "Alphabet Inc", "sector": "Technology"}],
        },
    }

    summary = _summarize_india_equity_holdings(holdings_by_bucket)

    assert "TCS" in summary
    assert "GOOGL" not in summary


def test_summarize_india_equity_holdings_handles_no_in_holdings():
    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _summarize_india_equity_holdings,
    )

    summary = _summarize_india_equity_holdings({})

    assert "no India equities" in summary
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_india_equity_specialist.py -k summarize -v`
Expected: FAIL with `ImportError: cannot import name '_summarize_india_equity_holdings'`

- [ ] **Step 3: Implement the function**

Add to `src/agentic_frameworks/graph/nodes/india_equity_specialist.py`:

```python
def _summarize_india_equity_holdings(holdings_by_bucket: dict[str, dict[str, list[dict]]]) -> str:
    """Build a real, current summary of IN equity holdings for the portfolio-fit check.

    Built fresh from holdings_by_bucket on every node invocation - never
    a hardcoded list - so it automatically reflects whatever the user
    actually holds whenever this node runs.
    """
    lines = []
    for holding in holdings_by_bucket.get("equity", {}).get("IN", []):
        lines.append(f"- {holding['ticker']} ({holding['name']}), sector: {holding['sector']}")
    if not lines:
        return "The user currently holds no India equities."
    return "The user's current India equity holdings are:\n" + "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_india_equity_specialist.py -k summarize -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/nodes/india_equity_specialist.py tests/graph/test_india_equity_specialist.py
git commit -m "feat: add live IN equity holdings summary for portfolio-fit check"
```

---

### Task 6: India candidate generator, Conviction Gate evaluator, and node

**Files:**
- Modify: `src/agentic_frameworks/graph/nodes/india_equity_specialist.py`
- Test: Modify `tests/graph/test_india_equity_specialist.py`

**Interfaces:**
- Consumes: `get_specialist_model` (`agentic_frameworks.graph.models`), `get_yfinance_toolset` (`agentic_frameworks.graph.mcp_clients`), `_verify_ticker` (`agentic_frameworks.graph.nodes.us_equity_etf_specialist`, reused unchanged), `_apply_candidate_cap` (`agentic_frameworks.graph.nodes._specialist_helpers`, Task 2), `ProposedCandidate`/`CandidateEvaluation` (`agentic_frameworks.graph.schemas`), `GraphState`/`SpecialistState` (`agentic_frameworks.graph.state`).
- Produces: `_build_candidate_generator(model) -> Agent`, `_build_conviction_gate_evaluator(model, toolset, holdings_summary) -> Agent`, `india_equity_specialist_node(state: GraphState) -> SpecialistState` — the node is consumed by Task 8's `graph.py` wiring.

- [ ] **Step 1: Write failing tests**

Add to `tests/graph/test_india_equity_specialist.py`:

```python
async def test_india_candidate_generator_produces_proposed_candidates():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _build_candidate_generator,
    )

    generator = _build_candidate_generator(TestModel())
    result = await generator.run("propose candidates for: Energy")

    assert isinstance(result.output, list)
    assert all(hasattr(c, "ticker") and hasattr(c, "sector") for c in result.output)


async def test_india_conviction_gate_evaluator_produces_candidate_evaluation():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset
    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _build_conviction_gate_evaluator,
    )

    toolset = get_yfinance_toolset()
    async with toolset:
        evaluator = _build_conviction_gate_evaluator(
            TestModel(), toolset, "The user currently holds no India equities."
        )
        result = await evaluator.run("Evaluate ticker: RELIANCE.NS, gap sector: Energy")

        assert result.output.action in {"Buy", "Hold", "Sell", "Reduce"}
        assert isinstance(result.output.confidence, float)


async def test_india_equity_specialist_node_returns_candidate_evaluations():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes import india_equity_specialist as specialist_module

    def fake_get_specialist_model():
        return TestModel()

    state = {
        "holdings_by_bucket": {
            "equity": {
                "IN": [{"ticker": "TCS", "name": "Tata Consultancy Services", "sector": "Information Technology"}],
            },
        }
    }

    original = specialist_module.get_specialist_model
    specialist_module.get_specialist_model = fake_get_specialist_model
    try:
        result = await specialist_module.india_equity_specialist_node(state)
    finally:
        specialist_module.get_specialist_model = original

    assert "india_equity_candidates" in result
    assert isinstance(result["india_equity_candidates"], list)
    assert result["india_equity_candidates"]
    for candidate in result["india_equity_candidates"]:
        assert candidate["action"] in {"Buy", "Hold", "Sell", "Reduce"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_india_equity_specialist.py -k "generator or evaluator or node_returns" -v`
Expected: FAIL with `ImportError`/`AttributeError` for the not-yet-defined names.

- [ ] **Step 3: Implement the generator, evaluator, and node**

Add to `src/agentic_frameworks/graph/nodes/india_equity_specialist.py` (imports first, then the rest at the end of the file):

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets.filtered import FilteredToolset

from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset
from agentic_frameworks.graph.models import get_specialist_model
from agentic_frameworks.graph.nodes._specialist_helpers import _apply_candidate_cap
from agentic_frameworks.graph.nodes.us_equity_etf_specialist import _verify_ticker
from agentic_frameworks.graph.schemas import CandidateEvaluation, ProposedCandidate
from agentic_frameworks.graph.state import GraphState, SpecialistState
```

```python
_CANDIDATE_GENERATOR_SYSTEM_PROMPT = """You are an India equity research assistant for a personal \
long-term portfolio advisor. Given a list of sector gaps in the user's \
current India equity holdings, propose real, currently-traded \
NSE-listed stock tickers (using the .NS suffix, e.g. RELIANCE.NS) \
that could fill each gap - up to 3 tickers per gap sector.

Only propose tickers you are confident are real, currently-listed NSE \
securities. Do not propose delisted, speculative, or fictional tickers. \
For each candidate, name the ticker symbol (with .NS suffix) and the \
gap sector it addresses."""


def _build_candidate_generator(model) -> Agent:
    """Build the India candidate-generator Agent.

    retries=2: bounded per CLAUDE.md's LLM cost control rule 6 - never
    left at the library default.
    """
    return Agent(
        model,
        output_type=list[ProposedCandidate],
        system_prompt=_CANDIDATE_GENERATOR_SYSTEM_PROMPT,
        retries=2,
    )


_CONVICTION_GATE_SYSTEM_PROMPT_TEMPLATE = """You are evaluating one India equity new-buy candidate for a \
personal, long-term portfolio advisor, using the New-Buy Conviction \
Gate. Run these four checks in order, stopping at the first failure:

1. Portfolio-fit: does this candidate address a real, identified gap \
in the user's current holdings? {holdings_summary}
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


def _build_conviction_gate_evaluator(model, toolset: FilteredToolset, holdings_summary: str) -> Agent:
    """Build the India Conviction Gate evaluator Agent.

    Same structure as the US specialist's evaluator - holdings_summary
    baked into the system prompt for a real portfolio-fit check,
    toolsets=[toolset] for real fundamentals/valuation data via
    yfinance-mcp (already confirmed to resolve .NS tickers), retries=2
    per CLAUDE.md's LLM cost control rule 6.
    """
    system_prompt = _CONVICTION_GATE_SYSTEM_PROMPT_TEMPLATE.format(holdings_summary=holdings_summary)
    return Agent(
        model,
        output_type=CandidateEvaluation,
        system_prompt=system_prompt,
        toolsets=[toolset],
        retries=2,
    )


async def india_equity_specialist_node(state: GraphState) -> SpecialistState:
    """Propose, verify, and evaluate India equity new-buy candidates.

    Same three-stage pipeline as us_equity_etf_specialist_node, with
    gap sectors computed live from real holdings_by_bucket rather than
    a static file/hardcoded list. Fails loud: any LLM/tool/verification
    error propagates uncaught.
    """
    model = get_specialist_model()
    toolset = get_yfinance_toolset()
    holdings_summary = _summarize_india_equity_holdings(state["holdings_by_bucket"])

    async with toolset:
        generator = _build_candidate_generator(model)
        gap_sectors = _compute_india_equity_gaps(state["holdings_by_bucket"])
        gap_sectors_text = ", ".join(gap_sectors)
        generation_result = await generator.run(
            f"Propose candidates for these gap sectors: {gap_sectors_text}"
        )
        capped_candidates = _apply_candidate_cap(generation_result.output)

        verified_candidates = [
            candidate
            for candidate in capped_candidates
            if await _verify_ticker(candidate.ticker, toolset)
        ]

        evaluator = _build_conviction_gate_evaluator(model, toolset, holdings_summary)
        evaluations = []
        for candidate in verified_candidates:
            eval_result = await evaluator.run(
                f"Evaluate ticker: {candidate.ticker}, gap sector: {candidate.sector}"
            )
            evaluations.append(eval_result.output.model_dump())

    return {"india_equity_candidates": evaluations}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_india_equity_specialist.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/nodes/india_equity_specialist.py tests/graph/test_india_equity_specialist.py
git commit -m "feat: add India Equity Specialist candidate generation and Conviction Gate

Mirrors the US Equity/ETF Specialist's three-stage pipeline
(propose -> verify -> evaluate), reusing the same model pair,
yfinance-mcp toolset (already NSE-ticker-capable), and cap
enforcement - only gap detection and NSE-ticker framing differ."
```

---

### Task 7: Wire `india_equity_candidates` into shared state

**Files:**
- Modify: `src/agentic_frameworks/graph/state.py`
- Test: Modify `tests/graph/test_state.py`

**Interfaces:**
- Produces: `SpecialistState.india_equity_candidates: list[dict]`, `GraphState.india_equity_candidates: list[dict]` — consumed by Task 8's `graph.py`.

- [ ] **Step 1: Write failing tests**

Modify `tests/graph/test_state.py`: update `test_specialist_state_has_expected_keys` and `test_graph_state_has_expected_keys` to include the new key:

```python
def test_specialist_state_has_expected_keys():
    from agentic_frameworks.graph.state import SpecialistState

    hints = get_type_hints(SpecialistState)
    assert set(hints.keys()) == {"us_equity_etf_candidates", "india_equity_candidates"}
    assert hints["us_equity_etf_candidates"] == list[dict]
    assert hints["india_equity_candidates"] == list[dict]


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
    }
```

Add a new test asserting the new key specifically (mirroring the existing `test_graph_state_includes_us_equity_etf_candidates`):

```python
def test_graph_state_includes_india_equity_candidates():
    from agentic_frameworks.graph.state import GraphState

    hints = get_type_hints(GraphState)
    assert "india_equity_candidates" in hints
    assert hints["india_equity_candidates"] == list[dict]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_state.py -v`
Expected: `test_specialist_state_has_expected_keys`, `test_graph_state_has_expected_keys`, `test_graph_state_includes_india_equity_candidates` FAIL (missing key).

- [ ] **Step 3: Update state.py**

In `src/agentic_frameworks/graph/state.py`, modify `SpecialistState`:

```python
class SpecialistState(TypedDict):
    """State the new-buy specialist nodes produce.

    us_equity_etf_candidates: the US Equity/ETF Specialist's evaluated
        candidates, each a `CandidateEvaluation` dict.
    india_equity_candidates: the India Equity Specialist's evaluated
        candidates, each a `CandidateEvaluation` dict.
    """

    us_equity_etf_candidates: list[dict]
    india_equity_candidates: list[dict]
```

and `GraphState`, adding the field and updating its docstring:

```python
    us_equity_etf_candidates: list[dict]
    india_equity_candidates: list[dict]
```

(appended after the existing `us_equity_etf_candidates: list[dict]` line, with a corresponding docstring line: `india_equity_candidates: the India Equity Specialist's evaluated candidates, each a CandidateEvaluation dict.`)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_state.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_frameworks/graph/state.py tests/graph/test_state.py
git commit -m "feat: add india_equity_candidates to shared graph state"
```

---

### Task 8: Wire both specialists into the graph as parallel branches

**Files:**
- Modify: `src/agentic_frameworks/graph/graph.py`
- Test: Modify `tests/graph/test_graph.py`

**Interfaces:**
- Consumes: `india_equity_specialist_node` (Task 6), `us_equity_etf_specialist_node` (existing, from `nodes/us_equity_etf_specialist.py`).

- [ ] **Step 1: Write failing tests**

Modify `tests/graph/test_graph.py`:

```python
def test_build_graph_wires_research_risk_analyst_and_both_specialists():
    from agentic_frameworks.graph.graph import build_graph

    graph = build_graph()

    assert set(graph.nodes.keys()) == {
        "research",
        "risk_analyst",
        "us_equity_etf_specialist",
        "india_equity_specialist",
    }
```

(replacing the old `test_build_graph_wires_research_and_risk_analyst`)

Update `test_graph_runs_end_to_end_with_real_data_and_persists_checkpoint` to keep the automated e2e run free of real LLM spend, by monkeypatching both specialists' model getters to `TestModel` before invoking the compiled graph, and to expect the two new candidate-list keys in the result:

```python
async def test_graph_runs_end_to_end_with_real_data_and_persists_checkpoint():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.checkpointer import get_checkpointer
    from agentic_frameworks.graph.graph import build_graph
    from agentic_frameworks.graph.nodes import india_equity_specialist as india_module
    from agentic_frameworks.graph.nodes import us_equity_etf_specialist as us_module

    def fake_get_specialist_model():
        return TestModel()

    original_india = india_module.get_specialist_model
    original_us = us_module.get_specialist_model
    india_module.get_specialist_model = fake_get_specialist_model
    us_module.get_specialist_model = fake_get_specialist_model
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
            }
            assert len(result["holdings_by_bucket"]["real_estate"]["IN"]) == 1
            assert result["concentration"]["by_holding"]
            assert result["currency_exposure"]
            assert result["liquidity"]["illiquid"]["value_base"] > 0
            assert result["us_equity_etf_candidates"]
            assert result["india_equity_candidates"]

            state_snapshot = await compiled.aget_state(config)
            assert state_snapshot.values
            assert state_snapshot.config["configurable"]["thread_id"] == thread_id
    finally:
        india_module.get_specialist_model = original_india
        us_module.get_specialist_model = original_us
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_graph.py -v`
Expected: `test_build_graph_wires_research_risk_analyst_and_both_specialists` FAILs (node set mismatch — the two specialists aren't wired in yet); the e2e test FAILs (missing candidate keys in result).

- [ ] **Step 3: Wire both specialist nodes into `build_graph()`**

Replace the contents of `src/agentic_frameworks/graph/graph.py`:

```python
"""Builds the Portfolio Advisor Fleet graph: wires existing nodes together.

Research -> Risk Analyst -> {US Equity/ETF Specialist, India Equity
Specialist} in parallel. Both specialist branches currently dead-end
at END - fan-in to a Synthesizer arrives in a later phase.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agentic_frameworks.graph.nodes.india_equity_specialist import (
    india_equity_specialist_node,
)
from agentic_frameworks.graph.nodes.research import research_node
from agentic_frameworks.graph.nodes.risk_analyst import risk_analyst_node
from agentic_frameworks.graph.nodes.us_equity_etf_specialist import (
    us_equity_etf_specialist_node,
)
from agentic_frameworks.graph.state import GraphState


def build_graph() -> StateGraph:
    """Wire research, risk_analyst, and both new-buy specialists into an uncompiled StateGraph."""
    graph = StateGraph(GraphState)
    graph.add_node("research", research_node)
    graph.add_node("risk_analyst", risk_analyst_node)
    graph.add_node("us_equity_etf_specialist", us_equity_etf_specialist_node)
    graph.add_node("india_equity_specialist", india_equity_specialist_node)
    graph.add_edge(START, "research")
    graph.add_edge("research", "risk_analyst")
    graph.add_edge("risk_analyst", "us_equity_etf_specialist")
    graph.add_edge("risk_analyst", "india_equity_specialist")
    graph.add_edge("us_equity_etf_specialist", END)
    graph.add_edge("india_equity_specialist", END)
    return graph
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/graph/test_graph.py -v`
Expected: all PASS. (This test makes real `portfolio_data_mcp` and `yfinance-mcp` calls, which is expected and free — only the LLM calls are mocked via `TestModel`.)

- [ ] **Step 5: Run the entire test suite to confirm no regressions**

Run: `.venv/Scripts/python.exe -m pytest tests/graph -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agentic_frameworks/graph/graph.py tests/graph/test_graph.py
git commit -m "feat: wire both new-buy specialists into the graph as parallel branches

First real fan-out in this graph: US Equity/ETF Specialist and India
Equity Specialist both run off risk_analyst independently. Both still
dead-end at END until the Synthesizer (fan-in) exists in a later
phase."
```

---

### Task 9: Manual real-model verification run (phase-completion gate)

This task has no automated test — it's the same manual, cost-bounded, human-reviewed gate Phase 4 required before being marked done, per `CLAUDE.md`'s LLM cost control rules and the spec's Testing section.

- [ ] **Step 1: Run the real graph end-to-end against the actual model pair**

Run (no monkeypatching — this uses real `OPENROUTER_API_KEY` spend, deliberately):

```bash
.venv/Scripts/python.exe -c "
import asyncio
from agentic_frameworks.graph.checkpointer import get_checkpointer
from agentic_frameworks.graph.graph import build_graph

async def main():
    async with get_checkpointer() as checkpointer:
        compiled = build_graph().compile(checkpointer=checkpointer)
        result = await compiled.ainvoke({}, config={'configurable': {'thread_id': 'manual-phase5-verification'}})
        print('US candidates:', result['us_equity_etf_candidates'])
        print('India candidates:', result['india_equity_candidates'])

asyncio.run(main())
"
```

- [ ] **Step 2: Review output with the user**

Present the real candidate evaluations (tickers, actions, confidence, reasoning, risk_flags) for both specialists. Confirm: tickers are real NSE/US instruments, `confidence` values are within `[0.0, 1.0]`, reasoning references real gap sectors and real holdings, no fabricated data. This is the same check that caught Phase 4's out-of-bounds `confidence=7.0` bug — `TestModel`-only testing cannot catch a real model's actual output quality issues.

- [ ] **Step 3: Update `HANDOFF.md`**

Once the user confirms the manual run looks correct, update `HANDOFF.md`'s "What's actually done" and "What's explicitly NOT done yet" sections to reflect Phase 5's completion, following the same structure used for Phases 1-4. Commit this alongside no code changes:

```bash
git add HANDOFF.md
git commit -m "docs: update handoff for Phase 5 (India Equity Specialist) completion"
```

- [ ] **Step 4: Push `main`**

Per `CLAUDE.md`'s phased development rule — push after this phase's tests pass and it's merged, before the next phase's brainstorming starts:

```bash
git push origin main
```
