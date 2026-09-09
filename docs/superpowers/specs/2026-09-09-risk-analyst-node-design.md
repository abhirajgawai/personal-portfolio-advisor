# Risk Analyst Node — Design Spec

**Status:** Approved for planning
**Phase:** 2 of the Portfolio Advisor Fleet LangGraph build (see `plan.md`)

## Goal

Build the second node of the Portfolio Advisor Fleet graph: `risk_analyst_node`, a pure function that consumes the Research node's `ResearchState` output and computes a risk assessment of the *existing* portfolio — concentration, category overlap (a proxy for true diversification), currency/FX exposure, and liquidity — before any new-buy candidates are considered downstream.

## Scope

**In scope (what a real portfolio risk analyst does, for a long-term investor, filtered to what's computable from data the Research node already fetches):**
- Concentration — raw weights per holding and per asset class (no thresholds/verdicts).
- Category overlap — a heuristic proxy for true diversification, grouping holdings by their `sector` field (populated meaningfully across every asset class: equity sector, mutual-fund category, ETF theme, real-estate property type) and surfacing sectors with more than one holding.
- Currency/FX exposure — total value and % of portfolio per currency.
- Liquidity — a simple asset-class split: `real_estate` = illiquid, everything else = liquid.

**Out of scope (explicitly deferred — see `future-enhancements.md`):**
- True statistical correlation from historical price/return data.
- Horizon-fit risk (needs a stated time horizon, which doesn't exist yet).
- Drawdown context (needs historical price data Research doesn't fetch).
- Allocation drift detection (needs a stated target allocation, which doesn't exist yet).
- Fund-level lock-in nuance for liquidity (e.g. ELSS lock-in periods) — needs confirming whether the data even carries this.
- Any threshold/verdict logic ("this is too concentrated") — this node reports raw numbers only; judgment is a later concern (you, or a future Synthesizer node).

## Background / Constraints

- Confirmed live from `portfolio.db`: the `instrument` table's `sector` column is populated for every asset class, not just equities — e.g. mutual funds carry fund category ("Small Cap", "Index Fund", "Flexi Cap", "Hybrid"), ETFs carry their theme ("Semiconductors", "Gold", "US Large Cap"), real estate carries property type ("Residential"). This makes a single `sector`-based grouping usable as a diversification-overlap proxy across the whole portfolio, with no new data-fetching capability required.
- This node needs **no MCP client and no subprocess** — every computation is a pure transformation over `ResearchState`'s existing `holdings`/`allocation`/`holdings_by_bucket` fields.

## Architecture

`risk_analyst_node(state: ResearchState) -> RiskState` — a pure, synchronous function composed of four private helpers, each computing one risk view:
- `_concentration(holdings)` — weight of each holding as % of total portfolio and % of its own asset-class bucket, plus per-asset-class totals.
- `_category_overlap(holdings)` — groups holdings by `sector` (any asset class), keeping only sectors with more than one holding.
- `_currency_exposure(holdings)` — total value and % of portfolio per currency.
- `_liquidity_split(holdings)` — total value and % split between `real_estate` (illiquid) and everything else (liquid).

The node's signature intentionally mirrors `research_node`'s (state dict in, state dict out) so both compose the same way once wired into a `StateGraph`.

## Components

- **`src/agentic_frameworks/graph/state.py`** (extended) — `RiskState(TypedDict)`:
  - `concentration: dict` — `{"by_holding": [{"ticker", "name", "asset_class", "pct_of_portfolio", "pct_of_asset_class"}], "by_asset_class": [{"asset_class", "pct_of_portfolio"}]}`.
  - `category_overlap: dict` — `{"by_sector": {"<sector>": [{"ticker", "name", "asset_class"}]}}`, only sectors with >1 holding.
  - `currency_exposure: dict` — `{"<currency>": {"value_base", "pct_of_portfolio"}}`.
  - `liquidity: dict` — `{"illiquid": {"value_base", "pct_of_portfolio"}, "liquid": {"value_base", "pct_of_portfolio"}}`.

- **`src/agentic_frameworks/graph/nodes/risk_analyst.py`** — `risk_analyst_node` and its four private helpers.

## Data Flow

1. `risk_analyst_node` receives a `ResearchState` dict (from `research_node`, or a hand-built equivalent in tests).
2. Calls each of the four helpers against `state["holdings"]`.
3. Returns a `RiskState` dict combining all four results.

No re-fetching, no MCP calls — this is purely a downstream transformation of data the Research node already produced.

## Error Handling

**Fail loud**, same as the Research node. No `try`/`except` anywhere in this node or its helpers. Since this node performs no I/O, the only realistic failure mode is malformed input (e.g. a holding missing `sector` or `currency`); direct dict indexing (not `.get()`) surfaces that immediately as a `KeyError` rather than silently producing wrong numbers — consistent with the project's "no fabricated data" rule.

## Testing (Phase 2 gate — must pass before Phase 3 is planned)

Per `CLAUDE.md`'s phased-development rule, this phase is not done until these tests pass. Unlike the Research node, this node has no external dependency, so tests do **not** need the real MCP server — they use realistic, hand-built `ResearchState`-shaped input (a small holding set covering multiple asset classes, sectors, and currencies, built from real field names/values, not fabricated tickers) and assert exact computed values:

- Concentration: known weights sum to expected percentages for a hand-built holding set.
- Category overlap: a sector with 2+ holdings across different asset classes appears in `by_sector`; a sector with exactly 1 holding does not.
- Currency exposure: totals and percentages per currency are correct for a mixed-currency holding set.
- Liquidity: a holding set with one `real_estate` entry splits correctly into illiquid/liquid totals.
- A malformed-input test: a holding missing a required field (e.g. `sector`) causes `risk_analyst_node` to raise, not silently skip or default the field.

No Playwright needed — no UI/browser surface yet.
