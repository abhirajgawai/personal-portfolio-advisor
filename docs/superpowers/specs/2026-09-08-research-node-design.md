# Research Node — Design Spec

**Status:** Approved for planning
**Phase:** 1 of the Portfolio Advisor Fleet LangGraph build (see `plan.md`)

## Goal

Build the first node of the Portfolio Advisor Fleet graph: a standalone, callable `research_node` that fetches the real portfolio's current holdings via the existing `portfolio_data_mcp` MCP server and returns them grouped by `asset_class × geography`. This is the entry point every later node (Risk Analyst, the three new-buy specialists, Synthesizer) will eventually read from — but none of those nodes exist yet, so this phase delivers only the Research node itself, not a wired graph.

## Scope

**In scope:**
- A `ResearchState` `TypedDict` defining the state shape this node produces.
- An MCP client helper that launches `portfolio_data_mcp` as a stdio subprocess and exposes its tools.
- The `research_node` function itself: async, calls `latest_snapshot` and `allocation_summary`, groups the results.
- Tests against the real, read-only `portfolio.db` (no mocks) — per `CLAUDE.md`'s phased-testing rule.

**Out of scope (explicitly deferred):**
- Any other graph node (Risk Analyst, specialists, Synthesizer).
- Wiring this node into an actual `StateGraph` with edges.
- LangGraph Studio/LangSmith observability wiring.
- Pydantic AI validation (that's for specialist/Synthesizer *output*, not this raw data fetch).
- The new fundamentals/valuation MCP server (separate, later stack item).

## Background / Constraints

- `portfolio_data_mcp` (`E:\agentic-portfolio-dashboard\portfolio_data_mcp\server.py`) is a **separate project with its own venv** (`E:\agentic-portfolio-dashboard\.venv`). It is not importable from this project's environment — it must be launched as a subprocess using its own interpreter: `E:\agentic-portfolio-dashboard\.venv\Scripts\python.exe -m portfolio_data_mcp.server`, with that project's root as the subprocess `cwd`, communicating over stdio (MCP's `transport="stdio"`).
- The server connects to `portfolio.db` in strict read-only mode (`mode=ro` SQLite URI) — confirmed in its own source; this node must never bypass that server to touch the DB directly.
- Confirmed real data shape (queried directly from `portfolio.db` for this spec): `asset_class` values are exactly `equity | etf | mutual_fund | real_estate`. Current counts: equity/IN=36, equity/US=3, etf/IN=2, etf/US=3, mutual_fund/IN=9, real_estate/IN=1 (real estate has a static `market_value_base`, no live price feed, but is queryable through the same `position` table as everything else).

## Architecture

One async node function, `research_node`, using `langchain-mcp-adapters` (`MultiServerMCPClient`) to talk to `portfolio_data_mcp` over stdio. On invocation it calls the server's `latest_snapshot` tool (no filters — full holding list) and `allocation_summary` tool (aggregate totals/percentages), then groups the `latest_snapshot` rows into `holdings_by_bucket[asset_class][geography]` using each row's own `asset_class`/`geography` fields (regrouping only — no re-deriving categories).

**Why `langchain-mcp-adapters` over the raw `mcp` SDK:** it wraps MCP tools as ordinary callables consistent with how LangGraph/Pydantic AI expect tools later in the build, avoiding hand-rolled protocol glue that would need to be redone once the graph has more nodes calling more MCP servers.

## Components

- **`src/agentic_frameworks/graph/state.py`** — `ResearchState(TypedDict)`:
  - `holdings: list[dict]` — raw rows from `latest_snapshot` (platform, geography, ticker, name, currency, sector, asset_class, quantity, price, market_value_base, updated_at).
  - `allocation: dict` — raw `allocation_summary` output (totalBase, holdingsCount, byGeography, byPlatform, bySector, byAssetClass).
  - `holdings_by_bucket: dict[str, dict[str, list[dict]]]` — `{asset_class: {geography: [holding, ...]}}`.

- **`src/agentic_frameworks/graph/mcp_clients.py`** — a helper (e.g. `get_portfolio_data_tools()`) that configures `MultiServerMCPClient` with one server entry for `portfolio_data_mcp`: `command` = the sibling project's venv Python, `args` = `["-m", "portfolio_data_mcp.server"]`, `transport` = `"stdio"`, `cwd` = `E:\agentic-portfolio-dashboard`. Returns the loaded tool objects (callables by name, e.g. `latest_snapshot`, `allocation_summary`).

- **`src/agentic_frameworks/graph/nodes/research.py`** — `async def research_node(state: dict) -> dict`: gets the tools, awaits both calls, groups the holdings, returns `{"holdings": ..., "allocation": ..., "holdings_by_bucket": ...}`.

## Data Flow

1. `research_node` invoked (standalone, e.g. from a test or a small script — no graph yet).
2. `get_portfolio_data_tools()` launches `portfolio_data_mcp` as a subprocess via `MultiServerMCPClient`, discovers its tools.
3. Node calls `latest_snapshot()` (no args → full list, `DEFAULT_LIMIT=100` applies per the server's own default, all current holdings fit well under that) and `allocation_summary()`.
4. Node groups `latest_snapshot`'s `holdings` list into `holdings_by_bucket` by iterating each row and appending to `bucket[row["asset_class"]][row["geography"]]`.
5. Node returns the three-field state update.

## Error Handling

**Fail loud.** No `try`/`except` around the MCP calls or the subprocess launch. If the subprocess fails to start, the DB is locked/missing, or a tool call returns a malformed response, the exception propagates out of `research_node` uncaught. Rationale: there is no meaningful recommendation any downstream node could make from empty or fabricated holdings data, and `CLAUDE.md` already prohibits fabricated/placeholder data — silently continuing with partial state would violate that.

## Testing (Phase 1 gate — must pass before Phase 2 is planned)

Per `CLAUDE.md`'s phased-development rule, this phase is not done until these tests pass:

- Integration test that invokes `research_node` against the **real** `portfolio_data_mcp` server (real, read-only `portfolio.db`) — no mocks.
- Assertions:
  - All 4 asset classes present in `holdings_by_bucket`: `equity`, `etf`, `mutual_fund`, `real_estate`.
  - `holdings_by_bucket["real_estate"]["IN"]` has exactly 1 entry.
  - `holdings_by_bucket["equity"]` has both `"IN"` (36 entries) and `"US"` (3 entries) keys.
  - `holdings_by_bucket["etf"]` has `"IN"` (2 entries) and `"US"` (3 entries).
  - `holdings_by_bucket["mutual_fund"]["IN"]` has 9 entries.
  - `allocation["totalBase"]` is a positive number roughly matching the known portfolio total (~₹10.29M).
- A failure test: point the MCP client config at a nonexistent DB path (or an invalid command) and assert `research_node` raises rather than returning an empty/partial result.

No Playwright needed for this phase — there is no UI/browser surface yet (that applies once Studio or a future UI is in play).
