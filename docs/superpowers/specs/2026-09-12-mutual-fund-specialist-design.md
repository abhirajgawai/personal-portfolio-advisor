# Mutual Fund Specialist Nodes — Design Spec

**Status:** Approved for planning
**Phase:** 6 of the Portfolio Advisor Fleet LangGraph build (see `plans/plan.md`)

## Goal

Build two new, single-purpose nodes covering mutual funds: `mutual_fund_holdings_analyzer_node` (judges your real 9 existing IN mutual fund holdings — hold/reduce/switch) and `mutual_fund_specialist_node` (proposes and evaluates new-buy fund candidates, mirroring the two equity specialists). First phase requiring a custom-built MCP server, picking up the "build an MCP server yourself" learning goal explicitly deferred from Phase 4's decision to reuse `yfinance-mcp`. First phase to do real LLM-driven judgment of *existing* holdings for any asset class (Risk Analyst measures risk across all assets, but never judges; the equity specialists only judge new-buy candidates, never existing holdings).

## Scope

**In scope:**
- New MCP server: `src/agentic_frameworks/mfapi_mcp/server.py`, wrapping `MFapi.in`'s search and scheme-NAV endpoints as two MCP tools, launched as a stdio subprocess in this project's own venv, built with the official MCP SDK's `mcp.server.fastmcp.FastMCP` (already installed transitively — no new dependency).
- A shared helpers module (`nodes/_mutual_fund_helpers.py`, mirroring the `nodes/_specialist_helpers.py` precedent): ISIN-matched scheme-code resolution, NAV-history-based metrics (rolling-window returns, volatility, max drawdown, track-record length), per-category benchmark-ticker lookup, cross-fund NAV-return correlation (redundancy signal).
- A curated `MF_CATEGORY_TAXONOMY` (~12 entries spanning equity cap-sizes, ELSS, hybrid types, and debt types) and `MF_CATEGORY_TO_TAXONOMY`, mapping the 5 real category strings currently present in `holdings_by_bucket["mutual_fund"]["IN"]` (Flexi Cap, Hybrid, Index Fund, Small Cap, Mid Cap) to it — same fail-loud-on-unmapped pattern as `IN_SECTOR_TO_GICS`.
- `mutual_fund_holdings_analyzer_node`: for each of the real 9 held funds, resolves its scheme code, fetches NAV history, computes raw metrics (pure, no LLM — mirrors the Risk Analyst's measure-not-judge pattern), then an LLM Conviction Gate evaluator (7 checks — see below) produces a `CandidateEvaluation` per fund.
- `mutual_fund_specialist_node`: live category-gap detection (pure function, from real holdings) → LLM proposes real, Direct-plan Indian mutual fund names per gap category (capped via the existing shared `_apply_candidate_cap`) → verification against real `MFapi.in` search results (reject hallucinated names, enforce Direct-plan resolution) → the same Conviction Gate evaluator.
- The redefined 7-check Conviction Gate for funds: portfolio-fit/redundancy → cost (qualitative Direct-vs-Regular only) → performance-vs-benchmark (rolling-window, where a verified benchmark exists) → risk-adjusted performance (volatility + max drawdown, not Sharpe) → cross-fund redundancy (NAV-return correlation) → track-record length → style-fit.
- Static tax-treatment domain knowledge (equity-fund vs. debt-fund capital-gains taxation, ELSS's 3-year lock-in) baked into the Conviction Gate's system prompt as fixed reference text — these are stable regulatory facts, not fetched data.
- New schema: `ProposedFundCandidate(scheme_name: str, category: Literal[*MF_CATEGORY_TAXONOMY])`. `CandidateEvaluation` is reused unchanged.
- New state fields: `existing_fund_evaluations: list[dict]`, `mutual_fund_candidates: list[dict]`.
- Graph wiring: both new nodes added as additional parallel branches off `risk_analyst`, alongside the two existing equity specialists — dead-ending at `END` like the others (no fan-in yet).

**Out of scope (explicitly deferred or disclosed as unavailable):**
- **Category peer-ranking** (comparing a fund against all peer funds in its category, not just one benchmark index) — explicitly deferred per the user's decision; a real, larger scope/cost expansion (many peer-fund fetches per evaluation), revisit later if the benchmark-only view proves insufficient.
- **True Sharpe/Sortino ratios** — no clean, free Indian risk-free-rate source confirmed available; volatility + max drawdown used as the risk-adjustment measure instead.
- **Client risk-profile/goals-based asset-allocation suitability** — no age/goals/risk-tolerance data exists anywhere in `portfolio.db` or elsewhere in this system; not fabricated, not guessed.
- **AMC/fund-manager governance or news monitoring** (SEBI actions, scheme mergers, manager departures) — a news-monitoring capability, same category as the general web-search tradeoff already rejected this session (unstructured data the model would have to interpret, no schema to fail loud against).
- **Numeric expense ratio and AUM** — confirmed unavailable from `MFapi.in`'s scheme metadata (7 fields only: `fund_house`, `scheme_type`, `scheme_category`, `scheme_code`, `scheme_name`, `isin_growth`, `isin_div_reinvestment`) and from AMFI's own official public NAV feed (same limited field set). Direct-vs-Regular plan is the qualitative substitute for cost.
- **Underlying portfolio holdings / top-10 concentration within a fund** — not exposed by any free source checked; a real portfolio-holdings-overlap check would need a paid data aggregator.
- **Exit-load specifics beyond ELSS's lock-in** — not exposed by `MFapi.in`; only ELSS's well-known, stable 3-year lock-in is included as static knowledge.
- **Recommendation Synthesizer / fan-in** — still not built, same state as after Phase 5.
- **LangGraph Store** — still deferred, per the user's standing instruction to make it its own phase after the Mutual Fund Specialist.

## Background / Constraints (all live-verified this session)

- **Real holdings data:** 9 IN mutual funds in `holdings_by_bucket["mutual_fund"]["IN"]`. The `ticker` field is actually an **ISIN** (e.g. `INF879O01027`), not an AMFI scheme code. The `sector` field holds exactly 5 distinct real category values: Flexi Cap, Hybrid, Index Fund (×4 funds), Small Cap (×2 funds), Mid Cap.
- **`MFapi.in` confirmed live:** rejects direct ISIN lookups (HTTP 400). Its `/mf/search?q=<name>` endpoint returns multiple near-identical variants (Direct/Regular plan, Growth/IDCW option) that must be disambiguated — a naive fuzzy-name match risks silently picking the wrong one. Each scheme's `/mf/<code>` response's `meta.isin_growth` field reliably carries that exact scheme's ISIN, enabling **deterministic** resolution: search by name, then match each candidate's `meta.isin_growth` against our known ISIN — never fuzzy text matching alone.
- **`MFapi.in`'s full meta schema confirmed** (exactly 7 fields, live-checked): `fund_house`, `scheme_type`, `scheme_category`, `scheme_code`, `scheme_name`, `isin_growth`, `isin_div_reinvestment` — no expense ratio, no AUM.
- **AMFI's own official public NAV feed** (`https://www.amfiindia.com/spages/NAVAll.txt`) confirmed live to carry the same limited field set (Scheme Code, ISIN, Scheme Name, Plan, Option, NAV, Date) — no expense ratio or AUM either, ruling out "we just didn't look in the right place."
- **Pre-built MFapi-wrapping MCP servers researched** (`mfapi-mcp-server`, sdutta0291's Mutual Fund MCP Server claiming auto-computed Sharpe/Beta/Alpha, `Mftool-MCP`, others) — decided against reuse: even the most capable option doesn't cover this project's actual requirements (ISIN-matched resolution against our exact holdings, category-gap detection against our curated taxonomy, cross-fund redundancy across our specific 9 funds), since those are business logic no generic tool would have. Building our own thin wrapper also fulfills the "build an MCP server yourself" learning goal explicitly deferred from Phase 4.
- **`mcp.server.fastmcp.FastMCP`** (the official MCP SDK's server-building class) confirmed already installed in this project's venv, transitively via `pydantic-ai`'s MCP client extras — no new dependency needed. (The separate third-party `fastmcp` PyPI package's own server extras are NOT installed and are not needed.)
- **Benchmark ticker availability verified live via `yfinance-mcp`** for all 9 real funds' categories: exact-index matches for HDFC BSE Sensex Index Fund (`^BSESN`) and UTI Nifty200 Momentum 30 Index Fund (`NIFTY200MOMENTM30.NS`); category-proxy matches for Flexi Cap (Nifty 500, `^CRSLDX`), Small Cap ×2 (Nifty Smallcap 250, `NIFTYSMLCAP250.NS`), Mid Cap (Nifty Midcap 150, `NIFTYMIDCAP150.NS`). Confirmed **NOT available** for HDFC Balanced Advantage Fund (Hybrid — no CRISIL Hybrid index on Yahoo), Edelweiss Nifty Midcap150 Momentum 50 Index Fund, and UTI Nifty 500 Value 50 Index Fund (both specialty/factor indices absent from Yahoo). Gap-category benchmark tickers (Large Cap, Multi Cap, ELSS, Conservative Hybrid, and the debt categories, needed for new-buy candidates) will be individually verified during implementation, same practice as every prior mapping this session — debt-category benchmarks are expected to mostly be unavailable.
- **No true risk-free-rate source confirmed available** for Sharpe/Sortino computation.
- **No client risk-profile/goals/age data exists** anywhere in `portfolio.db` or elsewhere in this system.

## Architecture

Two independent, single-purpose nodes, sharing common infrastructure via `nodes/_mutual_fund_helpers.py`:

1. **`mutual_fund_holdings_analyzer_node(state) -> {"existing_fund_evaluations": [...]}`**
   a. For each of the 9 real IN mutual fund holdings: resolve its scheme code (ISIN-matched search), fetch NAV history.
   b. Pure, no-LLM computation (mirrors the Risk Analyst's measure-not-judge pattern): rolling-window returns (1yr/3yr/since-inception), volatility, max drawdown, track-record length (from the oldest available NAV date), benchmark difference (where a verified ticker exists for that fund's category), and cross-fund NAV-return correlation across all 9 holdings (the redundancy signal).
   c. The Conviction Gate evaluator (7 checks, "existing holding" framing) receives the real computed metrics for one fund at a time, plus its redundancy signal against the other 8, and produces a `CandidateEvaluation` (Hold/Reduce/Sell/Buy-more) per fund.

2. **`mutual_fund_specialist_node(state) -> {"mutual_fund_candidates": [...]}`**
   a. Live category-gap detection (pure function) from the real 9 funds' categories against `MF_CATEGORY_TAXONOMY`.
   b. LLM candidate generator proposes real, Direct-plan Indian mutual fund names per gap category, capped 3/category and 5/total via the existing shared `_apply_candidate_cap`.
   c. Verification: each proposed name is searched against real `MFapi.in` results; a candidate is dropped unless a Direct-plan variant genuinely matching the proposed name/category is found — reject-early against hallucinated fund names, mirroring `_verify_ticker`'s role in the equity specialists.
   d. Surviving candidates get the same metrics computation (steps 1b's NAV/benchmark/volatility/drawdown/track-record pieces — cross-fund redundancy against the *existing* 9 funds is folded into the portfolio-fit check instead) and the same Conviction Gate evaluator.

## Components

- **`src/agentic_frameworks/mfapi_mcp/server.py`** (new) — the custom MCP server. `search_scheme(query: str) -> list[dict]` (scheme_code, scheme_name), `get_scheme_nav(scheme_code: int) -> dict` (meta + NAV history). Built with `mcp.server.fastmcp.FastMCP`, run via stdio.
- **`src/agentic_frameworks/graph/mcp_clients.py`** (modified) — `get_mfapi_toolset()`, launching the new server as a subprocess in this project's own venv (`sys.executable -m agentic_frameworks.mfapi_mcp.server`), with `tool_error_behavior="retry"` from the start (applying this session's lesson directly, not repeating the Phase 4/5 mistake).
- **`src/agentic_frameworks/graph/nodes/_mutual_fund_helpers.py`** (new) — `MF_CATEGORY_TAXONOMY`, `MF_CATEGORY_TO_TAXONOMY`, `MF_CATEGORY_BENCHMARK_TICKERS` (per-category, individually verified), `_resolve_scheme_code` (ISIN-matched search), `_fetch_nav_history`, `_compute_fund_metrics` (pure: rolling-window returns, volatility, max drawdown, track-record length), `_compute_benchmark_difference`, `_compute_redundancy_correlations`.
- **`src/agentic_frameworks/graph/nodes/mutual_fund_holdings_analyzer.py`** (new) — the existing-holdings node.
- **`src/agentic_frameworks/graph/nodes/mutual_fund_specialist.py`** (new) — the new-buy node.
- **`src/agentic_frameworks/graph/schemas.py`** (modified) — new `ProposedFundCandidate(BaseModel)`: `scheme_name: str`, `category: Literal[*MF_CATEGORY_TAXONOMY]`. `CandidateEvaluation` reused unchanged.
- **`src/agentic_frameworks/graph/state.py`** (modified) — `existing_fund_evaluations: list[dict]`, `mutual_fund_candidates: list[dict]` added to `SpecialistState`/`GraphState`.
- **`src/agentic_frameworks/graph/graph.py`** (modified) — both new nodes added as parallel branches off `risk_analyst`, alongside the two equity specialists, dead-ending at `END`.
- Conviction Gate system prompt additions — static tax-treatment reference text (equity-fund vs. debt-fund capital-gains taxation, ELSS's 3-year lock-in).

## Data Flow

**`mutual_fund_holdings_analyzer_node`:**
1. Reads `state["holdings_by_bucket"]["mutual_fund"]["IN"]` (9 real holdings).
2. `_resolve_scheme_code` searches `MFapi.in` by a cleaned fund name, matches candidates' `meta.isin_growth` against the holding's real ISIN, fails loud if no match.
3. `_fetch_nav_history` pulls the full NAV series for the resolved scheme code.
4. `_compute_fund_metrics` computes real, live-derived rolling-window returns/volatility/drawdown/track-record from the NAV series.
5. `_compute_benchmark_difference` fetches the category's benchmark ticker (if verified available) via `yfinance-mcp` and computes the return difference over the matching period.
6. `_compute_redundancy_correlations` computes NAV-return correlation between this fund and each of the other 8.
7. The Conviction Gate evaluator receives all of the above as real, computed context for one fund, produces a `CandidateEvaluation`.
8. Returns `{"existing_fund_evaluations": [<CandidateEvaluation as dict>, ...]}` (9 entries).

**`mutual_fund_specialist_node`:**
1. Reads `state["holdings_by_bucket"]["mutual_fund"]["IN"]`.
2. Live gap detection against `MF_CATEGORY_TAXONOMY`.
3. Candidate generator proposes fund names per gap category; capped in code.
4. Verification searches `MFapi.in` for each proposed name, resolves to its Direct-plan scheme code, drops candidates with no real match.
5. Metrics computation (data-flow steps 3-5 above) runs for each surviving candidate.
6. Conviction Gate evaluator produces a `CandidateEvaluation` per surviving candidate.
7. Returns `{"mutual_fund_candidates": [...]}`.

## Error Handling

Same fail-loud standard as every prior node: no `try`/`except` swallowing errors, except `pydantic-ai`'s bounded `Agent(retries=2)`. The new `mfapi` toolset is built with `tool_error_behavior="retry"` from the start (this session's Phase 5 lesson applied proactively, not re-learned). `_resolve_scheme_code` raises loud if no ISIN match is found among search results — never silently guesses a "close enough" scheme. Benchmark-unavailable categories are explicitly represented (e.g. `benchmark_difference: None` with a documented reason), never silently omitted or fabricated as zero.

## Testing (Phase 6 gate — must pass before Phase 7 is planned)

- New MCP server: tested with a real, free call to the actual `MFapi.in` API (search + scheme-NAV fetch for a real known scheme code) — no mocking, consistent with every other MCP integration in this project.
- `_resolve_scheme_code`: tested against real data — one of your actual 9 fund names/ISINs, asserting it resolves to the correct real scheme code; a fabricated/nonexistent fund name asserted to fail loud (raise), not silently return a wrong match.
- `_compute_fund_metrics`, `_compute_benchmark_difference`, `_compute_redundancy_correlations`: tested against real NAV history pulled live for one or more of your actual 9 funds.
- Category-gap detection: tested against real current holdings, asserting the correct gap set (Large Cap, Multi Cap, ELSS, Conservative Hybrid, and the debt categories).
- Candidate generator / Conviction Gate evaluator wiring: `TestModel` for both nodes' LLM-calling agents, same standard as every LLM-calling node so far — zero real spend in the automated suite.
- **Phase-completion gate (manual, not automated):** one real run of both new nodes against the actual model pair, output reviewed by the user, before this phase is marked done — same practice as Phases 4 and 5, which each caught a real bug no `TestModel`-only testing could have found.

No Playwright needed — no UI/browser surface yet.
