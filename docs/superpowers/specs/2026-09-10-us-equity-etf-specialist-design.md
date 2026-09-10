# US Equity/ETF Specialist Node — Design Spec

**Status:** Approved for planning
**Phase:** 4 of the Portfolio Advisor Fleet LangGraph build (see `plans/plan.md`)

## Goal

Build the first new-buy specialist node: `us_equity_etf_specialist_node`. Consumes the graph's existing state (holdings, allocation, risk views) and the gap-analysis doc, proposes US equity/ETF new-buy candidates for genuinely underrepresented sectors, verifies each candidate is a real instrument, and runs surviving candidates through the New-Buy Conviction Gate — producing a strictly validated recommendation per candidate. First node in this graph to call an LLM, first use of `pydantic-ai`, first use of `yfinance-mcp`.

## Scope

**In scope:**
- Candidate generation: an LLM call proposes up to 3 US equity/ETF tickers per gap sector (from `docs/portfolio-gap-analysis.md`'s US section), capped at 5 total candidates per run.
- Ticker verification: each proposed candidate is confirmed to be a real, current instrument via `yfinance-mcp` — deterministic, no LLM, before any further work happens on it.
- Conviction Gate evaluation: each verified candidate goes through portfolio-fit → fundamentals → valuation → style-fit, one candidate at a time, short-circuiting on the first failed check (reject-early).
- Structured output per candidate: `{action: Buy|Hold|Sell|Reduce, confidence: float, reasoning: str, risk_flags: list[str], entry_approach: LumpSum|Staggered|None, suggested_platform: str|None}`, validated by `pydantic-ai`.
- Model: `deepseek/deepseek-v4.1-flash` via OpenRouter, configurable (not hardcoded to this one string forever — a named constant, easy to change later).

**Out of scope (explicitly deferred):**
- India Equity Specialist, Mutual Fund Specialist, Synthesizer — separate future phases.
- LangGraph Store (cross-run memory of rejected candidates) — deferred again this phase, per the "one piece at a time" build discipline. `future-enhancements.md` gets this noted.
- Wiring this node into `build_graph()`'s actual `StateGraph` — this phase builds and tests the node standalone, same pattern as Research and Risk Analyst. Graph wiring (parallel fan-out across all three specialists) is a later phase once all three exist.
- Any fallback-model list — single model only this phase, per the user's explicit choice; `pydantic-ai`'s native fallback-model support is available later without a redesign.

## Background / Constraints

- **Cost control (per `CLAUDE.md`'s "LLM cost control" section) is binding, not advisory, for this phase:**
  - Model choice is explicit: `deepseek/deepseek-v4.1-flash`, a named constant.
  - Reject-early: cheap ticker verification (no LLM) happens before the Conviction Gate evaluator is ever called on a candidate.
  - Prompt caching: the Conviction Gate evaluator's system prompt (the 4-check rules) is static across every candidate in a run — written once, not regenerated per call.
  - Minimal context per call: the evaluator only receives the one candidate's data, not the full portfolio state.
  - Hard cap: 3 candidates per gap sector, 5 total per run — enforced in code, not left to the model.
  - Bounded retries: `pydantic-ai`'s `Agent(retries=...)` is set explicitly, not left at the library default.
  - Store (long-term memory) is deferred, so this phase does not yet get the "never re-research a rejected candidate" cost saving — noted as a known limitation, not a silent gap.
- **Live-verified integration (2026-09-10):** `yfinance-mcp` connects via `pydantic_ai.mcp.MCPToolset(FastMCPClient(StdioTransport(command="uvx", args=["yfmcp@latest"])))` — confirmed working end-to-end, including a real tool call, against the actual `yfinance-mcp` package.
- **Live-verified API shapes (2026-09-10, installed `pydantic-ai` version):** `Agent(model, *, output_type=..., system_prompt=..., retries=..., toolsets=[...])`; `OpenRouterModel(model_name, *, provider=OpenRouterProvider(api_key=...))`.
- **Testing tension, resolved explicitly:** every other node's tests hit real dependencies. This node calls a real, paid LLM, which conflicts with cost control if done on every test run. Resolution: automated tests use `pydantic_ai.models.test.TestModel` (proves the code is wired correctly — right tools, right schema, right cap enforcement — at zero cost) for the LLM-calling parts; ticker verification (deterministic, no LLM, and `yfinance-mcp` itself is free) is tested against the real MCP server, same standard as every other node. **Before this phase is considered done, one real, deliberate, cost-bounded manual run against the actual `deepseek/deepseek-v4.1-flash` model is required** — output captured and reviewed by Abhi — as a phase-completion gate, not part of the automated suite.

## Architecture

`us_equity_etf_specialist_node(state: GraphState) -> dict` composes three stages:
1. **Candidate generator** — a `pydantic-ai` `Agent` (model = `deepseek/deepseek-v4.1-flash`) given the US-relevant gaps from `docs/portfolio-gap-analysis.md` (read as static text, not re-derived), proposes ticker candidates. Output type: a list of ticker strings, capped at 3 per sector / 5 total in code after the call, not trusted from the model alone.
2. **Ticker verification** — for each proposed ticker, a direct `yfinance-mcp` tool call (`yfinance_get_ticker_info`) confirms it resolves to a real instrument. A ticker that doesn't resolve is dropped, not retried or guessed at.
3. **Conviction Gate evaluator** — a second `pydantic-ai` `Agent`, given one verified candidate at a time plus the 4-check rules as a static, cacheable system prompt, plus `yfinance-mcp` tools (fundamentals/valuation data) as a toolset. Output type: `CandidateEvaluation` (the strict schema above). Runs the 4 checks in order, short-circuiting (rejecting without further tool calls) on the first failure.

## Components

- **`src/agentic_frameworks/graph/mcp_clients.py`** (extended) — `get_yfinance_toolset() -> MCPToolset`, wrapping `FastMCPClient(StdioTransport(command="uvx", args=["yfmcp@latest"]))`. Same file as the existing `get_portfolio_data_client()`, since both are MCP client wiring.
- **`src/agentic_frameworks/graph/models.py`** (new) — `SPECIALIST_MODEL_NAME = "deepseek/deepseek-v4.1-flash"` constant; `get_specialist_model() -> OpenRouterModel`, reading `OPENROUTER_API_KEY` from the environment.
- **`src/agentic_frameworks/graph/schemas.py`** (new) — `CandidateEvaluation(BaseModel)`: `action: Literal["Buy", "Hold", "Sell", "Reduce"]`, `confidence: float`, `reasoning: str`, `risk_flags: list[str]`, `entry_approach: Literal["LumpSum", "Staggered"] | None`, `suggested_platform: str | None`.
- **`src/agentic_frameworks/graph/nodes/us_equity_etf_specialist.py`** (new) — the three-stage pipeline and `us_equity_etf_specialist_node`.

## Data Flow

1. Node reads `docs/portfolio-gap-analysis.md`'s US section (static file read, not re-derived from live data this phase).
2. Candidate generator `Agent` proposes tickers per gap; code enforces the 3-per-sector/5-total cap on the result.
3. Each candidate ticker is verified against `yfinance-mcp`; non-resolving tickers are dropped.
4. Each surviving candidate goes through the Conviction Gate evaluator `Agent`, one at a time, producing a `CandidateEvaluation`.
5. Node returns `{"us_equity_etf_candidates": [<CandidateEvaluation as dict>, ...]}`.

## Error Handling

Fail loud, same as every prior node — no `try`/`except` swallowing errors in the deterministic parts (ticker verification, cap enforcement). For the LLM calls specifically: `pydantic-ai`'s own retry-on-validation-failure mechanism (bounded via `retries=`) is the one sanctioned exception to "no retries anywhere" in this project, since it's validating *output correctness*, not swallowing a real error — if retries are exhausted, the exception propagates uncaught, same as everywhere else.

## Testing (Phase 4 gate — must pass before Phase 5 is planned)

- Ticker verification: real test against `yfinance-mcp` — a real ticker (e.g. a current US holding) resolves; a fabricated/delisted ticker does not, and is dropped, not silently kept.
- Candidate cap: a test proving the 3-per-sector/5-total cap is enforced in code even if a `TestModel`-driven generator call returns more than that.
- Candidate generator and Conviction Gate evaluator wiring: tested with `TestModel`, proving correct tool registration, correct output schema, correct system-prompt content — not proving model judgment quality.
- **Phase-completion gate (manual, not automated):** one real run against `deepseek/deepseek-v4.1-flash` via OpenRouter, output reviewed by Abhi, before this phase is marked done.

No Playwright needed — no UI/browser surface yet.
