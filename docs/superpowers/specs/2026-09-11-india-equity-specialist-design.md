# India Equity Specialist Node — Design Spec

**Status:** Approved for planning
**Phase:** 5 of the Portfolio Advisor Fleet LangGraph build (see `plans/plan.md`)

## Goal

Build the second new-buy specialist node: `india_equity_specialist_node`. Same three-stage shape as Phase 4's US Equity/ETF Specialist (propose → verify → Conviction Gate), reusing `yfinance-mcp` (already confirmed to support NSE tickers via `.NS`) and the same model pair. The one real design change from Phase 4: gap sectors are **computed live from real holdings data**, not a hardcoded list — and this phase also retrofits that same fix onto the Phase 4 node, since Phase 4's hardcoded `US_GAP_SECTORS` was found (during this phase's design review) to have a real bug: it omits Information Technology, because IT wasn't a gap on the day the list was written, and the list never updates as the portfolio changes.

## Scope

**In scope:**
- `_compute_india_equity_gaps(holdings_by_bucket)` — a pure function (no LLM) that rolls up real IN-geography equity holdings by sector, maps each raw sector string to one of the 11 standard GICS buckets via a small constant mapping, and returns the GICS buckets with zero coverage. Computed fresh on every node call.
- `_compute_us_gaps(holdings_by_bucket)` — the equivalent fix applied to the existing US specialist, replacing the hardcoded `US_GAP_SECTORS`-as-gap-list usage. US-geography **equity only** (not ETF — see Background), via a small `US_SECTOR_TO_GICS` mapping.
- India candidate generation: an LLM call proposes up to 3 NSE-listed (`.NS` suffix) tickers per live-computed gap sector, capped at 5 total candidates per run — same cap logic as Phase 4, shared not duplicated.
- Ticker verification: reused as-is from Phase 4 (`_verify_ticker`, already ticker-format-agnostic).
- Conviction Gate evaluation: same 4-check structure as Phase 4, with a new `_summarize_india_equity_holdings` (IN-geography-equity-only) baked into the evaluator's system prompt for the portfolio-fit check.
- Structured output per candidate: `CandidateEvaluation`, reused unchanged from Phase 4.
- Schema consolidation: `ProposedCandidate` becomes shared between both specialists, `sector` typed against the full `GICS_SECTORS` taxonomy (11 buckets) rather than the old US-specific 10-item list — this is what fixes the IT-omission bug at the validation layer.
- Graph wiring: `india_equity_specialist_node` added to `build_graph()` as a second branch after `risk_analyst`, running independently of (not dependent on) `us_equity_etf_specialist_node` — first real fan-out in the graph, though both branches still dead-end at `END` since the Synthesizer doesn't exist yet.

**Out of scope (explicitly deferred):**
- Mutual Fund Specialist, Synthesizer — separate future phases.
- Whole-portfolio gap analysis (rolling up sector coverage across mutual funds/real estate too) — explicitly rejected this phase. Mutual funds are tagged by *category* (Flexi Cap, Hybrid, etc.) in `portfolio.db`, not by underlying GICS sector composition, so computing "is this sector already covered by a fund" would mean fabricating data that doesn't exist — a direct violation of the fail-loud/never-fabricate rule. Revisit once the Mutual Fund Specialist exists and can supply real fund-holdings sector data (e.g. via a factsheet API), not before.
- ETF sector data in gap computation, for both specialists — ETF `sector` values in `portfolio.db` are a mix of real sectors and non-sector labels (cap-size, geography, commodity), not reliably GICS-mappable. Both `_compute_india_equity_gaps` and `_compute_us_gaps` are equity-only. Known side effect: VOO's genuine diversified sector exposure won't suppress any "gap," same limitation the original gap-analysis doc already noted in prose — now enforced structurally instead of eyeballed.
- LangGraph Store (cross-run memory of rejected candidates) — deferred again this phase, consistent with Phase 4 and the user's explicit instruction to do it as its own phase after the next specialist.
- Fan-in / Synthesizer wiring — both specialist branches remain dead ends at `END` this phase.
- Deleting `docs/portfolio-gap-analysis.md` — it stays as the historical record of the spike that motivated this fix, but is no longer read by either specialist node at runtime.

## Background / Constraints

- **The bug this phase fixes:** `US_GAP_SECTORS` (`schemas.py`) has 10 entries; the standard GICS taxonomy has 11. Information Technology is missing — not an oversight in the list itself, but a consequence of hardcoding a point-in-time fact (the portfolio held Technology/Semiconductors on 2026-09-09) as if it were a stable rule. If US Technology holdings were ever sold, this hardcoded list would never surface IT as a gap. Confirmed live during this phase's design review.
- **Real sector data confirmed live from `portfolio.db` (read-only) this phase:**
  - IN equity sector strings (18, excluding two `sector=NULL` `MANUAL_TOTAL` platform-total rows — a known data-quality caveat per `docs/portfolio-gap-analysis.md`, not an unmapped-sector failure): Agri & Sugar, Auto Components, Banking, Chemicals, Consumer Packaging, Electrical Equipment, Financial Services, Industrial Equipment, Industrials, Information Technology, Infrastructure, Media & Entertainment, Pharmaceuticals, Power, Renewables, Steel, Travel & Tourism, Water & Environment.
  - US equity sector strings (2): Technology, Semiconductors.
  - ETF sector strings (5, excluded from gap computation per above): Gold, International Equity, Semiconductors, Small Cap, US Large Cap.
  - Mutual fund `sector` strings (5, confirmed to be fund *category* not GICS sector, excluded from gap computation per above): Flexi Cap, Hybrid, Index Fund, Mid Cap, Small Cap.
- **Fail-loud on unmapped sectors:** if `_compute_india_equity_gaps` or `_compute_us_gaps` encounters a non-null sector string not present in their respective mapping constants, they raise — never silently drop or miscount. This is a real, live possibility (a newly-added holding with a sector not seen before), not a hypothetical.
- **Cost control (per `CLAUDE.md`), same standard as Phase 4:** explicit named model constant (reused, not re-chosen), reject-early ticker verification before the Conviction Gate evaluator runs, cacheable static system prompt per run, minimal per-candidate context, hard caps enforced in code, bounded retries.
- **Model:** reused unchanged from Phase 4 — `deepseek/deepseek-v4.1-flash` primary, `z-ai/glm-5.3-flash` fallback via `pydantic_ai.models.fallback.FallbackModel`. No new model decision this phase.
- **`yfinance-mcp` already confirmed to support NSE tickers** (`.NS` suffix) live, from `plan.md`'s MCP decision log — no new integration work, same toolset (`get_yfinance_toolset()`) reused as-is.
- **Testing tension, resolved the same way as Phase 4:** `TestModel` for both LLM-calling agents (candidate generator, Conviction Gate evaluator) in the automated suite; one real, deliberate, cost-bounded manual run against the actual model pair, reviewed by the user, is required before this phase counts as done.

## Architecture

`india_equity_specialist_node(state: GraphState) -> dict` composes four stages (one more than Phase 4, since gap computation is now real work, not a static-file read):

1. **Gap computation** — `_compute_india_equity_gaps(state["holdings_by_bucket"])`, pure function, no LLM, no MCP call. Filters IN equity holdings, maps sector → GICS bucket, returns zero-coverage buckets.
2. **Candidate generator** — `pydantic-ai` `Agent` (shared model), given the live gap list, proposes NSE-listed ticker candidates. Cap enforced in code via the now-shared `_apply_candidate_cap` helper.
3. **Ticker verification** — `_verify_ticker`, reused unchanged from Phase 4, against `yfinance-mcp`.
4. **Conviction Gate evaluator** — `pydantic-ai` `Agent` (shared model), given one verified candidate at a time, the 4-check rules as a static system prompt, `_summarize_india_equity_holdings`'s real holdings summary baked in for the portfolio-fit check, and `yfinance-mcp` tools for fundamentals/valuation. Same reject-early short-circuit behavior as Phase 4.

The Phase 4 node (`us_equity_etf_specialist_node`) is refactored in step 1 only: `_compute_us_gaps(state["holdings_by_bucket"])` replaces the `US_GAP_SECTORS` join. Stages 2-4 are unchanged in shape, only now consuming the live-computed gap list instead of a hardcoded one.

## Components

- **`src/agentic_frameworks/graph/schemas.py`** (modified) — add `GICS_SECTORS` (the full 11-sector taxonomy, a stable public standard, not a business decision); change `ProposedCandidate.sector` to `Literal[*GICS_SECTORS]`; delete `US_GAP_SECTORS`.
- **`src/agentic_frameworks/graph/nodes/_specialist_helpers.py`** (new) — `_apply_candidate_cap`, moved here from `us_equity_etf_specialist.py` since both specialists now use it identically; `MAX_CANDIDATES_PER_SECTOR`/`MAX_CANDIDATES_TOTAL` constants move here too.
- **`src/agentic_frameworks/graph/nodes/us_equity_etf_specialist.py`** (modified) — add `_compute_us_gaps` and `US_SECTOR_TO_GICS`; remove the `US_GAP_SECTORS` join in `us_equity_etf_specialist_node`, call `_compute_us_gaps` instead; import `_apply_candidate_cap` from the new shared module instead of defining it locally.
- **`src/agentic_frameworks/graph/nodes/india_equity_specialist.py`** (new) — `IN_SECTOR_TO_GICS`, `_compute_india_equity_gaps`, `_summarize_india_equity_holdings`, `_build_candidate_generator`/`_build_conviction_gate_evaluator` (India-specific system prompts, same structure as Phase 4's), `india_equity_specialist_node`.
- **`src/agentic_frameworks/graph/state.py`** (modified) — `SpecialistState` and `GraphState` gain `india_equity_candidates: list[dict]`.
- **`src/agentic_frameworks/graph/graph.py`** (modified) — add `india_equity_specialist_node` as a node with an edge from `risk_analyst`, parallel to `us_equity_etf_specialist` (both edge to `END` independently for now).

## Data Flow

1. Node reads `state["holdings_by_bucket"]` (Research node's live output, already in the graph state — no new MCP call needed for gap computation itself).
2. `_compute_india_equity_gaps` returns the current zero-coverage GICS buckets.
3. Candidate generator `Agent` proposes NSE tickers per gap; `_apply_candidate_cap` enforces 3-per-sector/5-total.
4. Each candidate ticker is verified against `yfinance-mcp`; non-resolving tickers are dropped.
5. Each surviving candidate goes through the Conviction Gate evaluator `Agent`, producing a `CandidateEvaluation`.
6. Node returns `{"india_equity_candidates": [<CandidateEvaluation as dict>, ...]}`.

## Error Handling

Same as every prior node: fail loud, no `try`/`except` swallowing errors anywhere in the deterministic parts (gap computation, ticker verification, cap enforcement) — including the new fail-loud-on-unmapped-sector behavior in both gap-computation functions. `pydantic-ai`'s bounded retry-on-validation-failure (`retries=2`) remains the one sanctioned exception, same rationale as Phase 4.

## Testing (Phase 5 gate — must pass before Phase 6 is planned)

- `_compute_india_equity_gaps`: tested against the real current portfolio's `holdings_by_bucket` shape (via the real Research node output, not fabricated fixtures) — asserts the correct zero-coverage buckets given actual IN equity sector weights; a separate test asserts an unmapped sector string raises.
- `_compute_us_gaps`: same treatment — asserts Information Technology is now correctly excluded from the gap list (the bug fix), given the real current US equity holdings.
- `_apply_candidate_cap`: existing Phase 4 test moves with the function to the new shared module, unchanged in assertions.
- Ticker verification: no new tests needed — `_verify_ticker` is reused unchanged and already covered by Phase 4's real `yfinance-mcp` tests (including live NSE ticker resolution, since `yfinance-mcp`'s `.NS` support was already confirmed working).
- Candidate generator and Conviction Gate evaluator wiring (India): tested with `TestModel`, same standard as Phase 4 — correct tool registration, correct output schema, correct system-prompt content, non-empty candidate list assertion (per Phase 4's final-review fix).
- Schema change regression: existing Phase 4 tests referencing `US_GAP_SECTORS` or the old `ProposedCandidate` sector constraint are updated to reflect `GICS_SECTORS`.
- **Phase-completion gate (manual, not automated):** one real run of `india_equity_specialist_node` against the actual model pair via OpenRouter, output reviewed by the user, before this phase is marked done.

No Playwright needed — no UI/browser surface yet.
