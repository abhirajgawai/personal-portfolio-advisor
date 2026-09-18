# LangGraph Store (Cross-Run Memory) — Design Spec

**Status:** Draft for review (designed autonomously overnight 2026-09-18; needs Abhi's sign-off on the four Open Decisions before planning)
**Phase:** 7 of the Portfolio Advisor Fleet LangGraph build (see `plans/plan.md`)

## Goal

Give the graph a real long-term/cross-run memory — LangGraph's Store API, backed by the same `sandbox-data/agent_memory.db` the checkpointer already writes to — and use it as a **cost control**: a new-buy candidate that already failed the Conviction Gate in a prior run must not be re-verified, re-fetched, and re-evaluated (a paid LLM call) in the next run without new information.

This is the memory half `plan.md`'s "Harness & Memory" section always called for and that every phase since Phase 3 has explicitly deferred. Today nothing stops the US Equity/ETF, India Equity, and Mutual Fund specialists from paying for the exact same rejected candidate on every single run — and once Temporal runs this daily, that repeat cost becomes a daily cost.

## Scope

**In scope:**
- `src/agentic_frameworks/graph/store.py` — `get_memory_store()`, an async context manager yielding an `AsyncSqliteStore` pointed at the same `sandbox-data/agent_memory.db` path the checkpointer uses (path constant reused from `checkpointer.py`, not re-derived).
- `src/agentic_frameworks/graph/nodes/_candidate_memory.py` — a small, single-purpose helpers module (mirroring the `_specialist_helpers.py` / `_mutual_fund_helpers.py` precedent) holding: the namespace constants, the identity-key derivation functions, `_load_recent_rejections()`, `_filter_previously_rejected()`, and `_record_rejection()`.
- Read + write integration in the three **new-buy** specialist nodes: `us_equity_etf_specialist.py`, `india_equity_specialist.py`, `mutual_fund_specialist.py`.
- Store injection into those nodes via LangGraph's signature-based injection (`*, store: BaseStore | None`) — live-verified to work on the installed version (see Background).
- `graph.py`/call-site change: the compiled graph is compiled with `store=` as well as `checkpointer=`.
- A `RejectedCandidate` Pydantic model in `schemas.py` for the stored record, so what goes into the store is a validated schema and not an ad-hoc dict (same standard as every LLM output in this project).
- Tests: real `AsyncSqliteStore` against a temp SQLite file (free — no mocking), real `portfolio.db` via the existing path, `TestModel` for anything that would otherwise cost money.

**Out of scope (explicitly deferred, with reasons):**
- **The "cached style profile"** named in `plan.md`. **Nothing in this codebase computes a style profile today.** The style-fit check in all three specialists is static instructional prompt text ("consistent with a long-term, buy-and-hold approach"), not a derived sector/cap-size tilt. Caching a value that is never computed would mean inventing the computation inside a memory phase — silent scope growth, and a judgment-shaped feature smuggled into an infrastructure phase. The namespace `("style_profile", "portfolio")` is reserved in this design and left empty; the actual style-profile computation should be its own decision, most naturally alongside or after the Synthesizer.
- **`mutual_fund_holdings_analyzer_node`** gets no store integration at all. It judges the 9 *existing* holdings; its verdict on every held fund is the product the user reads each run. Suppressing a re-evaluation there wouldn't save wasted work, it would silently drop a holding from the report.
- **Caching hallucinated/unverifiable tickers** (a `("unverified_tickers", …)` namespace). Ticker verification is a free `yfinance-mcp` call, so the savings are latency-only; weighed against a second write path and a second staleness question, it is not worth it this phase. Revisit only if real runs show the generator repeatedly proposing the same fake symbols.
- **Store-based semantic/vector search.** `AsyncSqliteStore` supports a vector index (`SqliteIndexConfig`, `sqlite_vec`), but every lookup in this design is an exact-key `aget`/prefix `asearch`. No embeddings, no embedding cost, no new dependency.
- **TTL-based automatic expiry** as the *mechanism* for staleness — see the verified TTL quirk in Background. Staleness is enforced in our own code against a stored timestamp instead.
- Synthesizer, fan-in, human-approval gate, Studio/LangSmith, Temporal — all still their own later phases, untouched here.

## Background / Constraints (all live-verified against the installed versions this session)

- **Backend choice is not open-ended: exactly one shipped store can point at a SQLite file.** Installed: `langgraph 1.2.11`, `langgraph-checkpoint 4.2.0`, `langgraph-checkpoint-sqlite 3.1.1`. Walking `langgraph.store` in the project venv gives exactly `langgraph.store.base`, `langgraph.store.memory` (`InMemoryStore` — not durable, useless for cross-run memory), and `langgraph.store.sqlite` (`SqliteStore`, and `langgraph.store.sqlite.aio.AsyncSqliteStore`). Postgres stores live in a separate uninstalled package and would violate the "agent writes go to `agent_memory.db` only" guardrail anyway. **`AsyncSqliteStore` is the only correct option** — and it must be the async one, for the same reason the checkpointer is `AsyncSqliteSaver`: every node here is `async def` and the graph runs via `ainvoke`.
- **Checkpointer and Store can share the one `agent_memory.db` file — verified end-to-end, not assumed.** A probe graph with `AsyncSqliteSaver.from_conn_string(db)` and `AsyncSqliteStore.from_conn_string(db)` open simultaneously, **four parallel nodes** each writing to the store in the same superstep (the exact fan-out shape this graph has), ran clean with no `database is locked` errors, across two separate `ainvoke` runs. Resulting tables are disjoint: `checkpoints`, `writes` (saver) and `store`, `store_migrations` (store). The file is in `journal_mode=wal`, which is why concurrent readers/writers coexist.
- **Cross-process persistence verified**: a second Python process opened the same file and read back all 4 records written by the first — this really is cross-run memory, not per-process state.
- **Store injection into a node works by signature on this version** — `async def node(state, *, store: BaseStore)` receives the compiled graph's store automatically. Verified both ways: injected as `AsyncSqliteStore` when compiled with `store=`, and injected as **`None`** (no exception) when compiled without one. That `None` case is why the nodes must check-and-raise explicitly (see Error Handling). `langgraph.config.get_store()` also works inside a node, but it raises `RuntimeError: Called get_config outside of a runnable context` when a node function is called directly — which every existing node test does. Signature injection keeps those tests able to call nodes directly, passing a real store.
- **`await store.setup()` is not strictly required** — migrations were verified to run lazily on first `aput` against a brand-new file. It is still called explicitly at store creation so the schema exists deterministically before the first *read*.
- **TTL is half-wired on `AsyncSqliteStore` in this version — verified, and the reason this design does not rely on it.** `aput(..., ttl=…)` does persist `expires_at`/`ttl_minutes` columns, and `sweep_ttl()` genuinely deletes expired rows. But expired items are still returned by `aget`/`asearch` (verified: item still readable 2s after a 1.2s TTL, `omit_expired: True` set). Root cause found by reading the installed source: `AsyncSqliteStore.__mro__` is `AsyncSqliteStore → AsyncBatchedBaseStore → BaseStore → ABC → BaseSqliteStore`, and `BaseStore.supports_ttl = False` shadows `BaseSqliteStore.supports_ttl = True`, so the instance reports `supports_ttl == False` (the sync `SqliteStore` correctly reports `True`). **Consequence:** staleness must be enforced by our own code comparing a stored `rejected_at` timestamp, which is fail-loud, testable, and version-independent. A TTL is still written as belt-and-braces garbage collection, but nothing correctness-relevant depends on it.
- **Existing identity facts that determine what "the same candidate" means:** US/India equity candidates are already canonicalised by the time they are evaluated — `_verify_ticker` proves the exact symbol (`.NS`-suffixed for NSE) resolves to a real instrument via `yfinance-mcp` before anything expensive runs. Mutual funds are not: the LLM proposes a free-text `scheme_name`, and only `_resolve_scheme_by_name_and_direct_plan` (a free `MFapi.in` call) turns it into an authoritative `(scheme_code, category, real name)` — which is also where the Phase 6 review found that name-based matching alone can silently resolve to a different fund. This asymmetry is what forces the two-stage fund lookup below.

## Architecture

Three ideas, in order:

1. **One store, opened next to the checkpointer, injected into nodes.** `get_memory_store()` mirrors `get_checkpointer()` exactly — an `@asynccontextmanager` over `AsyncSqliteStore.from_conn_string(AGENT_MEMORY_DB_PATH)`. Call sites open both and compile with both. Nodes never construct or open a store themselves (same reason they never open the checkpointer): the store's lifetime is the run's, not a node's.

2. **"Rejected" means a Conviction Gate verdict that was not `Buy`.** A `Buy` is deliberately *never* suppressed: a live buy recommendation is the product, and its valuation/fundamentals context is time-sensitive — the user must see it fresh every run. Only Hold/Reduce/Sell verdicts on a *new-buy candidate* are worth remembering as "we already paid to find out this isn't worth buying."

3. **Suppression runs before the cap, never after.** Order in each specialist becomes: live gap filter → **rejection filter** → `_apply_candidate_cap` (3/group, 5/run) → verification → paid evaluation. Rationale in the section below; the short version is that memory must free budget for *fresh* ideas, not shrink the funnel.

### Namespace and record design

LangGraph Store namespaces are tuples; keys are strings within a namespace.

| Namespace | Key (identity) | Written by |
|---|---|---|
| `("rejected_candidates", "us_equity")` | verified ticker, upper-cased (e.g. `XLV`) | `us_equity_etf_specialist_node` |
| `("rejected_candidates", "india_equity")` | verified `.NS` ticker, upper-cased (e.g. `ONGC.NS`) | `india_equity_specialist_node` |
| `("rejected_candidates", "mutual_fund")` | resolved AMFI `scheme_code` as a string (e.g. `"119551"`) | `mutual_fund_specialist_node` |
| `("rejected_candidates", "mutual_fund", "by_name")` | normalised proposed scheme name | `mutual_fund_specialist_node` (alias record) |
| `("style_profile", "portfolio")` | — reserved, unused this phase — | — |

**What counts as "the same candidate":**

- **Equity/ETF: the ticker alone, not ticker+sector.** A ticker is a single security; it does not become a different candidate because it was proposed under a different gap label. The gap sector it was proposed for is stored as *metadata* (so the record is auditable and the Synthesizer can read it later), but it is not part of the identity. The portfolio-fit concern this raises — a candidate rejected while a sector was a gap should not stay suppressed after the portfolio changes — is already handled structurally: gap sectors are computed live each run, and `_filter_to_gap_sectors` drops off-gap candidates *before* the rejection filter is even consulted.
- **Mutual funds: the AMFI `scheme_code` is the identity** (equivalently the fund's ISIN — `scheme_code` is what the node already holds after resolution, and Phase 6 established the ISIN↔scheme_code mapping is deterministic). The LLM-proposed free-text name is **not** an identity; it is only a cheap alias index, because two runs can name the same fund slightly differently.
- Because of that, funds get a **two-stage lookup**: (a) a cheap pre-cap filter on the normalised proposed name, which catches the common exact-repeat case for free; (b) a second filter on `scheme_code` after `_resolve_scheme_by_name_and_direct_plan` (a free `MFapi.in` call) and before the paid evaluator runs. Stage (b) is the authoritative one; stage (a) just saves a few free API round-trips. Both alias and canonical records are written on rejection.
- Name normalisation for the alias key: case-fold, collapse whitespace, strip punctuation, and drop the noise words `fund / direct / plan / growth / scheme`. Deliberately conservative — a normalisation miss costs one free resolution call and is then caught by stage (b); an over-aggressive normalisation could collide two genuinely different funds, which is the failure that actually matters.

**Stored record** (`RejectedCandidate` in `schemas.py`, stored via `.model_dump()`):

```
identity: str              # the key, stored redundantly so a search hit is self-describing
specialist: str            # "us_equity" | "india_equity" | "mutual_fund"
display_name: str          # resolved scheme name, or the ticker for equities
group: str                 # gap sector (GICS) or fund category it was proposed for
action: str                # the non-Buy verdict: Hold | Reduce | Sell
confidence: float
risk_flags: list[str]
reasoning: str             # the evaluator's own reasoning, verbatim — never a re-summary
rejected_at: str           # ISO-8601 UTC, the staleness clock
thread_id: str | None      # which run produced it, for audit/replay in Studio later
schema_version: int        # 1 — so a future shape change can be detected, not guessed at
```

**Staleness:** a rejection suppresses re-evaluation only while `rejected_at` is within `REJECTION_MEMORY_DAYS` (proposed default **90**, a module constant). Older records are ignored on read (and a matching TTL is written so `sweep_ttl()` can eventually reclaim them). 90 days is the proposed value because these are long-term-investing verdicts driven by fundamentals/valuation and by a portfolio that changes slowly — roughly one reporting quarter is the interval after which "we already looked at this" stops being true. It is one constant in one place, trivially changed.

### Interaction with the existing caps (3/sector, 5/run)

**A previously-rejected, filtered-out candidate does not consume a cap slot** — the rejection filter runs *before* `_apply_candidate_cap`.

Why that is the right way round: the caps exist to bound spend per run (LLM cost control rule 5). If suppressed candidates still consumed slots, then on a run where the generator repeated 5 old rejects, the graph would evaluate *zero* new ideas and produce nothing — memory would have degraded the product without saving anything that wasn't already going to be free. Filtering first means the cap is always filled with *evaluable, not-yet-judged* candidates.

The cost ceiling is unchanged and is never raised by this design: at most 5 evaluator calls per specialist per run, exactly as today. The saving shows up as the runs where the generator can only produce old names — those runs now cost less instead of the same. The honest limitation to state plainly: memory here reduces *wasted* spend; it does not reduce the *maximum* spend.

Secondary, optional optimisation (recommended, but explicitly listed as a decision): pass up to 20 recently-rejected identifiers for that specialist into the **candidate generator's** prompt as a "do not propose these" list, so the generator spends its one call producing fresh names rather than names that will be discarded. This is a prompt-level optimisation only — the in-code filter stays authoritative, per the "guardrails are structural, not instructional" principle. Token cost is ~20 short strings.

## Components

- **`src/agentic_frameworks/graph/store.py`** (new) — `get_memory_store()`, an `@asynccontextmanager` yielding a ready `AsyncSqliteStore` on `AGENT_MEMORY_DB_PATH` (imported from `checkpointer.py`, not redefined), calling `await store.setup()` before yielding.
- **`src/agentic_frameworks/graph/nodes/_candidate_memory.py`** (new) — `REJECTION_MEMORY_DAYS`, `MAX_EXCLUSION_HINTS`, the namespace constants, `_normalize_scheme_name(name) -> str`, `_is_fresh(record) -> bool`, `_load_recent_rejections(store, namespace) -> dict[str, dict]` (one prefix `asearch`, freshness-filtered), `_filter_previously_rejected(candidates, rejections, key) -> list` (pure, mirrors `_filter_to_gap_sectors`'s `key=` shape exactly), `_record_rejection(store, namespace, record)`.
- **`src/agentic_frameworks/graph/schemas.py`** (modified) — `RejectedCandidate(BaseModel)` as above. `CandidateEvaluation` unchanged.
- **`src/agentic_frameworks/graph/nodes/us_equity_etf_specialist.py`**, **`india_equity_specialist.py`**, **`mutual_fund_specialist.py`** (modified) — signature gains `*, store: BaseStore | None = None`; read/write points as in Data Flow.
- **`src/agentic_frameworks/graph/graph.py`** — unchanged (it returns an *uncompiled* graph; the store, like the checkpointer, is attached at the call site). Call sites and `tests/graph/test_graph.py` compile with `store=` as well as `checkpointer=`.
- **`src/agentic_frameworks/graph/nodes/mutual_fund_holdings_analyzer.py`** — unchanged, deliberately.

## Data Flow

**US / India equity specialist (identical shape):**
1. Live gap computation; empty ⇒ return early (unchanged).
2. `store` is `None` ⇒ raise (see Error Handling).
3. `rejections = await _load_recent_rejections(store, ("rejected_candidates", "<specialist>"))` — one local SQLite read, before any model or toolset setup.
4. Generator runs, now with a bounded "already rejected, do not propose" hint list built from `rejections`.
5. `_filter_to_gap_sectors` (unchanged) → **`_filter_previously_rejected(..., key=lambda c: c.ticker.upper())`** (new) → `_apply_candidate_cap` (unchanged).
6. `_verify_ticker` per surviving candidate (unchanged).
7. Conviction Gate evaluator per verified candidate (unchanged). **After each verdict**, if `action != "Buy"`, `await _record_rejection(...)` with the real ticker, group, verdict, reasoning, and `rejected_at=now(UTC)`.
8. Return the evaluations (unchanged shape — `SpecialistState` is not modified this phase).

**Mutual fund specialist:**
1. Live category-gap computation; empty ⇒ return early (unchanged).
2. Load both `("rejected_candidates", "mutual_fund")` and `…("by_name")` rejection maps.
3. Generator runs with the bounded exclusion hint list (resolved fund names).
4. `_filter_to_gap_sectors(key=category)` → **name-alias rejection filter** → `_apply_candidate_cap(key=category)`.
5. Per surviving candidate: `_resolve_scheme_by_name_and_direct_plan` (free) → `_category_matches_taxonomy` guard (unchanged) → **`scheme_code` rejection filter — the authoritative one, the last gate before anything paid** → NAV history + metrics + benchmark (unchanged) → evaluator.
6. On a non-`Buy` verdict, write **both** records: canonical (`scheme_code` key) and alias (normalised proposed name key), both carrying the resolved real scheme name.
7. Return the evaluations (unchanged shape).

## Error Handling

Fail loud, same standard as every prior node — no `try`/`except` swallowing anything.

- **A missing store is an error, not a silent degradation.** Because LangGraph injects `None` rather than raising when a graph is compiled without a store (verified), each store-using node begins with an explicit `if store is None: raise ValueError(...)`. Without this, a call site that forgot `store=` would produce a graph that silently re-pays for every rejected candidate forever — exactly the class of "silently wrong" failure `CLAUDE.md` forbids.
- Store read/write failures (locked DB, corrupt row, schema drift) propagate uncaught. A broken memory layer must crash the run, not quietly turn the cost control off.
- A stored record that fails `RejectedCandidate` validation on read raises — it is not skipped. That is what `schema_version` exists to make diagnosable.
- Nothing about the store touches `portfolio.db`; the store's only file is `sandbox-data/agent_memory.db`, the same file the checkpointer already owns.

## Testing (Phase 7 gate — must pass before the Synthesizer phase is planned)

Real dependencies wherever they are free; `TestModel` wherever a real call would cost money; `portfolio.db` never mocked.

1. **Store wiring** — `get_memory_store()` yields a working `AsyncSqliteStore` on the real `agent_memory.db` path; `aput`/`aget` round-trips; asserted the store's tables (`store`, `store_migrations`) coexist with the checkpointer's (`checkpoints`, `writes`) in the one file.
2. **Coexistence under real fan-out** — compile the *real* graph with both checkpointer and store against a temp DB file, `ainvoke` it with `TestModel` for the four LLM nodes and real `portfolio.db` data, assert no lock errors and that a checkpoint *and* store rows both exist afterwards. This is the test that would catch a future SQLite-concurrency regression.
3. **Identity/normalisation helpers** — pure-function tests: ticker upper-casing, `.NS` handling, and `_normalize_scheme_name` over the *real* proposed-name variants of an actual held fund (real names only, no invented schemes, per the no-fabrication rule).
4. **Freshness** — a record with `rejected_at` inside the window suppresses; a record just outside it does not. Real `AsyncSqliteStore`, real timestamps, no mocking (the constant is injected, time is not faked).
5. **Cap interaction** — the behavioural test that pins the decision above: given 3 previously-rejected + 5 fresh candidates for one sector, assert the cap is filled with *fresh* candidates and that the rejected ones consumed no slot.
6. **Cross-run suppression, end-to-end and measurable** — run a specialist node twice against the same real store with `TestModel`, counting evaluator invocations: run 1 evaluates and records; run 2 proposes the same candidate and performs **zero** evaluator calls. Counting the calls (not just asserting the output list) is what actually proves money was saved.
7. **Buy is never suppressed** — a `Buy` verdict writes no record and is re-evaluated on the next run.
8. **Fail-loud** — a node invoked with `store=None` raises.
9. **Mutual fund two-stage** — a proposed name that normalises differently but resolves to an already-rejected `scheme_code` is caught by stage (b), before any evaluator call. Uses real `MFapi.in` resolution (free).
10. **`portfolio.db` mtime unchanged** before/after the suite — the same guardrail check every phase has run.
11. **Phase-completion gate (manual, not automated):** two consecutive *real* runs against the real model pair — the first producing real rejections, the second visibly skipping them — reviewed by Abhi before the phase is marked done. Same practice that caught real bugs in Phases 4, 5, and 6. This is the only place real money is spent, and it is bounded by the unchanged 5-per-run cap.

No Playwright needed — still no UI/browser surface.

## Open Decisions (need Abhi's call before planning)

1. **`REJECTION_MEMORY_DAYS = 90`** — proposed, not decided. Shorter (30) means more re-spend but fresher judgments; longer (180) saves more but risks staying negative on a name whose fundamentals have genuinely turned.
2. **Generator exclusion hints** — recommended in, but it is a real (small) token cost and a real prompt change to three nodes, so it is called out rather than folded in silently.
3. **Cache the fund ISIN → `scheme_code` resolution** in a second namespace (`("resolved_schemes", "mutual_fund")`). This is the one genuinely well-defined "cached profile"-shaped thing that exists today: `_resolve_scheme_code` re-runs progressive-query-shortening searches for all 9 held funds on every run. It is free in money, costly in latency and `MFapi.in` politeness. **Recommended in, but it is scope beyond the stated "rejected candidates" goal, so it needs an explicit yes/no.**
4. **Whether rejections should be namespaced per-portfolio-version at all** — this design says no (live gap computation already handles portfolio change). Flagged because it is the one identity assumption that a future multi-portfolio or "what-if" mode would break.

## Decisions (made autonomously, 2026-09-18 overnight run — Abhi explicitly delegated architectural calls for this session; each is reversible and logged here for morning review)

1. **`REJECTION_MEMORY_DAYS = 90`** — accepted as proposed. It's a named constant, not buried, so it's a one-line change to revisit.
2. **Generator exclusion hints** — accepted. The token cost is small and bounded (≤20 rejections), and it reduces wasted evaluator calls on names the generator would otherwise re-propose; the code-level rejection filter remains authoritative regardless of whether the hint lands.
3. **Cache fund ISIN → `scheme_code` resolution** in `("resolved_schemes", "mutual_fund")` — accepted, in scope for this phase. It's free in money and directly serves the same "don't repeat expensive-ish free-API work every run" cost-control spirit as rejection memory, and the design already names the exact namespace and problem (progressive-query-shortening search re-run for all 9 funds every run). Kept as a visibly separate namespace/helper from rejection memory, not folded into the same code path, so it stays easy to back out independently.
4. **Per-portfolio-version namespacing** — declined, per the design's own recommendation. No multi-portfolio/what-if mode exists in this project; adding the scoping now would be speculative complexity with nothing to test it against.

## Risks / known weaknesses

- **The `supports_ttl` MRO quirk** in `langgraph-checkpoint-sqlite 3.1.1` means TTL-based expiry silently does not filter reads on the async store. This design routes around it, but a future version fixing the MRO would start omitting expired items on read — harmless here (our own freshness check is stricter), but worth knowing before a version bump.
- **SQLite concurrency** is verified safe for the current four-parallel-branch shape under WAL, single-process. It is *not* verified for two processes writing concurrently — which is exactly what the Temporal phase could introduce (a scheduled run overlapping a manual one). Flagging now: that phase should either serialise runs or re-verify this.
- **Identity is exact-match by design.** Two share classes of the same fund, or a ticker that changes after a corporate action, read as different candidates. Correct-but-wasteful rather than wrong, and preferable to fuzzy matching that could suppress a genuinely different security.
- **Rejection reasons are not typed.** A candidate rejected on valuation (which can change in weeks) and one rejected on fundamentals (which change slowly) are suppressed for the same 90 days. A `rejection_reason: Literal[...]` field with per-reason windows is the obvious refinement — deliberately not built now, because it requires the evaluator to classify its own rejection, which is a prompt/schema change to three nodes and belongs in its own decision.
- **This phase does not surface memory to the user.** Rejected candidates are stored and used for suppression, but nothing yet *reports* "we skipped X because we rejected it in March." That is naturally the Synthesizer's job, and the record shape above (verbatim `reasoning`, `group`, `rejected_at`, `thread_id`) is designed to make it possible without a migration.
