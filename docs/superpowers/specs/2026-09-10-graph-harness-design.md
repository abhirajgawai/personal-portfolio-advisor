# Graph Harness — Design Spec

**Status:** Approved for planning
**Phase:** 3 of the Portfolio Advisor Fleet LangGraph build (see `plans/plan.md`)

## Goal

Build the minimal harness that turns the two existing standalone nodes (`research_node`, `risk_analyst_node`) into an actual, runnable LangGraph system: a compiled `StateGraph` wiring `Research → Risk Analyst` linearly, backed by a real `AsyncSqliteSaver` checkpointer pointed at a fresh `agent_memory.db`. This is the first phase where the system runs end-to-end as a graph, not as isolated tested functions — and the first phase where anything is written to disk by this project's own code.

## Scope

**In scope:**
- A new `GraphState` TypedDict combining every field the two existing nodes produce.
- `checkpointer.py` — an `AsyncSqliteSaver`-backed checkpointer pointed at `sandbox-data/agent_memory.db`.
- `graph.py` — builds the `StateGraph(GraphState)` with `Research → Risk Analyst` wired linearly.
- A real end-to-end test: invoke the compiled graph (real `portfolio_data_mcp`, real read-only `portfolio.db`), assert the final state has all 7 combined fields correct, and confirm the checkpointer actually persisted a checkpoint for the run.

**Out of scope (explicitly deferred):**
- Any additional graph node (specialists, Synthesizer) — this phase only wires what already exists.
- Parallel branches, conditional edges, human-approval interrupts — the graph is linear for now; these arrive once the specialists/Synthesizer exist and there's something to branch on or pause for.
- LangGraph Store API (long-term/cross-run memory) — only the checkpointer (short-term/thread memory) is in scope this phase, per `plan.md`'s Harness & Memory split. Store arrives once there's something worth remembering across runs (rejected candidates, cached style profile) — nothing produces that yet.
- LangGraph Studio / LangSmith wiring — deferred per the earlier decision to finish all graph nodes first, then wire observability once, per `plan.md`'s original stack sequencing.

## Background / Constraints

- `StateGraph` takes exactly one `state_schema` — not one per node. Live-verified against the installed LangGraph version (`StateGraph.__init__(self, state_schema, ...)`). `ResearchState` and `RiskState` remain as-is (each still documents one node's own partial contribution and is still what that node's tests assert against); `GraphState` is the union both nodes' outputs merge into.
- `SqliteSaver` (the plan's originally-named checkpointer) is explicitly synchronous-only per its own docstring ("does not scale to multiple threads... for async use cases, consider AsyncSqliteSaver"). Since `research_node` is `async def` (it awaits the MCP client) and the graph runs via `ainvoke`, this phase uses `AsyncSqliteSaver` instead — a correction to `plan.md`'s wording, not a deviation from its intent.
- `sandbox-data/` is already gitignored — `agent_memory.db` will never be committed once created.

## Architecture

`checkpointer.py` provides `get_checkpointer()`, an async context manager wrapping `AsyncSqliteSaver.from_conn_string(<path to sandbox-data/agent_memory.db>)`. `graph.py` provides `build_graph()`, which constructs `StateGraph(GraphState)`, adds `research_node` and `risk_analyst_node` as nodes with edges `START → research → risk_analyst → END`, and returns the graph *uncompiled*. Compilation (attaching the checkpointer) happens at the call site, since the checkpointer is an async context manager that must stay open for the duration of a run — `build_graph()` itself has no async lifetime to manage.

## Components

- **`src/agentic_frameworks/graph/state.py`** (extended) — `GraphState(TypedDict)`: `holdings: list[dict]`, `allocation: dict`, `holdings_by_bucket: dict[str, dict[str, list[dict]]]`, `concentration: dict`, `category_overlap: dict`, `currency_exposure: dict`, `liquidity: dict` — the union of every field `ResearchState` and `RiskState` already define.
- **`src/agentic_frameworks/graph/checkpointer.py`** — `get_checkpointer()`, an async context manager yielding a ready `AsyncSqliteSaver` pointed at `sandbox-data/agent_memory.db` (path resolved relative to the repo root, directory created if missing).
- **`src/agentic_frameworks/graph/graph.py`** — `build_graph() -> StateGraph`, wiring `research_node` and `risk_analyst_node` with `START → research → risk_analyst → END`. Uses the string names `"research"` and `"risk_analyst"` as node ids.

## Data Flow

1. Caller opens the checkpointer: `async with get_checkpointer() as checkpointer:`.
2. Caller builds and compiles the graph: `graph = build_graph().compile(checkpointer=checkpointer)`.
3. Caller invokes it with a thread id: `result = await graph.ainvoke({}, config={"configurable": {"thread_id": "<id>"}})`.
4. `research_node` runs first, fetching real holdings via `portfolio_data_mcp`; its `ResearchState`-shaped return merges into `GraphState`. The checkpointer saves a snapshot.
5. `risk_analyst_node` runs next, computing risk views from that same merged state; its `RiskState`-shaped return merges in too. The checkpointer saves another snapshot.
6. `result` is the final `GraphState` — all 7 fields populated.

## Error Handling

Fail loud, same as every node so far. No `try`/`except` anywhere in `checkpointer.py` or `graph.py`. If either node raises (both are already proven to, on bad input), the graph invocation raises uncaught — nothing in the harness swallows a node's failure.

## Testing (Phase 3 gate — must pass before Phase 4 is planned)

Per `CLAUDE.md`'s phased-development rule:

- Real end-to-end invocation test: build, compile, and `ainvoke` the graph for real (real `portfolio_data_mcp` subprocess, real read-only `portfolio.db`), with a known thread id. Assert the final state has all 7 `GraphState` fields present with values matching the same real-data assertions already proven in the Research and Risk Analyst nodes' own tests (e.g. `holdings_by_bucket["real_estate"]["IN"]` has exactly 1 entry; `concentration`/`category_overlap`/`currency_exposure`/`liquidity` are all present and non-empty).
- Checkpointer-persistence test: after the invocation above, query the compiled graph's own state for that thread id (LangGraph's `get_state` API) and assert a real checkpoint exists — not just that the checkpointer was configured, but that it actually wrote something.
- Confirm `portfolio.db`'s mtime is unchanged before/after the test run (same check as Phases 1 and 2) — this phase adds a *new* database (`agent_memory.db`), but must not touch the existing read-only one.

No Playwright needed — no UI/browser surface yet.
