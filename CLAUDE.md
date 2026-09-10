# Portfolio Advisor Fleet — Plan A (LangGraph Stack)

Full plan: `plan.md` in this repo (mirrors `C:\Users\Abhiraj Gautam Gavai\.claude\plans\where-it-lives-concurrent-globe.md`).

## Working mode: explain-as-you-go, no vibe coding

**HARD RULE — no blind execution, ever.** Before any implementation step runs, Abhi must know what is about to be done, why this approach (over alternatives), and how it works. This is not a style preference to weigh against convenience — it overrides the instinct to just get on with the task. If an instruction elsewhere in this file or a request in the moment could be read as "go implement X," that request still passes through this rule first: explain, then build, one piece at a time.

This is a learning project first, a working system second. The user (Abhi) wants to understand every step, not just receive finished code. This is a standing rule, not a one-time reminder:

- **Before writing or changing code**, state in plain language: what you're about to do, why this approach (vs. alternatives), and what happens if it goes wrong or is skipped. A sentence or two is enough — don't pad it into an essay, but don't skip it either.
- **After a non-trivial step**, briefly say what changed and what it means in practice — not a code walkthrough, a "here's what this buys us."
- **When introducing a new concept from the stack** (a LangGraph primitive, a Pydantic AI pattern, an MCP mechanic, a Temporal concept), explain the concept itself in plain terms before or alongside using it — assume Abhi wants to be able to explain it to someone else afterward, not just see it work.
- **Never silently make an architectural choice.** If there's a real decision point (which checkpointer backend, how a node routes, how a schema is shaped), surface it and the reasoning — don't just pick one and move on without saying so.
- This doesn't mean asking permission for every line of code — routine implementation (writing a function once the approach is agreed) doesn't need a play-by-play. It means the *why* and *what it does* should always be visible, never left implicit in a diff Abhi is expected to just trust.
- **One piece at a time, confirmed before moving on.** Build in small increments (one node, one schema, one tool — not "the whole graph" in one pass). After each increment, stop, show what was built and why, and get an explicit go-ahead before starting the next piece. Do not chain multiple unrelated build steps together without a checkpoint in between.
- **No silent scope growth.** If a step turns out to need more than what was just agreed (an extra file, a new dependency, a design change), say so and get confirmation before doing it — don't fold it in quietly because it seemed convenient.

## Phased development — testing gate between phases

- Build happens in phases (e.g., one LangGraph node, one MCP server, one Temporal workflow — matches the "one piece at a time" rule above).
- Each phase ends with real testing before it's considered done — test cases (pytest against real data, no mocks of `portfolio.db`) and, for anything with a UI/browser surface, Playwright.
- Planning the next phase does not start until the current phase's tests pass. No "we'll test it later" or parallel phase-planning-while-untested.
- Once a phase's tests pass and its branch is merged to `main`, push `main` to GitHub before starting the next phase's brainstorming. Commits and pushes happen per phase, not batched across multiple phases and not held back for a later "big push."

## Agentic workflow design principles

Standing design principles for every node/agent in this graph, established 2026-09-10 after building the Research and Risk Analyst nodes. Apply going forward; existing code is already compliant (see audit).

1. **Workflows before agents.** Default to a fixed, deterministic pipeline (a plain function per node). Reach for genuine LLM-driven autonomy only where a fixed pipeline can't do the job (specialist judgment) — not for data-fetching or measurement.
2. **Small, single-purpose, independently testable units.** One node = one job. Split internal logic into small private helpers when a node does more than one distinct computation.
3. **Explicit, typed state between nodes.** Every node's input/output is a `TypedDict` in `state.py` — never an opaque blob or free-form dict passed by convention.
4. **Separate "gather/measure" from "judge/decide."** Data-fetching/measurement nodes report raw numbers only — no verdicts, no thresholds. Judgment concentrates in later nodes (Conviction Gate, Synthesizer), so that's the only place needing scrutiny for *opinion*, not fact.
5. **Fail loud, never fabricate.** No `try`/`except` swallowing errors. No placeholder/fabricated data anywhere, including test fixtures (real tickers/sectors/values only). A crashed node beats a silently-wrong one.
6. **Tools as external, reusable protocols (MCP) — never ad-hoc logic baked into a node.** A node calls a tool; it never embeds the data-fetching logic itself.
7. **Guardrails are structural, not instructional.** Enforce constraints by never building/registering the capability that would violate them — not by prompting an LLM to avoid it. This is why the "advisory only" guardrail above means an MCP server bundling execution capability is ruled out entirely, never "used carefully."
8. **Bounded autonomy via reject-early gates.** Expensive work only runs on candidates that already cleared a cheap check first (Conviction Gate's portfolio-fit check gates fundamentals/valuation).
9. **Human-in-the-loop at consequential decision points.** Any sell or new-buy suggestion routes through human approval before finalizing.
10. **Memory is a deliberate architectural choice.** Short-term/thread memory and long-term/cross-run memory (see Harness & Memory below) are different concerns — never conflated into one undifferentiated "memory" blob.
11. **Observability is not optional tooling.** Every node/run must be inspectable after the fact (LangGraph Studio / LangSmith) — not just trusted to have worked.

(Incremental, real-data, one-piece-at-a-time building is already covered by "Phased development" above — that rule exists *because of* this list, not separately from it.)

## What this project is

A personal, long-term portfolio advisor built as a LangGraph agent graph, working off a **real** SQLite portfolio database (`E:\agentic-portfolio-dashboard\data\portfolio.db`, ₹10.29M, mixed IN/US, mixed equity/fund/real-estate). Advisory only — it never places trades. This is also a hands-on learning project for LangGraph, LangGraph Studio, LangSmith, Pydantic AI, MCP, and Temporal — depth over breadth, one real system, not toy demos.

## Non-negotiable guardrails

- **Advisory only, never execution.** No code here ever places an order or calls a broker API. Every output is a recommendation for the user to act on manually.
- **Long-term investing only.** No day-trading, options/derivatives, or intraday timing logic. "When to buy" means entry approach (lump sum vs. staggered), never market-timing.
- **Two-sided advice.** Every build must be able to address both (a) existing holdings — hold/sell/reduce/add-to — and (b) new-buy candidates — why, how much, how to enter.
- **Avoid is the default, not buy.** New-buy candidates must clear the Conviction Gate (see skill below) before being surfaced. "No good candidate this cycle" is a valid, expected, common result.
- **`portfolio.db` is read-only, always.** Every connection to it MUST use the `mode=ro` SQLite URI pattern (see `E:\agentic-portfolio-dashboard\portfolio_data_mcp\server.py:52-63` for the reference implementation). Never write to it, never copy it. Any new read capability is a new, separate MCP server — `portfolio_data_mcp/server.py` is never edited.
- **Agent writes go to `agent_memory.db` only.** Location: `E:\agentic-frameworks\sandbox-data\agent_memory.db` (created fresh, empty). Decision logs, LangGraph checkpointer state, and LangGraph Store long-term memory all live here — never in `portfolio.db`.

## Data reality (drives every agent's design)

Real portfolio composition — not a single-market, single-asset-type portfolio:

| Asset class | Geography | Weight |
|---|---|---|
| Real estate (1 property) | IN | 34% |
| Mutual funds (9 funds) | IN | 26% |
| Direct equity (36 stocks) | IN | 18% |
| ETFs (SMH, VOO, IAU) | US | 15% |
| Direct equity (GOOGL, IBM, +1) | US | 6% |
| ETF | IN | 0.7% |

- Real estate: illiquid, no live price/PnL — a fixed input, never an active hold/sell/buy subject.
- Mutual funds: evaluated by category/expense-ratio/manager-consistency, not price action.
- Two currencies, two geographies (IN/US): allocation and FX-exposure reasoning is first-class.
- Direct equity is a minority (24% combined) of the portfolio — weight agent effort toward allocation-level reasoning, not just single-security picking.

## Harness & memory

- No separate agent harness — LangGraph's own runtime executes the graph.
- Short-term/thread memory: LangGraph `AsyncSqliteSaver` checkpointer, pointed at `agent_memory.db` (async, not the sync `SqliteSaver` — the graph's nodes are async and run via `ainvoke`; corrected 2026-09-10 during the Graph Harness phase).
- Long-term/cross-run memory: LangGraph Store API, same database — rejected candidates, cached style profile.
- No third-party memory platform (Hindsight/Flue etc.) — LangGraph's native mechanisms already cover this project's memory needs. Don't introduce one without an explicit decision to revisit this.

## Stack (Plan A — build once, to real depth)

1. **LangGraph** — core advisor graph: Research → Risk Analyst → three parallel new-buy specialists (US Equity/ETF, India Equity, Mutual Fund) → Recommendation Synthesizer, with a human-approval gate on any sell or new-buy suggestion.
2. **LangGraph Studio + LangSmith** — observability: visualize the graph, step through execution, trace every run.
3. **Pydantic AI** — the Synthesizer's (and each specialist's) output is a strictly validated schema, never freeform text.
4. **MCP** — a new, separate, read-only MCP server for news/fundamentals/valuation/fund-factsheet lookups, feeding the Conviction Gate.
5. **Temporal** — a durable daily monitor that invokes the graph on a schedule, crash/restart-safe.

CrewAI and Microsoft Agent Framework are **Plan B**, deferred, and will use a different use case — not a rebuild of this system. Don't pull them into Plan A work. Plan B sketch (framework choice, candidate use cases, guardrails): `plan-b.md` in this repo.

## Deployment

Local-only for now (P1). Do not deploy to the Hostinger VPS yet — revisit once the graph and Temporal monitor are working and verified locally (P2).

## Verification checklist (every build)

- Runs end-to-end against the real `portfolio.db` (read-only), referencing actual holdings — no fabricated tickers.
- Never calls a broker/execution API; no order-placement tool exists anywhere in the fleet.
- Scratch writes land only in `agent_memory.db`.
- Every run inspectable in LangGraph Studio / LangSmith.
- New-buy suggestions either clear the full Conviction Gate or are explicitly rejected with reasons — never skip the gate.
