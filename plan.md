# Plan A — Personal Portfolio Advisor (LangGraph Stack)

## Context

You're transitioning into a Senior Software Engineer, AI and Automation role and want real, hands-on capability — not a toy demo, but a genuinely working system built to production standards, deep enough to speak to design tradeoffs in an interview or on the job.

This is **Plan A**: one real system, built once, to real depth, on one core orchestration stack — **LangGraph** (agent graph) + **LangGraph Studio** + **LangSmith** (observability/tracing) + **Pydantic AI** (typed output) + **MCP** (tool protocol) + **Temporal** (durable scheduled execution). A separate **Plan B** (comparative builds in CrewAI, Microsoft Agent Framework, and possibly a second LangGraph use case) is deferred — different use case(s) TBD, not a rebuild of this system, decided later.

The system: a personal portfolio advisor that works like an actual portfolio manager — a fleet of agents that understands your existing holdings (in `E:\agentic-portfolio-dashboard\data\portfolio.db`) *and* proactively researches what's missing, giving both "what to do with what you own" (hold/sell/reduce/add-to) and "what you don't own yet but should consider" (what to buy, why, roughly how much, and how to enter) — long-term, advisory-only guidance, never executed trades. This project already has a read-only MCP server (`portfolio_data_mcp/server.py`, 8 tools: `latest_snapshot`, `holding_returns`, `allocation_summary`, `holding_history`, `portfolio_total`, `recent_runs`, `fired_alerts`, `daily_performance`) exposing your real data — that's a running head start, not something to rebuild.

No fixed time budget — depth over pace.

## The Real Portfolio, in Numbers (drives the agent roster below)

Queried directly from `portfolio.db` (₹10.29M total, current):

| Asset class | Geography | Weight |
|---|---|---|
| Real estate (1 property, Taloja) | IN | 34% |
| Mutual funds (9 funds) | IN | 26% |
| Direct equity (36 stocks) | IN | 18% |
| ETFs (SMH, VOO, IAU) | US | 15% |
| Direct equity (GOOGL, IBM, +1) | US | 6% |
| ETF | IN | 0.7% |

This is **not** a single-market, single-asset-type stock-picking portfolio, and the roster below is shaped by that:
- **Real estate (34%) is illiquid and has no live price/PnL feed** — it's a fixed input agents reason *around*, never an active hold/sell/buy subject.
- **Mutual funds (26%) need category/expense-ratio/manager-consistency analysis, not price-action analysis** — evaluated differently than a single stock.
- **Two currencies, two geographies (IN/US)** — allocation and FX-exposure reasoning is a first-class concern, not an afterthought.
- **Direct equity is a minority of the portfolio (24% combined)** — so agent effort is weighted toward allocation-level reasoning (are you overweight real estate, underweight international) as much as single-security analysis.

## Scope Guardrails

- **Advisory only, never execution.** The graph never places an order, calls a broker API to trade, or auto-executes anything. Every output is a recommendation for you to act on manually.
- **Long-term investing only.** No day-trading, no options/derivatives, no intraday timing signals. "When to buy" means entry approach (lump sum vs. staggered/SIP-style, valuation-aware timing) — never short-term market timing.
- **Two-sided advice.** Covers both (a) what to do with *existing* holdings (hold/sell/reduce/add-to), and (b) what's *missing* — new positions worth considering, with why, roughly how much, and how to enter.
- **"How to buy" stays advisory.** Practical guidance only — which of your existing platforms (zerodha/moomoo/groww/sharekhan) suits the instrument, position sizing relative to total portfolio, staggered vs. lump-sum entry. Never a trade instruction.
- **Avoid is the default, not buy.** Given current market conditions, every new-buy candidate must clear a conviction gate before it's ever surfaced — "no good candidate this cycle" is a valid, expected, and common result. A suggestion is the exception, not the norm.

## New-Buy Conviction Gate (applies to every specialist below)

Before any specialist (US, India, or Fund) surfaces a candidate, it must pass all four checks — reject early, don't research everything:

1. **Portfolio-fit check** — does it address a real, identified gap in your actual current holdings (e.g., "near-zero healthcare exposure beyond CAPLIPOINT")? No identified gap → rejected before research starts.
2. **Fundamentals check** — real financials via the research MCP tool (revenue/earnings trend, debt, margins) — not model priors.
3. **Valuation check** — reasonably priced now vs. its own history and peers. Richly-valued candidates are flagged or rejected, not rationalized, given current market conditions.
4. **Style-fit check** — consistent with the cap-size/sector tilt your actual holdings show (derived from a real query over `instrument`/`position`), or an explicit, justified deviation.

## Data Architecture — Two Databases, Not a Copy

Copying `portfolio.db` once would go stale the moment your real pipeline updates. Instead:

1. **Reads** go straight to your **real** `portfolio.db`, opened strictly read-only. The existing `portfolio_data_mcp` server already connects via a `mode=ro` SQLite URI (`portfolio_data_mcp/server.py:52-63`) — a filesystem-level lock, not just an application convention. The graph reuses this server (or a same-pattern sibling) as-is, unmodified. Data is always current — no sync problem, because there's nothing to sync.
2. **Writes** (agent decision logs, reasoning traces, recommendation history, checkpoint/thread state) go to a **separate, fresh** `E:\agentic-frameworks\sandbox-data\agent_memory.db`, created empty before build starts. Never in `portfolio.db`.
3. Any *new* read capability (e.g., the research/valuation/fund-factsheet tools) is a **new, separate, read-only MCP server** — the existing `portfolio_data_mcp/server.py` is never edited.

This means: real data, always fresh, zero risk to the live dashboard/pipeline, and a clean separation between "what agents read" and "what agents write."

## Model Access

OpenRouter/OpenCode API keys — model-agnostic, no vendor lock-in. LangSmith tracing is used purely for observability (which node ran, state at each step) — it doesn't require using LangChain's own model wrappers exclusively; any model callable through OpenRouter can be used inside the graph.

## Harness & Memory

No separate agent harness — LangGraph's own runtime executes the graph. Memory uses two LangGraph-native mechanisms, both backed by the existing `agent_memory.db` (no third-party memory platform like Hindsight/Flue — those solve a real problem but duplicate what's already covered here, and adding one would cut against keeping Plan A minimal):

1. **Short-term / thread memory — `SqliteSaver` checkpointer**, pointed at `agent_memory.db`. Gives automatic state snapshots per node, crash/interrupt recovery, and is what makes the human-approval pause/resume actually work.
2. **Long-term / cross-run memory — LangGraph Store API**, same database. Remembers things across separate runs: previously rejected new-buy candidates (don't re-propose without new information), a cached style profile (sector/cap-size tilt derived from `instrument`/`position`) so it isn't recomputed every run.

## The Stack

**1. LangGraph — Core Advisor Graph**
Research (pull holdings context via MCP, split by asset type — direct equity / mutual funds / real estate) → Risk Analyst → **three parallel new-buy specialists** — US Equity/ETF Specialist, India Equity Specialist, Mutual Fund Specialist — each applying the New-Buy Conviction Gate before proposing anything → Recommendation Synthesizer, with a conditional edge that routes any "sell" *and* any new-buy suggestion through a human-approval gate before finalizing.
*Teaches*: explicit state graphs, conditional branching, parallel branches that fan back in, cycles (an analyst can loop back for more data), human-in-the-loop interrupts, checkpointing/resumability.

**2. LangGraph Studio + LangSmith — Observability**
Run the graph through Studio to visualize nodes/edges, step through execution, inspect state at each step, and replay/resume from a checkpoint. LangSmith traces every run: node order, input/output, latency, token cost, full state diffs — this is how you answer "why did the Risk Analyst reject this" after the fact, and watch a run live.
*Teaches*: what "observability" means for an agent system in practice — not logs, but structured, replayable execution traces.

**3. Pydantic AI — Type-Safe Recommendation Output**
The Synthesizer's final output (and each specialist's candidate proposal) is a strictly validated structured object: `{action: Buy|Hold|Sell|Reduce, confidence: float, reasoning: str, risk_flags: list[str], entry_approach: LumpSum|Staggered|None, suggested_platform: str|None}` — no freeform text leaks through unvalidated. `action=Buy` is always advisory metadata, never a triggered order.
*Teaches*: schema-first agent output, validation-before-use, why "just parse the LLM's JSON" is fragile compared to a typed contract.

**4. MCP — Tool Layer**
A new, separate, read-only MCP server (existing `portfolio_data_mcp/server.py` untouched) adding: news/filing/fundamentals lookup (US and India), a valuation/multiples tool (for the Conviction Gate's valuation check), and mutual fund category/factsheet lookup for the Fund Specialist. Built from scratch so you understand the protocol itself — tool schemas, request/response shape.
*Teaches*: MCP as the vendor-neutral tool-calling standard; a tool built once, callable from the graph or any MCP-aware client.

**5. Temporal — Durable Daily Monitor**
A Temporal workflow that runs daily, checks alert rules / daily performance data via the existing read-only MCP server, invokes the LangGraph advisor (or a scoped subgraph) to interpret anomalies, and notifies you — built so a crash or restart mid-run resumes cleanly rather than duplicating work or losing state.
*Teaches*: durable execution, automatic retries, long-running/scheduled workflows — the production-reliability half of "AI and Automation" that agent-reasoning frameworks alone don't address.

## Cross-Cutting Learning Checkpoint (per stack component)

For each of the five pieces above, explicitly answer before moving on:
1. **What problem does this piece solve that a plain LLM API call / plain cron job doesn't?**
2. **What did it make easy that was hard/verbose without it?**
3. **What did it make hard or awkward?**
4. **When would you reach for this in a real job?**

## Verification

- The graph runs end-to-end against your real `portfolio.db` (read-only) and produces output referencing your actual holdings (not fabricated/placeholder tickers) — plus either a genuinely researched new-buy candidate that clears the Conviction Gate, or an explicit "no good candidate this cycle" with the reasons it was rejected.
- The graph never calls a broker/execution API or writes to `portfolio.db`. Confirmed by inspecting the DB connection string (must be the `mode=ro` URI pattern) and by there being no order-placement tool registered anywhere.
- All scratch writes (decision logs, memory, checkpoints) land only in `agent_memory.db` — never `portfolio.db`.
- Every run is inspectable end-to-end in LangGraph Studio / LangSmith — you can point to which node made which call and why.
- The Temporal workflow survives a simulated crash/restart mid-run without duplicating work.
- Checkpoint questions above are answered in short notes files (e.g., `notes/01-langgraph.md`, `notes/02-temporal.md`) per stack component.

## Open Items (not blocking the plan)

- Exact news/fundamentals/valuation data source for the US, India, and Fund specialists (which API/provider per market) — decide when we reach the MCP build.
- Notification channel for the Temporal build (email/Slack/console) — decide when we reach that build.
- Position-sizing heuristic for new-buy suggestions (e.g., cap at X% of portfolio, or relative to existing sector weight) — decide when we reach the LangGraph build, since it's the first to need it.
- Which sectors/themes are genuinely missing from the 36 IN equities (concrete gap list) — worth a quick data pass before the LangGraph build so the specialists' portfolio-fit check has real gaps to check against, not guesses.

## Immediate Next Step: Claude Setup (this session, before any agent code)

Scope is setup only — no portfolio-advisor code yet:

1. **`E:\agentic-frameworks\CLAUDE.md`** — durable project instructions so future sessions don't need this plan re-explained. Captures: the real portfolio data profile, all Scope Guardrails, the New-Buy Conviction Gate, the two-database architecture (`portfolio.db` read-only / `agent_memory.db` writes), the Harness & Memory decisions (LangGraph checkpointer + Store, no third-party memory platform), and the Stack list — written as instructions Claude must follow, not prose to re-read.
2. **A skill capturing the New-Buy Conviction Gate** as a reusable checklist (portfolio-fit → fundamentals → valuation → style-fit, reject-early) — since it applies identically across all three specialists and should be invoked consistently rather than re-derived per agent.
3. Directory scaffolding only where needed for the above (e.g., confirming `E:\agentic-frameworks\sandbox-data\` exists for the future `agent_memory.db`) — no framework code, no agent implementation.

## Deployment

Local-only for Plan A (P1). The real DB is local and read-only by design — deploying to the Hostinger VPS now would mean either exposing that DB over the network or running against stale/synced data, both of which conflict with the "always current, never copied" architecture. Build and verify everything locally (LangGraph Studio + LangSmith for correctness). Revisit VPS deployment in P2, once the graph and the Temporal daily monitor are working and verified — Temporal in particular only earns its keep once it needs to run unattended.

## Plan B (deferred, separate document)

CrewAI, Microsoft Agent Framework, and possibly a second LangGraph use case — comparative hands-on builds on a **different use case** (not a rebuild of the Plan A portfolio advisor). Use case(s) and scope TBD later, once Plan A is done.
