# Plan B (sketch) — Microsoft Agent Framework

## Context

Plan A (`plan.md`) is the active build: the real Portfolio Advisor Fleet, LangGraph stack, no other framework mixed in. Plan B is a **comparative learning build** — a second, smaller system in **Microsoft Agent Framework** to get hands-on with a different orchestration model, so Abhi can speak to the tradeoffs between them (not just LangGraph in isolation).

This is a sketch, not a build-ready plan. **No code, scaffolding, or dependencies for Plan B are created until Plan A is fully built and verified.** Writing this down now just captures the reasoning while it's fresh.

Non-goals:
- Not a rebuild of the Plan A advisor graph.
- Not started until Plan A ships (graph + Studio/LangSmith + Pydantic AI schemas + MCP server + Temporal monitor all working and verified).
- No timeline beyond "after Plan A."

## Framework: Microsoft Agent Framework

Microsoft's newer unified agent SDK — effectively the convergence point of **AutoGen** (multi-agent conversation patterns) and **Semantic Kernel** (plugin/skill-based orchestration, enterprise integration). Where LangGraph models an agent system as an explicit state graph you wire node-by-node, Microsoft Agent Framework leans more on declarative agent/plugin composition and conversational multi-agent patterns. That contrast — explicit graph vs. declarative multi-agent composition — is the actual learning target of Plan B, not just "a different vendor's syntax."

## Candidate use cases (same portfolio domain, different angle — none picked yet)

All reuse the existing real `portfolio.db`, read-only, same as Plan A — none are a second copy of the Plan A advisor graph:

- **Tax-loss harvesting checker** — scans real holdings for unrealized-loss positions that could offset gains; surfaces IN/US tax-rule caveats. Different workflow shape from Plan A (a scan-and-flag checker, not a multi-specialist advisory graph). *(Covers the "Tax optimization" gap below.)*
- **Daily market-news digest agent** — pulls news/filings relevant to actual holdings, summarizes what's material vs. noise. Simplest of the candidates — good for learning Microsoft Agent Framework's basics without much orchestration complexity. *(Covers part of the "Communication" gap below.)*
- **Rebalancing drift monitor** — tracks allocation drift vs. target bands (asset class/geography) from `portfolio.db`, explains what's pulling the portfolio out of shape. Complements Plan A's specialists (which handle new-buy conviction) without duplicating them. *(Covers the "Rebalancing to explicit targets" gap below.)*

Pick deferred to when Plan B actually starts.

## Coverage gaps carried over from Plan A (financial-advisor responsibilities)

A full financial-advisor/planner role check against Plan A surfaced several responsibilities Plan A doesn't (and, in one case, structurally can't) cover. Per Abhi's direction: Plan A stays unchanged as-is; everything it doesn't cover is acknowledged and parked here for Plan B to pick up from, rather than silently dropped.

- **Goal & risk-profile capture** — Plan A's Risk Analyst infers risk posture from actual holdings only; it never captures stated time horizon, risk tolerance, or liquidity needs. Plan B should have some way to take this as an input (even a simple static profile) rather than inferring blind.
- **Tax optimization** — tax-loss harvesting, tax-efficient placement, timing of gains/losses. Already the Tax-loss harvesting checker candidate above.
- **Performance reporting (returns vs. benchmark)** — Plan A has the raw data (`portfolio_total`, `holding_returns`, `daily_performance` — existing `portfolio_data_mcp` tools) and LangSmith reports on *agent execution* (cost/latency), but nothing turns portfolio data into a benchmark-vs-return report for Abhi. A candidate for Plan B, not yet on the shortlist above.
- **Rebalancing to explicit targets** — Plan A reasons about allocation qualitatively; it never sets or tracks drift against explicit target-weight bands. Already the Rebalancing drift monitor candidate above.
- **Risk management — downside protection** — Plan A covers diversification reasoning but not hedging/downside-protection strategies. **Scoped deliberately to non-derivative methods only** (e.g., correlation/concentration-based downside analysis) — options/derivatives-based hedging conflicts with the "long-term investing only" guardrail below and is excluded, not silently included.
- **Financial planning (retirement/estate/insurance/cash flow)** — not addressed by Plan A at all. Genuinely new scope for Plan B to size up when it starts — likely the largest of these gaps.
- **Communication (conversational Q&A during volatility)** — Plan A produces structured, readable output per run but isn't a chat-style agent you can interrogate ad hoc. Partly addressed by the news-digest candidate above; a fuller conversational interface is still open.

**Explicitly NOT carried forward — permanent exclusion, not a gap to fill:** trade execution. Both Plan A and Plan B are advisory-only; no order-placement capability belongs in either, ever. Investment selection & execution's *selection* half is covered (Plan A's Conviction Gate); the *execution* half stays out by design.

## Guardrails (carried over from Plan A, restated so Plan B doesn't silently inherit or collide)

- Advisory only — never execution, never a broker/order-placement call.
- `portfolio.db` stays read-only, same `mode=ro` URI pattern as Plan A.
- Any Plan B writes (scratch state, decision logs) go to a **new, separate** database — never `agent_memory.db`, which belongs to Plan A only.
- Long-term investing framing only — no day-trading/timing logic, consistent with Plan A's scope guardrails.

## Open items (deferred to Plan B start)

- Which of the three candidate use cases to build.
- Exact data needs beyond what `portfolio_data_mcp` already exposes — reuse that server as-is, or does the chosen use case need a new read-only MCP server (e.g., tax-rule lookup, news source)?
- Whether Plan B gets its own observability/tracing approach or skips it (Microsoft Agent Framework's native tooling vs. nothing, since LangSmith is LangGraph-specific).
- CrewAI and "a second LangGraph use case" remain noted Plan B alternatives per `plan.md:130` but are not pursued — Microsoft Agent Framework is the committed Plan B framework choice.
