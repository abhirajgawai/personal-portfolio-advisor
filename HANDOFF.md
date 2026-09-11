# Session Handoff — Portfolio Advisor Fleet

**Last updated:** 2026-09-11, end of the session that built Phases 1-4.

**Read this first in a new session**, then `CLAUDE.md` (project rules) and `plans/plan.md` (the full architecture plan with its decision log). This doc is the "what's been done, what's next" summary — the other two are the durable rulebook and design record.

## How this project works (read before doing anything)

This project is built entirely through the **`superpowers`** Claude Code plugin's skill pipeline. Every phase follows the same sequence:

1. **`superpowers:brainstorming`** — classify the work as spike/bounded/architectural, ask clarifying questions one at a time, propose approaches, present a design section-by-section, get explicit approval. For architectural work, this ends with a written spec at `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`, committed to git.
2. **`superpowers:writing-plans`** — turns the approved spec into a granular, TDD-structured implementation plan at `docs/superpowers/plans/YYYY-MM-DD-<topic>.md` — bite-sized tasks, each with exact code, exact test assertions, no placeholders.
3. **`superpowers:subagent-driven-development`** — executes the plan: a fresh subagent per task, an independent task reviewer after each one (spec compliance + code quality), a ledger at `.superpowers/sdd/<plan-basename>/progress.md` tracking everything, a capped fix-loop when reviewers find real issues, and a final whole-branch review (on the most capable model) before merge.
4. **`superpowers:finishing-a-development-branch`** — merge/cleanup (the user has been doing actual merges themselves via GitHub PRs).

**Non-negotiable standing rules, all in `CLAUDE.md`:**
- **Explain-as-you-go**: before any implementation step, explain what/why/how; after, explain what changed and what it means. One piece at a time, confirmed before moving on. No silent scope growth.
- **Phased development**: each phase ends with real testing (pytest against real data, no mocks of `portfolio.db`) before the next phase is even planned. Once a phase's tests pass and it's merged, push `main` before starting the next phase's brainstorming.
- **Agentic workflow design principles** (11 of them): workflows before agents, small single-purpose units, explicit typed state (`TypedDict`s in `state.py`), separate gather/measure from judge/decide, fail loud never fabricate, tools as external MCP protocols not ad-hoc logic, guardrails as structure not instruction, reject-early gating, human-in-the-loop at consequential decisions, memory as a deliberate choice, observability is not optional.
- **LLM cost control** (7 rules): explicit per-node model choice (never silent default), reject-early before expensive calls, cacheable stable system prompts, minimal context per call, hard numeric caps enforced in code, bounded retries, long-term memory as a cost control.
- **Non-negotiable guardrails**: advisory only, never execution (no order-placement tool anywhere, ever); long-term investing only (no options/derivatives/day-trading); `portfolio.db` is read-only always (`mode=ro` URI, never edit `portfolio_data_mcp/server.py`); agent writes go only to `sandbox-data/agent_memory.db`; new-buy candidates must clear the full Conviction Gate or be explicitly rejected — "avoid is the default, not buy."

**Verification habit used throughout this session, worth continuing:** never trust a library's documented API shape — verify it live (`python -c "..."` against the actually-installed version) before writing plan code. This caught several real surprises (MCP tool result shapes, `yfinance-mcp`'s `symbol` not `ticker` argument, a webpage-only model alias that wasn't a valid API ID, `FilteredToolset` not exposing `direct_call_tool`). Reviewers in this session also independently re-verified claims rather than trusting reports — re-running tests themselves, reading library source to confirm a monkeypatch pattern, hand-tracing cap logic against concrete scenarios. Keep that standard.

## What's actually done (merged to `main`, tested, working)

**Phase 1 — Research node.** `src/agentic_frameworks/graph/nodes/research.py`. Fetches real holdings via the existing `portfolio_data_mcp` server (a separate project at `E:\agentic-portfolio-dashboard`, its own venv, launched as a stdio subprocess). Groups holdings by asset_class × geography. Real portfolio: ₹10.29M, mixed IN/US, mixed equity/fund/real-estate.

**Phase 2 — Risk Analyst node.** `src/agentic_frameworks/graph/nodes/risk_analyst.py`. Pure functions (no LLM, no MCP) computing four risk views from Research's output: concentration, category overlap (sector-based diversification proxy), currency exposure, liquidity (real-estate vs everything else). Reports raw numbers only, no verdicts — judgment is deferred to later nodes.

**Phase 3 — Graph Harness.** `src/agentic_frameworks/graph/graph.py`, `checkpointer.py`. First time the system ran as an actual `StateGraph`, not isolated functions. Linear `Research → Risk Analyst`, backed by a real `AsyncSqliteSaver` checkpointer writing to `sandbox-data/agent_memory.db` (gitignored, created fresh). Proved memory persistence works via a real end-to-end run.

**Phase 4 — US Equity/ETF Specialist.** `src/agentic_frameworks/graph/nodes/us_equity_etf_specialist.py`. First LLM-calling node, first `pydantic-ai` usage, first real spend of money on a paid API. Three-stage pipeline: (1) an LLM candidate generator proposes tickers per gap sector, capped in code at 3/sector and 5/run (never trusting the model's own count); (2) each candidate is verified against real `yfinance-mcp` data before any expensive evaluation runs (reject-early — this is also the safeguard against LLM-hallucinated tickers, since a fake ticker returns a near-empty payload rather than raising, and `longName` presence is the confirmed reliable signal); (3) surviving candidates go through a Conviction Gate evaluator (portfolio-fit → fundamentals → valuation → style-fit, reject-early on first failure), producing a strictly validated `CandidateEvaluation` per candidate.

- **Model:** `deepseek/deepseek-v4.1-flash` primary via OpenRouter, `z-ai/glm-5.3-flash` backup (added mid-phase after DeepSeek's shared free pool returned genuine repeated 429s — confirmed via `pydantic_ai.models.fallback.FallbackModel`, which triggers on `ModelAPIError`, live-confirmed a 429 raises as a subclass of that).
- **Data source:** `yfinance-mcp` (external package, launched via `uvx yfmcp@latest`, not a project dependency — same "sibling subprocess" pattern as `portfolio_data_mcp`). Covers US equity, US ETF fundamentals/holdings, AND India equity (via `.NS` ticker suffix, live-verified against real NSE stocks). Options/derivatives/intraday-gapper tools are structurally excluded from the toolset via `.filtered()` — not just told not to use them, actually can't reach them.
- **Required real-model gate:** every LLM-calling phase must include one real, deliberate, cost-bounded run against the actual paid model, reviewed by the user, before the phase counts as done — this is what caught a genuine bug (`confidence` field had no bounds, model returned `7.0` once) that no amount of `TestModel`-only testing could have found. `TestModel` (free, `pydantic_ai.models.test`) is used for all automated tests of LLM-calling logic; the real check is separate and manual.
- **Post-merge fix (2026-09-11, bounded, no branch):** the portfolio-fit check was a tautology (told a sector is a gap, asked to confirm it, no real data). Fixed: the node now builds a real holdings summary from `state["holdings_by_bucket"]` at call time (not hardcoded), baked into the Conviction Gate's system prompt. Verified end-to-end against the real graph output — genuinely reflects your actual 6 US holdings (GOOGL, IBM, MU, SMH, VOO, IAU), and updates automatically as the portfolio changes.

**Supporting infrastructure:**
- `.env` (gitignored, user-populated) + `.env.example` (committed template) for `OPENROUTER_API_KEY`, auto-loaded via `python-dotenv`.
- `docs/portfolio-gap-analysis.md` — a real, queried-live gap analysis of US/India equity sector coverage, used by the specialist's candidate generator.
- `.claude/skills/conviction-gate/SKILL.md` — the 4-check reject-early gate as a reusable skill doc.
- 32 automated tests, all passing, all using real dependencies where free (portfolio.db via portfolio_data_mcp, yfinance-mcp) and `TestModel` where a real call would cost money.

## What's explicitly NOT done yet (disclosed, not silent gaps)

1. **India Equity Specialist** — not started. Should reuse most of Phase 4's pattern (`yfinance-mcp` already confirmed working for `.NS` tickers).
2. **Mutual Fund Specialist** — not started. Needs a new MCP client for `MFapi.in` (free, no key, India mutual fund NAV/scheme data — not yet integrated anywhere in code, only researched/decided).
3. **Recommendation Synthesizer** — not started. Cross-specialist reasoning, `pydantic-ai` structured output, and the human-approval gate on any sell/new-buy suggestion — none of this exists yet.
4. **Full graph wiring** — only `Research → Risk Analyst` is actually wired into `build_graph()`. The 3 specialists (once all exist) need to fan out in parallel and feed into the Synthesizer, with a conditional edge routing sell/buy suggestions through human approval. Not started.
5. **LangGraph Store** (cross-run memory — rejected candidates, cached style profile) — deferred three phases running. Real cost implication: right now, nothing stops the US specialist from re-researching and re-paying for a candidate it already rejected in a prior run. **User has asked for this to be its own phase, done after the next specialist, before further specialists.**
6. **LangGraph Studio + LangSmith** (observability) — 0%, not started. Currently zero ability to watch a run happen or replay one after the fact; only pytest output exists.
7. **Temporal** (durable daily monitor) — 0%, entirely separate stack piece, not touched.
8. **Deployment** — explicitly deferred to P2 in `plan.md`, not blocking.
9. Per-stack-component learning-checkpoint notes files (`notes/01-langgraph.md` etc., per `plan.md`'s Cross-Cutting Learning Checkpoint) — not started.

Rough overall completion estimate discussed with the user: **~30-35% of the full Plan A scope**, with the harder/riskier foundational infrastructure (data access, cost-controlled LLM calling, the harness, guardrail patterns) now proven and reusable — but the Synthesizer, full graph wiring, Store, Studio/LangSmith, and Temporal are all still fully ahead.

## Agreed order for what's next (user's explicit instruction, end of this session)

1. ~~Fix the portfolio-fit tautology~~ — **done**, see above.
2. **Next specialist** (India Equity or Mutual Fund — not yet chosen, ask the user which) — this is the next thing to brainstorm.
3. **LangGraph Store**, as its own dedicated phase, after the next specialist.

## Known secrets exposure this session (already mitigated, just for the record)

Three separate incidents where the user pasted a raw API key directly into chat instead of running it as an env-var shell command (Finnhub key, OpenRouter key twice). None were destructive (data-read-only keys), but the user rotated the OpenRouter key mid-session via a fresh `.env` file (correctly gitignored, confirmed). If a fresh session sees old key-looking strings in scrollback, they're already rotated — no action needed, just don't reuse them.

## File/directory map for a fresh session to orient with

- `CLAUDE.md` — the standing rulebook (read this always).
- `plans/plan.md` — the full Plan A architecture, including a running decision log at the MCP/data-source section (what was researched, chosen, and ruled out, with dates).
- `docs/superpowers/specs/` — one design spec per phase, chronological.
- `docs/superpowers/plans/` — one implementation plan per phase, chronological.
- `docs/portfolio-gap-analysis.md` — real sector-gap findings feeding the specialists.
- `future-enhancements.md` — deferred technical refinements (not phases), e.g. true statistical correlation, fund lock-in nuance.
- `src/agentic_frameworks/graph/` — all production code: `state.py` (all `TypedDict` schemas), `mcp_clients.py`, `models.py`, `schemas.py`, `checkpointer.py`, `graph.py`, `nodes/`.
- `tests/graph/` — mirrors `src/agentic_frameworks/graph/` one-to-one.
- `.superpowers/sdd/` — SDD workspaces; completed plans' workspaces are deleted after merge (git history is the record), so this directory should be empty or near-empty between phases.
