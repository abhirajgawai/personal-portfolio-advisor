# Personal Portfolio Advisor

A multi-agent advisor for my own long-term investment portfolio, built as a LangGraph graph. It reads my real holdings, looks for gaps, and returns typed recommendations (hold, sell, reduce, add, or new-buy candidates). It is advisory only: no part of it can place a trade.

It is also a hands-on project to learn LangGraph, Pydantic AI, MCP and Temporal properly, on real data instead of a toy example.

## How it works

```
Research ──► Risk Analyst ──┬─► US Equity/ETF Specialist ─┐
                            ├─► India Equity Specialist ──┼─► Synthesizer ─► Human approval gate
                            └─► Mutual Fund Specialist ───┘
```

- **Research** and **Risk Analyst** are plain, deterministic functions. They gather and measure (holdings, allocation, concentration) and never give opinions.
- **Three specialists run in parallel.** Each new-buy idea must pass a reject-early **Conviction Gate**: portfolio fit, then fundamentals, then valuation, then style fit. "No good candidate this cycle" is a normal result.
- **Synthesizer** merges everything into one Pydantic-validated recommendation object, never free text.
- **Human approval gate.** Any sell or new-buy suggestion pauses the graph (LangGraph interrupt) until I approve it.
- **Temporal daily monitor** runs the graph on a schedule and resumes cleanly after a crash or restart.

## Design choices

- **Guardrails are structural, not prompts.** No broker or order tool is registered anywhere, so the graph cannot trade even if a model asks to.
- **Two databases.** Holdings are read from the portfolio database through a read-only (`mode=ro`) connection. Agent writes (checkpoints, decision logs, long-term memory) go to a separate `agent_memory.db`. An OS-level lock stops two writers from touching it at the same time.
- **Tools via MCP.** Fundamentals come from the open-source `yfinance-mcp` server (US and NSE tickers). Indian mutual fund data comes from a small MCP server I wrote over MFapi.in (`src/agentic_frameworks/mfapi_mcp`).
- **Memory.** LangGraph checkpointer for thread state, LangGraph Store for cross-run memory, such as rejected candidates so the same research isn't paid for twice.
- **Cost control.** Cheap checks run before any LLM call, output sizes and retries have hard caps, and models are reached through OpenRouter with a fallback model.

## How I checked it works

- pytest suites for every node, the graph, the Temporal workflow and the DB lock (`tests/`).
- Each LLM node has a **real-model gate**: a test run against the actual model, not a mock, before the phase counts as done.

## AI errors this caught

- **Truncated structured output.** The first real-model run of the Synthesizer failed before validation even ran. The model wrote so many long risk flags that it ran out of output tokens mid-JSON. Fix: a hard length cap on every free-text field in the schema (commit `a04c61d`).
- **Fallback that never fell back.** The gate failed with "exceeded maximum output retries". Reading the pydantic-ai source showed that `FallbackModel` only switches models on API errors, not on output-validation failures, so a model that kept returning invalid output was retried instead of replaced. Fix: change the primary model slot (commit `0eee63f`).

## Stack

Python 3.14, LangGraph, Pydantic AI, MCP (`langchain-mcp-adapters`), Temporal, SQLite, OpenRouter, uv, pytest.

## Running it

Needs a portfolio SQLite database in the schema of my portfolio dashboard project and an `OPENROUTER_API_KEY` (see `.env.example`). Install with `uv sync`, run tests with `uv run pytest`.
