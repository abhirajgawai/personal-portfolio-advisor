# Learning Checkpoint — MCP

Three MCP servers in this project: `portfolio_data_mcp` (pre-existing, reused unmodified, read-only), `yfinance-mcp` (third-party, reused via `uvx yfmcp@latest`), and `mfapi_mcp` (this project's own, built from scratch in Phase 6 wrapping MFapi.in).

## 1. What problem does this solve that a plain LLM API call / plain cron job doesn't?

Tool access as a vendor-neutral, composable protocol rather than bespoke per-provider SDK glue. The same MCP toolset pattern connects a totally different data source (yfinance for US/India equities, MFapi.in for Indian mutual funds, a project's own read-only SQLite view for holdings) to the same kind of `pydantic_ai.Agent` call, with the same reject-early/fail-loud conventions, without the graph code caring which transport or vendor is behind it.

## 2. What did it make easy that was hard/verbose without it?

- **Structural exclusion of unwanted capability, not just "don't use it."** `yfinance-mcp`'s options/derivatives/intraday tools are removed from the toolset via `.filtered()` before the model ever sees them — the guardrail is "the tool doesn't exist in this run" rather than "please don't call this tool." This is the same "guardrails are structural, not instructional" principle this project applies everywhere, and MCP's toolset-filtering made it a one-line enforcement rather than a prompt instruction to hope the model honors.
- **Reusing an existing, actively-maintained server instead of rebuilding.** `yfinance-mcp` covers US equity, US ETF, and (via `.NS` suffix) India equity fundamentals with zero new infrastructure — confirmed live against real tickers before committing to it, rather than assumed from its README.
- **Building a small custom server was genuinely small.** `mfapi_mcp` (Phase 6) is three tools (`search_scheme`, `get_scheme_nav`, `get_scheme_meta`) over one free, no-auth API, using the official `mcp.server.fastmcp.FastMCP` — already a transitive dependency via `pydantic-ai`, so it needed no new package at all.

## 3. What did it make hard or awkward?

- **A benign tool-calling problem can crash the whole node if `tool_error_behavior` is misconfigured.** Phase 4/5's real-model gate crashed on the very first live run because `tool_error_behavior="error"` turned an ordinary "model omitted a required argument" into a hard `ToolError` instead of giving the model a bounded self-correction chance via `ModelRetry`. The library default (`"retry"`) was the right choice, but nothing in the docs made the failure mode of the *wrong* setting obvious — it took a real run to find.
- **A fake ticker doesn't raise, it returns a near-empty payload.** The reject-early verification step for equity candidates relies on `longName` presence rather than an exception, because a hallucinated or delisted ticker doesn't error out — it just comes back mostly empty. Any code trusting "no exception = real ticker" would be structurally wrong.
- **Free, no-auth APIs like MFapi.in have real, undocumented data gaps** (no expense ratio field anywhere in the feed, confirmed only by checking) — the Conviction Gate for mutual funds had to be redesigned around what data genuinely exists rather than what a more idealized fund-evaluation checklist would want.

## 4. When would you reach for this in a real job?

Whenever an agent needs to call more than one external data source and you want the graph/agent code to stay decoupled from any one vendor's SDK — especially when some capability (like trade execution) must be structurally impossible to reach, not just discouraged. Less relevant for a single-tool, single-provider integration where a direct SDK call is simpler and MCP's indirection buys nothing.
