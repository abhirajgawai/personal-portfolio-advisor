"""MCP client wiring for the read-only portfolio_data_mcp server.

portfolio_data_mcp lives in a separate project (E:\\agentic-portfolio-dashboard)
with its own venv - it is not importable from this project's environment, so
it must be launched as a subprocess speaking MCP over stdio.
"""

from __future__ import annotations

from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic_ai.mcp import FastMCPClient, MCPToolset, StdioTransport
from pydantic_ai.toolsets.filtered import FilteredToolset

PORTFOLIO_DATA_MCP_ROOT = r"E:\agentic-portfolio-dashboard"
PORTFOLIO_DATA_MCP_PYTHON = r"E:\agentic-portfolio-dashboard\.venv\Scripts\python.exe"

# Guardrail (CLAUDE.md: "long-term investing only, no options/derivatives,
# no intraday timing") enforced structurally here, not just left to the
# system prompt: these tools are never registered, so the model can never
# call them regardless of what a prompt says.
_EXCLUDED_TOOLS = {
    "yfinance_get_option_chain",
    "yfinance_get_option_dates",
    "yfinance_screen_gappers",
}


def get_portfolio_data_client() -> MultiServerMCPClient:
    """Build the MCP client for the portfolio_data_mcp stdio server.

    handle_tool_errors=False so a failed tool call raises instead of being
    swallowed into error-content - Research fails loud, never on fabricated
    or partial data.
    """
    return MultiServerMCPClient(
        {
            "portfolio_data": {
                "transport": "stdio",
                "command": PORTFOLIO_DATA_MCP_PYTHON,
                "args": ["-m", "portfolio_data_mcp.server"],
                "cwd": PORTFOLIO_DATA_MCP_ROOT,
            }
        },
        handle_tool_errors=False,
    )


def get_yfinance_toolset() -> FilteredToolset:
    """Build the toolset for the yfinance-mcp server (US and India equity/ETF data).

    Launched via `uvx yfmcp@latest` - an external package, not a project
    dependency, same pattern as portfolio_data_mcp being a sibling
    subprocess rather than an import.

    tool_error_behavior="retry" (the library default) so a malformed
    tool call - e.g. the model omitting a required argument - raises
    ModelRetry with the actual validation message, letting the model
    self-correct within the bounded retry budget already set on each
    Agent (retries=2), rather than crashing the whole node. This is a
    different case from a tool call that succeeds but returns
    sparse/incomplete real data - that's not an error at all, just
    less data for the model's own judgment to weigh, and is untouched
    by this setting either way. _verify_ticker's direct_call_tool call
    is unaffected: yfinance_get_ticker_info is confirmed (Phase 4 live
    testing) to never raise ToolError even on a fake ticker - it
    returns a near-empty payload instead - regardless of this setting.

    `.filtered(...)` removes options-chain/gapper-screener tools from
    what the model can even see - a structural guardrail (CLAUDE.md:
    "long-term investing only, no options/derivatives") rather than one
    that relies on the system prompt alone to avoid using them.
    """
    transport = StdioTransport(command="uvx", args=["yfmcp@latest"])
    client = FastMCPClient(transport)
    toolset = MCPToolset(client, tool_error_behavior="retry")
    return toolset.filtered(lambda ctx, tool_def: tool_def.name not in _EXCLUDED_TOOLS)
