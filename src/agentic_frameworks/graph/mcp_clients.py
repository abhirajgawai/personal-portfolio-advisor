"""MCP client wiring for the read-only portfolio_data_mcp server.

portfolio_data_mcp lives in a separate project (E:\\agentic-portfolio-dashboard)
with its own venv - it is not importable from this project's environment, so
it must be launched as a subprocess speaking MCP over stdio.
"""

from __future__ import annotations

from langchain_mcp_adapters.client import MultiServerMCPClient

PORTFOLIO_DATA_MCP_ROOT = r"E:\agentic-portfolio-dashboard"
PORTFOLIO_DATA_MCP_PYTHON = r"E:\agentic-portfolio-dashboard\.venv\Scripts\python.exe"


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
