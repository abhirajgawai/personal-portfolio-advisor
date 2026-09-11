import pytest

from agentic_frameworks.graph.mcp_clients import get_portfolio_data_client, get_yfinance_toolset


async def test_get_portfolio_data_client_exposes_expected_tools():
    client = get_portfolio_data_client()
    tools = await client.get_tools()
    names = {tool.name for tool in tools}
    assert {"latest_snapshot", "allocation_summary"}.issubset(names)


async def test_yfinance_toolset_retries_on_malformed_tool_call_instead_of_crashing():
    """A tool call missing a required argument must raise pydantic_ai's
    ModelRetry (bounded, self-correctable) rather than crashing the node
    with a raw fastmcp.exceptions.ToolError.

    Real, unmocked call against the yfinance-mcp server: deliberately omits
    the required `symbol` argument to `yfinance_get_price_history`.
    """
    from pydantic_ai.exceptions import ModelRetry

    toolset = get_yfinance_toolset()
    async with toolset:
        with pytest.raises(ModelRetry):
            await toolset.wrapped.direct_call_tool("yfinance_get_price_history", {})
