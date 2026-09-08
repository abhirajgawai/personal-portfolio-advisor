from agentic_frameworks.graph.mcp_clients import get_portfolio_data_client


async def test_get_portfolio_data_client_exposes_expected_tools():
    client = get_portfolio_data_client()
    tools = await client.get_tools()
    names = {tool.name for tool in tools}
    assert {"latest_snapshot", "allocation_summary"}.issubset(names)
