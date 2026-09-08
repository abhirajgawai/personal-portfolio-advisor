import pytest

from agentic_frameworks.graph.nodes.research import research_node


async def test_research_node_groups_real_holdings_by_asset_class_and_geography():
    result = await research_node({})

    buckets = result["holdings_by_bucket"]
    assert set(buckets.keys()) == {"equity", "etf", "mutual_fund", "real_estate"}

    assert len(buckets["real_estate"]["IN"]) == 1
    assert len(buckets["equity"]["IN"]) == 36
    assert len(buckets["equity"]["US"]) == 3
    assert len(buckets["etf"]["IN"]) == 2
    assert len(buckets["etf"]["US"]) == 3
    assert len(buckets["mutual_fund"]["IN"]) == 9

    assert result["allocation"]["totalBase"] > 0
    assert result["holdings"]


async def test_research_node_fails_loud_on_bad_server_config(monkeypatch):
    import agentic_frameworks.graph.mcp_clients as mcp_clients
    from langchain_mcp_adapters.client import MultiServerMCPClient

    def broken_client():
        return MultiServerMCPClient(
            {
                "portfolio_data": {
                    "transport": "stdio",
                    "command": mcp_clients.PORTFOLIO_DATA_MCP_PYTHON,
                    "args": ["-m", "nonexistent_module_xyz"],
                    "cwd": mcp_clients.PORTFOLIO_DATA_MCP_ROOT,
                }
            },
            handle_tool_errors=False,
        )

    monkeypatch.setattr(mcp_clients, "get_portfolio_data_client", broken_client)

    with pytest.raises(Exception):
        await research_node({})
