"""Research node: fetch current portfolio holdings via portfolio_data_mcp."""

from __future__ import annotations

import json

from agentic_frameworks.graph import mcp_clients
from agentic_frameworks.graph.state import ResearchState


def _tool_result_to_dict(content: list[dict]) -> dict:
    """Parse an MCP tool's content-block list into the JSON dict it wraps.

    langchain-mcp-adapters returns tool output as
    [{"type": "text", "text": "<json string>"}], not a parsed dict.
    """
    return json.loads(content[0]["text"])


def _group_by_bucket(holdings: list[dict]) -> dict[str, dict[str, list[dict]]]:
    """Group holdings as {asset_class: {geography: [holding, ...]}}."""
    buckets: dict[str, dict[str, list[dict]]] = {}
    for holding in holdings:
        by_geography = buckets.setdefault(holding["asset_class"], {})
        by_geography.setdefault(holding["geography"], []).append(holding)
    return buckets


async def research_node(state: dict) -> ResearchState:
    """Fetch current holdings and allocation, grouped by asset_class x geography.

    Fails loud: any MCP/subprocess/DB error propagates uncaught, since no
    downstream node can produce a meaningful recommendation from partial or
    fabricated holdings data.
    """
    client = mcp_clients.get_portfolio_data_client()
    tools = await client.get_tools()
    tools_by_name = {tool.name: tool for tool in tools}

    snapshot_content = await tools_by_name["latest_snapshot"].ainvoke({})
    allocation_content = await tools_by_name["allocation_summary"].ainvoke({})

    snapshot = _tool_result_to_dict(snapshot_content)
    allocation = _tool_result_to_dict(allocation_content)

    return {
        "holdings": snapshot["holdings"],
        "allocation": allocation,
        "holdings_by_bucket": _group_by_bucket(snapshot["holdings"]),
    }
