"""The US Equity/ETF Specialist node: proposes, verifies, and evaluates
US equity/ETF new-buy candidates against the New-Buy Conviction Gate.

First LLM-calling node in this graph. Fails loud, same as every prior
node - no try/except, except pydantic-ai's own bounded retry-on-
validation-failure (Agent(retries=...)), which validates output
correctness rather than swallowing a real error.
"""

from __future__ import annotations

import json

from pydantic_ai.mcp import MCPToolset


async def _verify_ticker(symbol: str, toolset: MCPToolset) -> bool:
    """Confirm `symbol` resolves to a real, current instrument via yfinance-mcp.

    A nonexistent/delisted ticker doesn't raise - yfinance-mcp returns a
    near-empty payload instead (confirmed live: {"trailingPegRatio": null}
    for a fake symbol). `longName` being present is the reliable signal
    a real instrument was found.
    """
    result = await toolset.direct_call_tool("yfinance_get_ticker_info", {"symbol": symbol})
    data = json.loads(result)
    return data.get("longName") is not None
