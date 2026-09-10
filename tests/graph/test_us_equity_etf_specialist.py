"""Tests for the US Equity/ETF Specialist node and its stages."""

from __future__ import annotations


async def test_verify_ticker_confirms_real_instrument():
    from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset
    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import _verify_ticker

    toolset = get_yfinance_toolset()
    async with toolset:
        assert await _verify_ticker("AAPL", toolset) is True


async def test_verify_ticker_rejects_fake_ticker():
    from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset
    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import _verify_ticker

    toolset = get_yfinance_toolset()
    async with toolset:
        assert await _verify_ticker("NOTAREALTICKER999", toolset) is False
