"""Tests for the yfinance-mcp client wiring."""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


async def test_get_yfinance_toolset_connects_and_exposes_tools():
    from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset

    toolset = get_yfinance_toolset()
    agent = Agent(TestModel(), toolsets=[toolset])

    result = await agent.run("what tools do you have?")

    assert "yfinance_get_ticker_info" in result.output
