"""Tests for the custom mfapi_mcp server, launched as a real subprocess."""

from __future__ import annotations

import json


async def test_search_scheme_finds_a_real_known_fund():
    from agentic_frameworks.graph.mcp_clients import get_mfapi_toolset

    toolset = get_mfapi_toolset()
    async with toolset:
        result = await toolset.direct_call_tool("search_scheme", {"query": "Parag Parikh Flexi Cap"})
        candidates = json.loads(result)

    assert isinstance(candidates, list)
    assert any(c["scheme_code"] == 122639 for c in candidates)


async def test_get_scheme_nav_returns_real_meta_and_history():
    from agentic_frameworks.graph.mcp_clients import get_mfapi_toolset

    toolset = get_mfapi_toolset()
    async with toolset:
        result = await toolset.direct_call_tool("get_scheme_nav", {"scheme_code": 122639})
        data = json.loads(result)

    assert data["meta"]["isin_growth"] == "INF879O01027"
    assert data["meta"]["scheme_category"] == "Equity Scheme - Flexi Cap Fund"
    assert len(data["data"]) > 1000
    assert "date" in data["data"][0]
    assert "nav" in data["data"][0]


async def test_get_scheme_meta_returns_real_meta_and_single_latest_entry():
    """Live-verified this session: MFapi.in's /mf/<code>/latest endpoint
    returns the same meta fields as get_scheme_nav but with `data`
    truncated to a single, most-recent NAV entry - a much cheaper call
    for callers that only need scheme identity (Finding 4)."""
    from agentic_frameworks.graph.mcp_clients import get_mfapi_toolset

    toolset = get_mfapi_toolset()
    async with toolset:
        result = await toolset.direct_call_tool("get_scheme_meta", {"scheme_code": 122639})
        data = json.loads(result)

    assert data["meta"]["isin_growth"] == "INF879O01027"
    assert data["meta"]["scheme_category"] == "Equity Scheme - Flexi Cap Fund"
    assert len(data["data"]) == 1
    assert "date" in data["data"][0]
    assert "nav" in data["data"][0]
