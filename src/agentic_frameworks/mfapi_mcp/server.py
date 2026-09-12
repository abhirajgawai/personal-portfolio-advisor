"""A thin, custom MCP server wrapping MFapi.in's search and NAV-history
endpoints for Indian mutual funds.

MFapi.in is a plain REST API with no existing MCP server that met this
project's actual requirements (see docs/superpowers/specs/2026-09-12-
mutual-fund-specialist-design.md's Background/Constraints for the
pre-built-server evaluation). This server exposes exactly two thin,
real operations - the same role yfinance-mcp plays for equities: raw
data only, no analysis. All metrics/scheme-resolution logic lives in
graph/nodes/_mutual_fund_helpers.py, not here.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mfapi")

_BASE_URL = "https://api.mfapi.in"


@mcp.tool()
def search_scheme(query: str) -> str:
    """Search MFapi.in for mutual fund schemes matching a name query.

    Returns a JSON string: a list of {"scheme_code": int, "scheme_name": str}.
    """
    url = f"{_BASE_URL}/mf/search?q={urllib.parse.quote(query)}"
    with urllib.request.urlopen(url, timeout=15) as response:
        raw = json.loads(response.read())
    results = [{"scheme_code": item["schemeCode"], "scheme_name": item["schemeName"]} for item in raw]
    return json.dumps(results)


@mcp.tool()
def get_scheme_nav(scheme_code: int) -> str:
    """Fetch a mutual fund scheme's metadata and full NAV history from MFapi.in.

    Returns a JSON string: {"meta": {...7 fields...}, "data": [{"date": "DD-MM-YYYY", "nav": "..."}, ...]}
    (newest-first, exactly as MFapi.in returns it).
    """
    url = f"{_BASE_URL}/mf/{scheme_code}"
    with urllib.request.urlopen(url, timeout=15) as response:
        raw = json.loads(response.read())
    return json.dumps(raw)


if __name__ == "__main__":
    mcp.run()
