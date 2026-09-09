"""The Risk Analyst node: pure risk measurements over existing holdings.

No MCP client, no I/O - every function here is a synchronous
transformation of the holdings list the Research node already fetched.
Reports raw numbers only; no thresholds or verdicts.
"""

from __future__ import annotations


def _concentration(holdings: list[dict]) -> dict:
    total = sum(h["market_value_base"] for h in holdings)

    by_asset_class_totals: dict[str, float] = {}
    for h in holdings:
        by_asset_class_totals.setdefault(h["asset_class"], 0.0)
        by_asset_class_totals[h["asset_class"]] += h["market_value_base"]

    by_holding = [
        {
            "ticker": h["ticker"],
            "name": h["name"],
            "asset_class": h["asset_class"],
            "pct_of_portfolio": h["market_value_base"] / total * 100,
            "pct_of_asset_class": (
                h["market_value_base"] / by_asset_class_totals[h["asset_class"]] * 100
            ),
        }
        for h in holdings
    ]

    by_asset_class = [
        {"asset_class": asset_class, "pct_of_portfolio": value / total * 100}
        for asset_class, value in by_asset_class_totals.items()
    ]

    return {"by_holding": by_holding, "by_asset_class": by_asset_class}
