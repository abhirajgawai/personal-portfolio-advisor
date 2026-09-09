"""Tests for the Risk Analyst node and its helper functions."""

from __future__ import annotations

import pytest

HOLDINGS = [
    {
        "platform": "zerodha", "geography": "IN", "ticker": "INF846K01K35",
        "name": "Axis Small Cap Fund Direct Growth", "currency": "INR",
        "sector": "Small Cap", "asset_class": "mutual_fund",
        "quantity": 1000.0, "price": 10.0, "market_value_base": 10000.0,
        "updated_at": "2026-09-08T00:00:00Z",
    },
    {
        "platform": "zerodha", "geography": "IN", "ticker": "INF179K01WN9",
        "name": "HDFC BSE Sensex Index Fund - Direct Plan - Growth Option",
        "currency": "INR", "sector": "Index Fund", "asset_class": "mutual_fund",
        "quantity": 1000.0, "price": 10.0, "market_value_base": 10000.0,
        "updated_at": "2026-09-08T00:00:00Z",
    },
    {
        "platform": "zerodha", "geography": "IN", "ticker": "INF769K01LG4",
        "name": "Mirae Asset Nifty Smallcap 250 Momentum Quality 100 ETF Fund of Fund Direct Plan",
        "currency": "INR", "sector": "Small Cap", "asset_class": "etf",
        "quantity": 1000.0, "price": 10.0, "market_value_base": 10000.0,
        "updated_at": "2026-09-08T00:00:00Z",
    },
    {
        "platform": "moomoo", "geography": "US", "ticker": "VOO",
        "name": "Vanguard S&P 500 ETF", "currency": "USD",
        "sector": "US Large Cap", "asset_class": "etf",
        "quantity": 20.0, "price": 500.0, "market_value_base": 10000.0,
        "updated_at": "2026-09-08T00:00:00Z",
    },
    {
        "platform": "zerodha", "geography": "IN", "ticker": "HDFCBANK",
        "name": "HDFC Bank Ltd", "currency": "INR",
        "sector": "Banking", "asset_class": "equity",
        "quantity": 100.0, "price": 200.0, "market_value_base": 20000.0,
        "updated_at": "2026-09-08T00:00:00Z",
    },
    {
        "platform": "groww", "geography": "IN", "ticker": "TALOJA_PROPERTY",
        "name": "Asavari Taloja", "currency": "INR",
        "sector": "Residential", "asset_class": "real_estate",
        "quantity": 1.0, "price": 40000.0, "market_value_base": 40000.0,
        "updated_at": "2026-09-08T00:00:00Z",
    },
]


def test_concentration_computes_correct_weights():
    from agentic_frameworks.graph.nodes.risk_analyst import _concentration

    result = _concentration(HOLDINGS)

    by_holding = {h["ticker"]: h for h in result["by_holding"]}
    assert by_holding["INF846K01K35"]["pct_of_portfolio"] == pytest.approx(10.0)
    assert by_holding["INF846K01K35"]["pct_of_asset_class"] == pytest.approx(50.0)
    assert by_holding["HDFCBANK"]["pct_of_portfolio"] == pytest.approx(20.0)
    assert by_holding["HDFCBANK"]["pct_of_asset_class"] == pytest.approx(100.0)
    assert by_holding["TALOJA_PROPERTY"]["pct_of_portfolio"] == pytest.approx(40.0)

    by_asset_class = {a["asset_class"]: a["pct_of_portfolio"] for a in result["by_asset_class"]}
    assert by_asset_class["mutual_fund"] == pytest.approx(20.0)
    assert by_asset_class["etf"] == pytest.approx(20.0)
    assert by_asset_class["equity"] == pytest.approx(20.0)
    assert by_asset_class["real_estate"] == pytest.approx(40.0)


def test_category_overlap_includes_only_shared_sectors():
    from agentic_frameworks.graph.nodes.risk_analyst import _category_overlap

    result = _category_overlap(HOLDINGS)

    assert set(result["by_sector"].keys()) == {"Small Cap"}

    small_cap_tickers = {h["ticker"] for h in result["by_sector"]["Small Cap"]}
    assert small_cap_tickers == {"INF846K01K35", "INF769K01LG4"}

    small_cap_asset_classes = {h["asset_class"] for h in result["by_sector"]["Small Cap"]}
    assert small_cap_asset_classes == {"mutual_fund", "etf"}


def test_currency_exposure_computes_correct_totals():
    from agentic_frameworks.graph.nodes.risk_analyst import _currency_exposure

    result = _currency_exposure(HOLDINGS)

    assert result["INR"]["value_base"] == pytest.approx(90000.0)
    assert result["INR"]["pct_of_portfolio"] == pytest.approx(90.0)
    assert result["USD"]["value_base"] == pytest.approx(10000.0)
    assert result["USD"]["pct_of_portfolio"] == pytest.approx(10.0)
