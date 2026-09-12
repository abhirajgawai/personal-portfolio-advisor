"""Tests for the Mutual Fund Specialist nodes' shared helpers."""

from __future__ import annotations


def test_mf_category_to_taxonomy_maps_all_five_real_categories():
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import MF_CATEGORY_TO_TAXONOMY

    assert MF_CATEGORY_TO_TAXONOMY == {
        "Flexi Cap": "Flexi Cap",
        "Hybrid": "Aggressive Hybrid",
        "Index Fund": "Index Fund",
        "Small Cap": "Small Cap",
        "Mid Cap": "Mid Cap",
    }


def test_benchmark_tickers_reflect_live_verification():
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import MF_CATEGORY_BENCHMARK_TICKERS

    assert MF_CATEGORY_BENCHMARK_TICKERS["Large Cap"] == "^NSEI"
    assert MF_CATEGORY_BENCHMARK_TICKERS["Small Cap"] == "NIFTYSMLCAP250.NS"
    assert MF_CATEGORY_BENCHMARK_TICKERS["Mid Cap"] == "NIFTYMIDCAP150.NS"
    assert MF_CATEGORY_BENCHMARK_TICKERS["Liquid"] == "LIQUIDBEES.NS"
    assert MF_CATEGORY_BENCHMARK_TICKERS["Gilt"] is None
    assert MF_CATEGORY_BENCHMARK_TICKERS["Corporate Bond"] is None
    assert MF_CATEGORY_BENCHMARK_TICKERS["Short Duration"] is None
    assert MF_CATEGORY_BENCHMARK_TICKERS["Aggressive Hybrid"] is None
    assert MF_CATEGORY_BENCHMARK_TICKERS["Conservative Hybrid"] is None
    assert "Index Fund" not in MF_CATEGORY_BENCHMARK_TICKERS


def test_index_fund_benchmark_tickers_by_isin_reflect_live_verification():
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
        INDEX_FUND_BENCHMARK_TICKERS_BY_ISIN,
    )

    assert INDEX_FUND_BENCHMARK_TICKERS_BY_ISIN["INF179K01WN9"] == "^BSESN"
    assert INDEX_FUND_BENCHMARK_TICKERS_BY_ISIN["INF789F1AUT5"] == "NIFTY200MOMENTM30.NS"
    assert INDEX_FUND_BENCHMARK_TICKERS_BY_ISIN["INF754K01QD2"] is None
    assert INDEX_FUND_BENCHMARK_TICKERS_BY_ISIN["INF789F1AYN0"] is None
