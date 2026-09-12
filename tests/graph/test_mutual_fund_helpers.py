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


async def test_resolve_scheme_code_matches_real_fund_by_isin():
    from agentic_frameworks.graph.mcp_clients import get_mfapi_toolset
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import _resolve_scheme_code

    toolset = get_mfapi_toolset()
    async with toolset:
        code = await _resolve_scheme_code(
            "Parag Parikh Flexi Cap Fund - Direct Plan Growth", "INF879O01027", toolset
        )

    assert code == 122639


async def test_resolve_scheme_code_handles_letter_digit_spacing_mismatch():
    """Real edge case: the portfolio's stored name has 'Nifty200' (no
    space) while MFapi.in stores 'Nifty 200' (with space) for this
    specific fund - confirmed live this session. The resolver must
    normalize this, not just try the raw name.
    """
    from agentic_frameworks.graph.mcp_clients import get_mfapi_toolset
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import _resolve_scheme_code

    toolset = get_mfapi_toolset()
    async with toolset:
        code = await _resolve_scheme_code(
            "UTI Nifty200 Momentum 30 Index Fund - Direct Plan (Demat)", "INF789F1AUT5", toolset
        )

    assert code == 148703


async def test_resolve_scheme_code_raises_on_no_match():
    from agentic_frameworks.graph.mcp_clients import get_mfapi_toolset
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import _resolve_scheme_code

    toolset = get_mfapi_toolset()
    async with toolset:
        with __import__("pytest").raises(ValueError):
            await _resolve_scheme_code(
                "Completely Fictional Nonexistent Mutual Fund XYZ", "INF000000000", toolset
            )


async def test_fetch_nav_history_returns_chronological_real_data():
    from datetime import date

    from agentic_frameworks.graph.mcp_clients import get_mfapi_toolset
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import _fetch_nav_history

    toolset = get_mfapi_toolset()
    async with toolset:
        history = await _fetch_nav_history(122639, toolset)

    assert len(history) > 1000
    assert all(isinstance(h["date"], date) for h in history)
    assert all(isinstance(h["nav"], float) for h in history)
    assert history[0]["date"] < history[-1]["date"]
    assert history[0]["date"] == date(2013, 5, 28)


def test_compute_fund_metrics_on_synthetic_series():
    from datetime import date

    from agentic_frameworks.graph.nodes._mutual_fund_helpers import _compute_fund_metrics

    # Exactly 2 years, NAV doubles from 100 to 200 - CAGR should be
    # sqrt(2) - 1 ~= 41.42%. No drawdown (monotonic increase).
    history = [
        {"date": date(2024, 1, 1), "nav": 100.0},
        {"date": date(2024, 7, 1), "nav": 130.0},
        {"date": date(2025, 1, 1), "nav": 150.0},
        {"date": date(2025, 7, 1), "nav": 170.0},
        {"date": date(2026, 1, 1), "nav": 200.0},
    ]

    metrics = _compute_fund_metrics(history)

    assert metrics["track_record_years"] == 2.0
    assert abs(metrics["cagr_since_inception"] - (2 ** 0.5 - 1)) < 0.01
    assert metrics["cagr_1y"] is not None
    assert metrics["max_drawdown"] == 0.0


def test_compute_fund_metrics_detects_a_real_drawdown():
    from datetime import date

    from agentic_frameworks.graph.nodes._mutual_fund_helpers import _compute_fund_metrics

    history = [
        {"date": date(2024, 1, 1), "nav": 100.0},
        {"date": date(2024, 6, 1), "nav": 150.0},
        {"date": date(2024, 9, 1), "nav": 90.0},
        {"date": date(2025, 1, 1), "nav": 120.0},
    ]

    metrics = _compute_fund_metrics(history)

    assert abs(metrics["max_drawdown"] - (90.0 / 150.0 - 1)) < 0.001


def test_compute_fund_metrics_returns_none_cagr_when_period_exceeds_history():
    from datetime import date

    from agentic_frameworks.graph.nodes._mutual_fund_helpers import _compute_fund_metrics

    history = [
        {"date": date(2025, 1, 1), "nav": 100.0},
        {"date": date(2025, 6, 1), "nav": 110.0},
    ]

    metrics = _compute_fund_metrics(history)

    assert metrics["cagr_3y"] is None
