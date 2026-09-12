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


def test_compute_fund_metrics_returns_none_cagr_since_inception_for_single_entry_history():
    from datetime import date

    from agentic_frameworks.graph.nodes._mutual_fund_helpers import _compute_fund_metrics

    history = [{"date": date(2026, 1, 1), "nav": 100.0}]

    metrics = _compute_fund_metrics(history)

    assert metrics["cagr_since_inception"] is None


def test_compute_fund_metrics_returns_none_volatility_for_insufficient_history():
    from datetime import date

    from agentic_frameworks.graph.nodes._mutual_fund_helpers import _compute_fund_metrics

    # Only spans part of one month - _resample_month_end will produce
    # at most 1 monthly entry, so there are fewer than 2 monthly
    # returns to compute a standard deviation from.
    history = [
        {"date": date(2026, 1, 5), "nav": 100.0},
        {"date": date(2026, 1, 15), "nav": 102.0},
    ]

    metrics = _compute_fund_metrics(history)

    assert metrics["volatility"] is None


def test_parse_yfinance_price_history_extracts_date_and_close():
    from datetime import date

    from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
        _parse_yfinance_price_history,
    )

    raw = (
        "| Date                      |    Open |    High |     Low |   Close |   Volume |   Dividends |   Stock Splits |\n"
        "|:--------------------------|--------:|--------:|--------:|--------:|---------:|------------:|---------------:|\n"
        "| 2026-07-01 00:00:00+05:30 | 76545.2 | 78664.9 | 75474.4 | 78094.6 |   430400 |           0 |              0 |\n"
        "| 2026-08-01 00:00:00+05:30 | 78883.3 | 79143.1 | 76751.3 | 76957.3 |   319600 |           0 |              0 |"
    )

    parsed = _parse_yfinance_price_history(raw)

    assert parsed == [
        {"date": date(2026, 7, 1), "close": 78094.6},
        {"date": date(2026, 8, 1), "close": 76957.3},
    ]


async def test_compute_benchmark_difference_returns_none_when_ticker_unavailable():
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
        _compute_benchmark_difference,
    )

    result = await _compute_benchmark_difference([], None, None)

    assert result is None


async def test_compute_benchmark_difference_returns_none_when_years_is_zero():
    from datetime import date

    from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
        _compute_benchmark_difference,
    )

    # Single-entry history - track_record_years will be 0, so this must
    # return None WITHOUT making any real network call (verifying the
    # reject-early fix: pass toolset=None to prove the function never
    # tries to use it).
    fund_history = [{"date": date(2026, 1, 1), "nav": 100.0}]

    result = await _compute_benchmark_difference(fund_history, "^NSEI", None)

    assert result is None


async def test_compute_benchmark_difference_returns_none_when_benchmark_lacks_coverage():
    from datetime import date

    from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
        _compute_benchmark_difference,
    )

    # NIFTYSMLCAP250.NS's real yfinance history only starts 2005-04-01
    # (live-verified). A fund date range entirely before that has no
    # benchmark data to compare against on either end, so this must
    # return None - a real, disclosed coverage gap, not a fabricated
    # comparison.
    fund_history = [
        {"date": date(1990, 1, 1), "nav": 100.0},
        {"date": date(2000, 1, 1), "nav": 130.0},
    ]

    toolset = get_yfinance_toolset()
    async with toolset:
        result = await _compute_benchmark_difference(fund_history, "NIFTYSMLCAP250.NS", toolset)

    assert result is None


async def test_compute_benchmark_difference_computes_real_comparison():
    from datetime import date

    from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
        _compute_benchmark_difference,
    )

    fund_history = [
        {"date": date(2023, 1, 1), "nav": 100.0},
        {"date": date(2026, 1, 1), "nav": 130.0},
    ]

    toolset = get_yfinance_toolset()
    async with toolset:
        result = await _compute_benchmark_difference(fund_history, "^NSEI", toolset)

    assert result is not None
    assert "fund_cagr" in result
    assert "benchmark_cagr" in result
    assert "difference" in result


def test_compute_redundancy_correlations_identifies_perfectly_correlated_funds():
    from datetime import date

    from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
        _compute_redundancy_correlations,
    )

    # Fund B's NAV is always exactly 2x Fund A's - perfectly correlated (1.0).
    fund_a = [
        {"date": date(2024, 1, 1), "nav": 100.0},
        {"date": date(2024, 2, 1), "nav": 110.0},
        {"date": date(2024, 3, 1), "nav": 105.0},
        {"date": date(2024, 4, 1), "nav": 120.0},
    ]
    fund_b = [{"date": h["date"], "nav": h["nav"] * 2} for h in fund_a]

    correlations = _compute_redundancy_correlations({"ISIN_A": fund_a, "ISIN_B": fund_b})

    assert abs(correlations["ISIN_A"]["ISIN_B"] - 1.0) < 0.001
    assert abs(correlations["ISIN_B"]["ISIN_A"] - 1.0) < 0.001
    assert "ISIN_A" not in correlations["ISIN_A"]


def test_compute_redundancy_correlations_skips_pair_with_fewer_than_two_common_months():
    from datetime import date

    from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
        _compute_redundancy_correlations,
    )

    # Fund A resamples to monthly NAVs for Jan/Feb/Mar 2024, giving
    # monthly returns keyed by (2024, 2) [Jan->Feb] and (2024, 3)
    # [Feb->Mar]. Fund B only has Feb/Mar 2024, giving a single monthly
    # return keyed by (2024, 3) [Feb->Mar]. The two funds therefore
    # share exactly one common resampled month - below the len(...) < 2
    # threshold the implementation requires - so this pair must be
    # skipped entirely rather than crashing (division by zero/undefined
    # stdev) or fabricating a correlation from a single data point.
    fund_a = [
        {"date": date(2024, 1, 1), "nav": 100.0},
        {"date": date(2024, 2, 1), "nav": 105.0},
        {"date": date(2024, 3, 1), "nav": 110.0},
    ]
    fund_b = [
        {"date": date(2024, 2, 1), "nav": 200.0},
        {"date": date(2024, 3, 1), "nav": 210.0},
    ]

    correlations = _compute_redundancy_correlations({"ISIN_A": fund_a, "ISIN_B": fund_b})

    # Skipped means "not present as a key at all" - the implementation
    # pre-seeds an empty dict per fund and only adds an entry for pairs
    # that clear the >= 2 common months bar, so a skipped pair must be
    # absent from both sides' correlation maps (not present as, say,
    # None or 0.0).
    assert "ISIN_B" not in correlations["ISIN_A"]
    assert "ISIN_A" not in correlations["ISIN_B"]
    assert correlations["ISIN_A"] == {}
    assert correlations["ISIN_B"] == {}


def test_compute_redundancy_correlations_computes_genuine_non_scalar_correlation():
    from datetime import date

    from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
        _compute_redundancy_correlations,
    )

    # Fund A and Fund B are NOT scalar multiples of each other (their
    # NAV ratio changes every month: 100/100, 105/110, 126/121,
    # 132.3/108.9), so a correct "correlate the monthly RETURNS"
    # implementation and a buggy "correlate the raw NAV levels"
    # implementation would produce two genuinely different numbers here
    # (see hand-computation below) - unlike the existing 2x-scalar
    # fixture, which can't tell the two implementations apart.
    #
    # Monthly returns (resampling is a no-op here - one NAV entry per
    # calendar month):
    #   Fund A: Jan->Feb +10%, Feb->Mar +10%, Mar->Apr -10%
    #   Fund B: Jan->Feb  +5%, Feb->Mar +20%, Mar->Apr  +5%
    #
    # Hand-computed Pearson correlation of these two return series
    # (verified exactly using fractions, see task report):
    #   mean_A = 1/30, mean_B = 1/10
    #   deviations_A = [1/15, 1/15, -2/15]
    #   deviations_B = [-1/20, 1/10, -1/20]
    #   sum(dev_A * dev_B) = 1/100
    #   sum(dev_A^2) = 2/75, sum(dev_B^2) = 3/200
    #   r = (1/100) / sqrt((2/75) * (3/200)) = 0.01 / 0.02 = 0.5 exactly
    fund_a = [
        {"date": date(2024, 1, 1), "nav": 100.0},
        {"date": date(2024, 2, 1), "nav": 110.0},
        {"date": date(2024, 3, 1), "nav": 121.0},
        {"date": date(2024, 4, 1), "nav": 108.9},
    ]
    fund_b = [
        {"date": date(2024, 1, 1), "nav": 100.0},
        {"date": date(2024, 2, 1), "nav": 105.0},
        {"date": date(2024, 3, 1), "nav": 126.0},
        {"date": date(2024, 4, 1), "nav": 132.3},
    ]

    correlations = _compute_redundancy_correlations({"ISIN_A": fund_a, "ISIN_B": fund_b})

    assert abs(correlations["ISIN_A"]["ISIN_B"] - 0.5) < 0.001
    assert abs(correlations["ISIN_B"]["ISIN_A"] - 0.5) < 0.001


async def test_compute_mf_category_gaps_finds_expected_gaps_given_real_holdings():
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
        _compute_mf_category_gaps,
    )
    from agentic_frameworks.graph.nodes.research import research_node

    research_result = await research_node({})

    gaps = _compute_mf_category_gaps(research_result["holdings_by_bucket"])

    assert set(gaps) == {
        "Large Cap",
        "Multi Cap",
        "ELSS",
        "Conservative Hybrid",
        "Liquid",
        "Short Duration",
        "Corporate Bond",
        "Gilt",
    }


def test_compute_mf_category_gaps_raises_on_unmapped_category():
    import pytest

    from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
        _compute_mf_category_gaps,
    )

    holdings_by_bucket = {
        "mutual_fund": {
            "IN": [{"ticker": "INF000000000", "name": "Fictional Fund", "sector": "Not A Real Category"}],
        }
    }

    with pytest.raises(KeyError):
        _compute_mf_category_gaps(holdings_by_bucket)


async def test_resolve_scheme_by_name_and_direct_plan_finds_real_fund():
    from agentic_frameworks.graph.mcp_clients import get_mfapi_toolset
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
        _resolve_scheme_by_name_and_direct_plan,
    )

    toolset = get_mfapi_toolset()
    async with toolset:
        scheme_code, category, scheme_name = await _resolve_scheme_by_name_and_direct_plan(
            "ICICI Prudential Gilt Fund", toolset
        )

    assert scheme_code is not None
    assert "gilt" in category.lower()
    assert "direct" in scheme_name.lower()


async def test_resolve_scheme_by_name_and_direct_plan_raises_on_fabricated_name():
    import pytest

    from agentic_frameworks.graph.mcp_clients import get_mfapi_toolset
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
        _resolve_scheme_by_name_and_direct_plan,
    )

    toolset = get_mfapi_toolset()
    async with toolset:
        with pytest.raises(ValueError):
            await _resolve_scheme_by_name_and_direct_plan(
                "Completely Fictional Nonexistent Fund XYZ123", toolset
            )


def test_category_matches_taxonomy_accepts_real_mfapi_category_strings():
    """Live-verified real MFapi scheme_category strings (this session) for
    every case exercised: the taxonomy category name is always a literal
    (case-insensitive) substring."""
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import _category_matches_taxonomy

    assert _category_matches_taxonomy("Equity Scheme - Flexi Cap Fund", "Flexi Cap")
    assert _category_matches_taxonomy("Equity Scheme - Small Cap Fund", "Small Cap")
    assert _category_matches_taxonomy("Equity Schemes - ELSS- Tax Saver Fund", "ELSS")
    assert _category_matches_taxonomy("Hybrid Scheme - Conservative Hybrid Fund", "Conservative Hybrid")
    assert _category_matches_taxonomy("Index Funds - Equity Funds", "Index Fund")


def test_category_matches_taxonomy_rejects_a_real_mismatch():
    """A Gilt fund's real scheme_category must not be accepted as a match
    for a candidate proposed under a different taxonomy category - this is
    the exact class of mismatch Finding 1 guards against."""
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import _category_matches_taxonomy

    assert not _category_matches_taxonomy("Debt Scheme - Gilt Fund", "Flexi Cap")


async def test_resolve_scheme_by_name_and_direct_plan_returns_verified_identity_for_mismatch_check():
    """Confirms the resolved category for a real Gilt fund does NOT match
    an unrelated taxonomy category - proving the combination of
    _resolve_scheme_by_name_and_direct_plan + _category_matches_taxonomy
    is what the specialist node relies on to drop a mismatched candidate
    (Finding 1's node-level behavior, exercised directly here at the
    helper level with a real fund)."""
    from agentic_frameworks.graph.mcp_clients import get_mfapi_toolset
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import (
        _category_matches_taxonomy,
        _resolve_scheme_by_name_and_direct_plan,
    )

    toolset = get_mfapi_toolset()
    async with toolset:
        _scheme_code, resolved_category, _scheme_name = await _resolve_scheme_by_name_and_direct_plan(
            "ICICI Prudential Gilt Fund", toolset
        )

    assert _category_matches_taxonomy(resolved_category, "Gilt")
    assert not _category_matches_taxonomy(resolved_category, "Small Cap")


async def test_get_scheme_isin_uses_lightweight_meta_tool():
    """Confirms _get_scheme_isin returns the correct real ISIN using the
    new get_scheme_meta tool (Finding 4) - a real, unmocked resolution
    against a known scheme."""
    from agentic_frameworks.graph.mcp_clients import get_mfapi_toolset
    from agentic_frameworks.graph.nodes._mutual_fund_helpers import _get_scheme_isin

    toolset = get_mfapi_toolset()
    async with toolset:
        isin = await _get_scheme_isin(122639, toolset)

    assert isin == "INF879O01027"


def test_compute_fund_metrics_returns_none_max_drawdown_for_single_entry_history():
    """A single NAV point has no observable drawdown - this is different
    from "confirmed zero drawdown" (the same fabrication class already
    fixed for cagr_since_inception/volatility)."""
    from datetime import date

    from agentic_frameworks.graph.nodes._mutual_fund_helpers import _compute_fund_metrics

    history = [{"date": date(2026, 1, 1), "nav": 100.0}]

    metrics = _compute_fund_metrics(history)

    assert metrics["max_drawdown"] is None
