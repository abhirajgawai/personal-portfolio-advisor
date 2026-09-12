"""Shared helpers for the Mutual Fund Specialist nodes
(mutual_fund_holdings_analyzer.py, mutual_fund_specialist.py).

Curated taxonomy, real-category mapping, and live-verified benchmark
ticker tables, plus scheme resolution, NAV/meta fetching, and metrics
computation - all consolidated in this one shared module.
"""

from __future__ import annotations

import json
import re
import statistics
from datetime import date, datetime, timedelta

from agentic_frameworks.graph.schemas import MF_CATEGORY_TAXONOMY

MF_CATEGORY_TO_TAXONOMY = {
    "Flexi Cap": "Flexi Cap",
    "Hybrid": "Aggressive Hybrid",
    "Index Fund": "Index Fund",
    "Small Cap": "Small Cap",
    "Mid Cap": "Mid Cap",
}

MF_CATEGORY_BENCHMARK_TICKERS = {
    "Large Cap": "^NSEI",
    "Flexi Cap": "^CRSLDX",
    "Multi Cap": "^CRSLDX",
    "ELSS": "^CRSLDX",
    "Small Cap": "NIFTYSMLCAP250.NS",
    "Mid Cap": "NIFTYMIDCAP150.NS",
    "Liquid": "LIQUIDBEES.NS",
    "Short Duration": None,
    "Corporate Bond": None,
    "Gilt": None,
    "Aggressive Hybrid": None,
    "Conservative Hybrid": None,
}

INDEX_FUND_BENCHMARK_TICKERS_BY_ISIN = {
    "INF179K01WN9": "^BSESN",
    "INF789F1AUT5": "NIFTY200MOMENTM30.NS",
    "INF754K01QD2": None,
    "INF789F1AYN0": None,
}


def _normalize_letter_digit_spacing(name: str) -> str:
    """Insert a space between a letter and an immediately-following digit.

    Real edge case, live-verified this session: the portfolio's stored
    name for one fund is 'Nifty200' (no space) while MFapi.in stores
    'Nifty 200' (with space) - a single data-entry quirk, not a
    systematic pattern (every other fund's name already has correct
    spacing). This normalization is tried as a fallback, not the
    primary query, since the primary (raw) name resolves 8 of 9 real
    holdings correctly on its own.
    """
    return re.sub(r"([a-zA-Z])(\d)", r"\1 \2", name)


async def _search_scheme(query: str, toolset) -> list[dict]:
    """toolset here is the plain MCPToolset returned by get_mfapi_toolset()
    (not a FilteredToolset like the yfinance toolset), so it exposes
    direct_call_tool directly - no .wrapped indirection needed, unlike
    calls against the yfinance toolset elsewhere in this file."""
    result = await toolset.direct_call_tool("search_scheme", {"query": query})
    return json.loads(result)


async def _get_scheme_isin(scheme_code: int, toolset) -> str | None:
    """Uses get_scheme_meta (meta + single latest NAV entry), not
    get_scheme_nav (meta + full history) - this is called once per search
    candidate during resolution, often many times per run, and only the
    ISIN field is ever needed here. The full history is fetched separately
    (once, via _fetch_nav_history) only after resolution actually succeeds.
    """
    result = await toolset.direct_call_tool("get_scheme_meta", {"scheme_code": scheme_code})
    data = json.loads(result)
    return data["meta"].get("isin_growth")


async def _resolve_scheme_code(name: str, isin: str, toolset) -> int:
    """Deterministically resolve a fund name + ISIN to its AMFI scheme code.

    MFapi.in rejects direct ISIN lookups and its name search returns
    multiple near-identical variants (Direct/Regular plan, Growth/IDCW
    option) - so this never trusts a name match alone. It tries the
    raw name, then progressively shorter prefixes, then the same
    progression on a letter-digit-normalized name, and only accepts a
    candidate whose own meta.isin_growth matches our known ISIN exactly.
    Raises ValueError (fail loud) if no candidate matches after every
    attempt - never falls back to "the first result" or a fuzzy guess.
    """
    for candidate_name in (name, _normalize_letter_digit_spacing(name)):
        words = candidate_name.split()
        for n_words in (len(words), 4, 3, 2):
            query = " ".join(words[:n_words])
            candidates = await _search_scheme(query, toolset)
            for candidate in candidates:
                candidate_isin = await _get_scheme_isin(candidate["scheme_code"], toolset)
                if candidate_isin == isin:
                    return candidate["scheme_code"]
            if n_words <= 2:
                break
    raise ValueError(f"No MFapi.in scheme found matching name={name!r} isin={isin!r}")


def _parse_mfapi_date(date_str: str) -> date:
    """Parse a date string in DD-MM-YYYY format (MFapi.in standard)."""
    return datetime.strptime(date_str, "%d-%m-%Y").date()


async def _fetch_nav_history(scheme_code: int, toolset) -> list[dict]:
    """Fetch a scheme's full NAV history, parsed and sorted chronologically.

    MFapi.in returns newest-first; this re-sorts ascending since every
    downstream computation (CAGR, resampling, drawdown) reasons forward
    in time.
    """
    result = await toolset.direct_call_tool("get_scheme_nav", {"scheme_code": scheme_code})
    raw = json.loads(result)
    history = [
        {"date": _parse_mfapi_date(entry["date"]), "nav": float(entry["nav"])}
        for entry in raw["data"]
    ]
    return sorted(history, key=lambda h: h["date"])


def _resample_month_end(history: list[dict]) -> list[dict]:
    """Pick the last NAV entry on/before each month boundary present in the series.

    history must already be sorted ascending by date (as _fetch_nav_history
    returns it). Used for volatility and cross-fund correlation, so both
    are computed on a consistent, lower-noise monthly cadence rather than
    raw daily NAV entries.
    """
    by_month: dict[tuple[int, int], dict] = {}
    for entry in history:
        key = (entry["date"].year, entry["date"].month)
        by_month[key] = entry
    return [by_month[key] for key in sorted(by_month.keys())]


def _find_nav_on_or_before(history: list[dict], target_date) -> dict | None:
    candidates = [h for h in history if h["date"] <= target_date]
    return candidates[-1] if candidates else None


def _compute_fund_metrics(history: list[dict]) -> dict:
    """Pure computation from real NAV history - no LLM, no fabrication.

    Returns raw numbers only, mirroring the Risk Analyst's measure-not-
    judge pattern: track_record_years, cagr_1y/cagr_3y (None if the
    fund's history doesn't span that period), cagr_since_inception
    (float | None - None only when track_record_years <= 0, i.e. a
    single-NAV-entry or same-day history where CAGR is mathematically
    undefined), volatility (float | None - annualized stdev of monthly
    returns, None only when fewer than 2 monthly return observations
    exist), max_drawdown (a non-positive fraction, e.g. -0.4 for a 40%
    peak-to-trough decline; None only when history has a single entry -
    a single NAV point has no observable drawdown, which is different
    from "confirmed zero drawdown").
    """
    latest = history[-1]
    oldest = history[0]
    track_record_years = (latest["date"] - oldest["date"]).days / 365.25

    def cagr_over_years(years: float) -> float | None:
        target_date = latest["date"] - timedelta(days=int(years * 365.25))
        if target_date < oldest["date"]:
            return None
        start = _find_nav_on_or_before(history, target_date)
        if start is None or start["nav"] <= 0:
            return None
        return (latest["nav"] / start["nav"]) ** (1 / years) - 1

    cagr_since_inception = (
        (latest["nav"] / oldest["nav"]) ** (1 / track_record_years) - 1
        if track_record_years > 0
        else None
    )

    monthly = _resample_month_end(history)
    monthly_returns = [
        (monthly[i]["nav"] / monthly[i - 1]["nav"]) - 1
        for i in range(1, len(monthly))
        if monthly[i - 1]["nav"] > 0
    ]
    volatility = (
        statistics.stdev(monthly_returns) * (12 ** 0.5) if len(monthly_returns) >= 2 else None
    )

    if len(history) < 2:
        max_drawdown = None
    else:
        peak = history[0]["nav"]
        max_drawdown = 0.0
        for entry in history:
            peak = max(peak, entry["nav"])
            drawdown = (entry["nav"] - peak) / peak
            max_drawdown = min(max_drawdown, drawdown)

    return {
        "track_record_years": round(track_record_years, 2),
        "cagr_1y": cagr_over_years(1),
        "cagr_3y": cagr_over_years(3),
        "cagr_since_inception": cagr_since_inception,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
    }


def _parse_yfinance_price_history(text: str) -> list[dict]:
    """Parse yfinance_get_price_history's markdown table output.

    Confirmed live this session: this tool returns a markdown table
    (Date|Open|High|Low|Close|Volume|Dividends|Stock Splits), not JSON
    like yfinance_get_ticker_info - a real API-shape difference this
    parser exists specifically to handle.
    """
    lines = [line for line in text.strip().split("\n") if line.strip()]
    data_lines = lines[2:]  # skip header + markdown separator row
    parsed = []
    for line in data_lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        date_str = cells[0].split(" ")[0]
        parsed.append({"date": date.fromisoformat(date_str), "close": float(cells[4])})
    return parsed


async def _compute_benchmark_difference(
    nav_history: list[dict], benchmark_ticker: str | None, yfinance_toolset
) -> dict | None:
    """Compare a fund's own CAGR against its benchmark's CAGR over the same period.

    Returns None (not a fabricated number) when no verified benchmark
    ticker exists for this fund's category - a real, disclosed
    limitation, not a silently skipped check.
    """
    if benchmark_ticker is None:
        return None

    fund_metrics = _compute_fund_metrics(nav_history)
    years = fund_metrics["track_record_years"]
    if years <= 0:
        return None

    raw = await yfinance_toolset.wrapped.direct_call_tool(
        "yfinance_get_price_history", {"symbol": benchmark_ticker, "period": "max", "interval": "1mo"}
    )
    benchmark_history = _parse_yfinance_price_history(raw)

    start_date = nav_history[0]["date"]
    end_date = nav_history[-1]["date"]
    benchmark_start = _find_nav_on_or_before(
        [{"date": h["date"], "nav": h["close"]} for h in benchmark_history], start_date
    )
    benchmark_end = _find_nav_on_or_before(
        [{"date": h["date"], "nav": h["close"]} for h in benchmark_history], end_date
    )
    if benchmark_start is None or benchmark_end is None:
        return None

    benchmark_cagr = (benchmark_end["nav"] / benchmark_start["nav"]) ** (1 / years) - 1
    fund_cagr = fund_metrics["cagr_since_inception"]

    return {
        "fund_cagr": fund_cagr,
        "benchmark_cagr": benchmark_cagr,
        "difference": fund_cagr - benchmark_cagr,
    }


def _compute_redundancy_correlations(fund_histories: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    """Pairwise Pearson correlation of monthly returns across funds, keyed by ISIN.

    A statistical proxy for portfolio-holdings overlap (which MFapi.in
    doesn't expose) - two funds whose NAV moves nearly identically are
    functionally redundant even if their category labels differ, and
    two funds with the same category label can still genuinely
    diversify if their correlation is low. Symmetric: result[a][b] ==
    result[b][a]. A fund never appears as its own key in its own row.
    """
    monthly_returns: dict[str, dict[tuple[int, int], float]] = {}
    for isin, history in fund_histories.items():
        monthly = _resample_month_end(history)
        returns_by_month = {}
        for i in range(1, len(monthly)):
            if monthly[i - 1]["nav"] > 0:
                key = (monthly[i]["date"].year, monthly[i]["date"].month)
                returns_by_month[key] = (monthly[i]["nav"] / monthly[i - 1]["nav"]) - 1
        monthly_returns[isin] = returns_by_month

    result: dict[str, dict[str, float]] = {isin: {} for isin in fund_histories}
    isins = list(fund_histories.keys())
    for i, isin_a in enumerate(isins):
        for isin_b in isins[i + 1:]:
            common_months = sorted(set(monthly_returns[isin_a]) & set(monthly_returns[isin_b]))
            if len(common_months) < 2:
                continue
            series_a = [monthly_returns[isin_a][m] for m in common_months]
            series_b = [monthly_returns[isin_b][m] for m in common_months]
            correlation = statistics.correlation(series_a, series_b)
            result[isin_a][isin_b] = correlation
            result[isin_b][isin_a] = correlation
    return result


async def _resolve_scheme_by_name_and_direct_plan(scheme_name: str, toolset) -> tuple[int, str, str]:
    """Verify a proposed (not-yet-held) fund name resolves to a real,
    currently-listed Direct-plan scheme.

    Unlike _resolve_scheme_code (which verifies against a KNOWN ISIN for
    an existing holding), a new-buy candidate has no ISIN yet - this is
    the anti-hallucination safeguard instead: a fabricated or delisted
    fund name returns no real search results at all. Returns
    (scheme_code, scheme_category, scheme_name) - all three from the
    VERIFIED search hit, not the LLM's originally proposed name - for
    the first Direct-plan match found. Uses get_scheme_meta (not
    get_scheme_nav) since only identity fields are needed here, not the
    full NAV history. Raises ValueError (fail loud) if no genuine match
    exists.

    Note: this accepts the FIRST Direct-plan search hit with no check
    that it genuinely matches the proposed category - callers MUST
    independently verify the returned scheme_category corresponds to
    what was proposed (see _category_matches_taxonomy) before trusting
    this as "the same fund the LLM meant", since a fuzzy/wrong search
    match is possible for an ambiguous or generic proposed name.
    """
    candidates = await _search_scheme(scheme_name, toolset)
    for candidate in candidates:
        if "direct" in candidate["scheme_name"].lower():
            result = await toolset.direct_call_tool(
                "get_scheme_meta", {"scheme_code": candidate["scheme_code"]}
            )
            detail = json.loads(result)
            return (
                candidate["scheme_code"],
                detail["meta"]["scheme_category"],
                detail["meta"]["scheme_name"],
            )
    raise ValueError(f"No real Direct-plan scheme found matching name={scheme_name!r}")


def _category_matches_taxonomy(resolved_scheme_category: str, taxonomy_category: str) -> bool:
    """Best-effort check that a verified MFapi scheme_category genuinely
    corresponds to a curated MF_CATEGORY_TAXONOMY category.

    There is no clean, exhaustive mapping from MFapi's raw scheme_category
    strings (e.g. "Equity Scheme - Flexi Cap Fund", "Hybrid Scheme -
    Conservative Hybrid Fund", "Index Funds - Equity Funds") to this
    project's curated taxonomy - MFapi's naming isn't published as a
    stable enum. Live-verified this session across every taxonomy
    category: MFapi's scheme_category strings consistently embed the
    taxonomy category name itself as a literal substring (case
    differences aside), so a case-insensitive substring match is a
    pragmatic, good-enough proxy - not a guaranteed-clean mapping. This
    intentionally fails loud in the caller's favor: if the substring
    isn't present, treat it as a mismatch (reject) rather than assume a
    match, since a false rejection just drops one candidate while a
    false acceptance would attach the wrong fund's real data to a
    candidate shown under a different name (the exact bug this guards
    against).
    """
    return taxonomy_category.lower() in resolved_scheme_category.lower()


def _compute_mf_category_gaps(holdings_by_bucket: dict[str, dict[str, list[dict]]]) -> list[str]:
    """Live-compute curated MF categories with zero coverage in real IN mutual fund holdings.

    Mirrors _compute_us_gaps/_compute_india_equity_gaps: any category
    string not in MF_CATEGORY_TO_TAXONOMY raises - fail loud rather
    than silently miscounting.
    """
    covered: set[str] = set()
    for holding in holdings_by_bucket.get("mutual_fund", {}).get("IN", []):
        category = holding["sector"]
        covered.add(MF_CATEGORY_TO_TAXONOMY[category])
    return [category for category in MF_CATEGORY_TAXONOMY if category not in covered]
