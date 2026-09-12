"""Shared helpers for the Mutual Fund Specialist nodes
(mutual_fund_holdings_analyzer.py, mutual_fund_specialist.py).

Curated taxonomy, real-category mapping, and live-verified benchmark
ticker tables. Scheme resolution, NAV fetching, and metrics computation
are added in later tasks in this same file.
"""

from __future__ import annotations

import json
import re

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
    result = await toolset.direct_call_tool("get_scheme_nav", {"scheme_code": scheme_code})
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
