"""Shared helpers for the Mutual Fund Specialist nodes
(mutual_fund_holdings_analyzer.py, mutual_fund_specialist.py).

Curated taxonomy, real-category mapping, and live-verified benchmark
ticker tables. Scheme resolution, NAV fetching, and metrics computation
are added in later tasks in this same file.
"""

from __future__ import annotations

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
