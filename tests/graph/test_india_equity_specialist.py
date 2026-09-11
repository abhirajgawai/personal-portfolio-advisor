"""Tests for the India Equity Specialist node and its stages."""

from __future__ import annotations


async def test_compute_india_equity_gaps_finds_energy_and_real_estate_given_real_holdings():
    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _compute_india_equity_gaps,
    )
    from agentic_frameworks.graph.nodes.research import research_node

    research_result = await research_node({})

    gaps = _compute_india_equity_gaps(research_result["holdings_by_bucket"])

    assert set(gaps) == {"Energy", "Real Estate"}


def test_compute_india_equity_gaps_raises_on_unmapped_sector():
    import pytest

    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _compute_india_equity_gaps,
    )

    holdings_by_bucket = {
        "equity": {
            "IN": [{"ticker": "ZZZ.NS", "name": "Fictional Co", "sector": "Not A Real Sector"}],
        }
    }

    with pytest.raises(KeyError):
        _compute_india_equity_gaps(holdings_by_bucket)


def test_compute_india_equity_gaps_skips_null_sector_rows():
    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _compute_india_equity_gaps,
    )

    holdings_by_bucket = {
        "equity": {
            "IN": [{"ticker": "MANUAL_TOTAL", "name": "Manual Total", "sector": None}],
        }
    }

    gaps = _compute_india_equity_gaps(holdings_by_bucket)

    assert len(gaps) == 11
