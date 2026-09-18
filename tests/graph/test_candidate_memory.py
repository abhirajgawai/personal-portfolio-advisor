"""Tests for the shared cross-run rejection-memory helpers
(nodes/_candidate_memory.py) - real AsyncSqliteStore on a temp file,
no mocking, per the design spec's Testing section.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


def _make_rejected_candidate(**overrides):
    from agentic_frameworks.graph.schemas import RejectedCandidate

    defaults = dict(
        identifier="XLV",
        identifier_kind="ticker",
        specialist="us_equity",
        display_name="Health Care Select Sector SPDR Fund",
        group="Health Care",
        action="Hold",
        confidence=0.6,
        risk_flags=["thin_moat"],
        reasoning="Fairly valued but no clear edge over peers.",
        rejected_at=datetime.now(UTC).isoformat(),
        thread_id=None,
        schema_version=1,
    )
    defaults.update(overrides)
    return RejectedCandidate(**defaults)


def test_normalize_scheme_name_over_real_held_fund_name_variants():
    """Real proposed-name variants of an actual held fund (Parag Parikh
    Flexi Cap Fund, held in the real portfolio) - no invented scheme
    names, per the no-fabrication rule."""
    from agentic_frameworks.graph.nodes._candidate_memory import _normalize_scheme_name

    assert _normalize_scheme_name("Parag Parikh Flexi Cap Fund - Direct Plan - Growth") == _normalize_scheme_name(
        "Parag Parikh Flexi Cap Fund Direct Growth"
    )
    assert _normalize_scheme_name("PARAG PARIKH FLEXI CAP FUND") == "parag parikh flexi cap"


def test_normalize_scheme_name_collapses_whitespace_and_punctuation():
    from agentic_frameworks.graph.nodes._candidate_memory import _normalize_scheme_name

    assert _normalize_scheme_name("Nippon  India -- Large   Cap") == "nippon india large cap"


def test_is_fresh_within_window():
    from agentic_frameworks.graph.nodes._candidate_memory import REJECTION_MEMORY_DAYS, _is_fresh

    now = datetime.now(UTC)
    record = _make_rejected_candidate(
        rejected_at=(now - timedelta(days=REJECTION_MEMORY_DAYS - 1)).isoformat()
    )

    assert _is_fresh(record, now=now) is True


def test_is_fresh_just_outside_window():
    from agentic_frameworks.graph.nodes._candidate_memory import REJECTION_MEMORY_DAYS, _is_fresh

    now = datetime.now(UTC)
    record = _make_rejected_candidate(
        rejected_at=(now - timedelta(days=REJECTION_MEMORY_DAYS + 1)).isoformat()
    )

    assert _is_fresh(record, now=now) is False


async def test_load_recent_rejections_round_trips_and_filters_stale(memory_store):
    from agentic_frameworks.graph.nodes._candidate_memory import _load_recent_rejections, _record_rejection

    fresh = _make_rejected_candidate(identifier="XLV")
    stale = _make_rejected_candidate(
        identifier="JNJ",
        rejected_at=(datetime.now(UTC) - timedelta(days=200)).isoformat(),
    )
    namespace = ("rejected_candidates", "us_equity")

    await _record_rejection(memory_store, namespace, fresh)
    await _record_rejection(memory_store, namespace, stale)

    rejections = await _load_recent_rejections(memory_store, namespace)

    assert "XLV" in rejections
    assert "JNJ" not in rejections
    assert rejections["XLV"].display_name == fresh.display_name


def test_filter_previously_rejected_drops_rejected_candidates():
    from agentic_frameworks.graph.nodes._candidate_memory import _filter_previously_rejected
    from agentic_frameworks.graph.schemas import ProposedCandidate

    candidates = [
        ProposedCandidate(ticker="XLV", sector="Health Care"),
        ProposedCandidate(ticker="UNH", sector="Health Care"),
    ]
    rejections = {"XLV": _make_rejected_candidate(identifier="XLV")}

    filtered = _filter_previously_rejected(candidates, rejections, key=lambda c: c.ticker.upper())

    assert [c.ticker for c in filtered] == ["UNH"]


def test_cap_interaction_rejected_candidates_consume_no_slot():
    """Behavioural pin (design spec Testing #5): given 3 previously-
    rejected + 5 fresh candidates for one sector, the cap must be
    filled entirely with fresh candidates - rejection filtering runs
    before the cap, so rejected candidates never occupy a slot.

    This is a pure identifier-matching test of the helpers only (no
    real market-data lookup happens here), but per CLAUDE.md's
    no-fabrication rule it still uses real, currently-traded Health
    Care tickers rather than invented placeholders - nothing about
    this test requires a fake symbol."""
    from agentic_frameworks.graph.nodes._candidate_memory import _filter_previously_rejected
    from agentic_frameworks.graph.nodes._specialist_helpers import _apply_candidate_cap
    from agentic_frameworks.graph.schemas import ProposedCandidate

    rejected_tickers = ["XLV", "UNH", "JNJ"]
    fresh_tickers = ["PFE", "ABBV", "MRK", "LLY", "ABT"]
    candidates = [
        ProposedCandidate(ticker=t, sector="Health Care") for t in rejected_tickers + fresh_tickers
    ]
    rejections = {t: _make_rejected_candidate(identifier=t) for t in rejected_tickers}

    not_rejected = _filter_previously_rejected(candidates, rejections, key=lambda c: c.ticker.upper())
    capped = _apply_candidate_cap(not_rejected)

    capped_tickers = {c.ticker for c in capped}
    assert capped_tickers == {"PFE", "ABBV", "MRK"}
    assert capped_tickers.isdisjoint(rejected_tickers)


async def test_record_rejection_never_written_for_buy_is_a_node_contract_not_a_helper_one():
    """_record_rejection itself has no action guard - RejectedCandidate's
    own schema (action: Literal["Hold", "Reduce", "Sell"]) is what makes
    "Buy is never stored" a structural guarantee, not a convention the
    caller might forget."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _make_rejected_candidate(action="Buy")


async def test_resolved_scheme_cache_round_trips(memory_store):
    from agentic_frameworks.graph.nodes._candidate_memory import (
        _cache_resolved_scheme,
        _load_resolved_scheme,
    )

    assert await _load_resolved_scheme(memory_store, "parag parikh flexi cap") is None

    await _cache_resolved_scheme(
        memory_store,
        "parag parikh flexi cap",
        122639,
        "Equity Scheme - Flexi Cap Fund",
        "Parag Parikh Flexi Cap Fund - Direct Plan - Growth",
    )

    cached = await _load_resolved_scheme(memory_store, "parag parikh flexi cap")

    assert cached == {
        "scheme_code": 122639,
        "category": "Equity Scheme - Flexi Cap Fund",
        "scheme_name": "Parag Parikh Flexi Cap Fund - Direct Plan - Growth",
    }
