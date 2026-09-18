"""Cross-run rejection memory and fund-resolution caching, shared by the
three new-buy specialist nodes (us_equity_etf_specialist.py,
india_equity_specialist.py, mutual_fund_specialist.py).

See docs/superpowers/specs/2026-09-18-langgraph-store-design.md for the
full design and its "Decisions" section (90-day window, generator
exclusion hints in, ISIN->scheme_code caching in, no per-portfolio-
version namespacing).

Small, single-purpose helpers only - namespace constants, identity
normalization, and the three store operations (load/filter/record).
No LLM/tool calls live here; that keeps this module trivially testable
against a real AsyncSqliteStore with no mocking.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

from langgraph.store.base import BaseStore

from agentic_frameworks.graph.schemas import RejectedCandidate

REJECTION_MEMORY_DAYS = 90
MAX_EXCLUSION_HINTS = 20

REJECTED_US_EQUITY_NS = ("rejected_candidates", "us_equity")
REJECTED_INDIA_EQUITY_NS = ("rejected_candidates", "india_equity")
REJECTED_MUTUAL_FUND_NS = ("rejected_candidates", "mutual_fund")
REJECTED_MUTUAL_FUND_BY_NAME_NS = ("rejected_candidates", "mutual_fund", "by_name")
RESOLVED_SCHEMES_MUTUAL_FUND_NS = ("resolved_schemes", "mutual_fund")

_SCHEME_NAME_NOISE_WORDS = {"fund", "direct", "plan", "growth", "scheme"}


def _normalize_scheme_name(name: str) -> str:
    """Case-fold, collapse whitespace, strip punctuation, and drop the
    noise words fund/direct/plan/growth/scheme - a deliberately
    conservative normalization (see design spec): a normalization miss
    just costs one free resolution call (caught by the authoritative
    scheme_code filter downstream), while an over-aggressive one could
    collide two genuinely different funds.
    """
    folded = name.casefold()
    stripped = re.sub(r"[^\w\s]", " ", folded)
    words = [word for word in stripped.split() if word not in _SCHEME_NAME_NOISE_WORDS]
    return " ".join(words)


def _is_fresh(record: RejectedCandidate, *, now: datetime | None = None) -> bool:
    """A rejection suppresses re-evaluation only while it's within
    REJECTION_MEMORY_DAYS of `rejected_at`. `now` is an injection point
    for tests - time itself is never faked, but callers can pin "now"
    to make a freshness boundary deterministic to test.
    """
    now = now or datetime.now(UTC)
    rejected_at = datetime.fromisoformat(record.rejected_at)
    return now - rejected_at <= timedelta(days=REJECTION_MEMORY_DAYS)


async def _load_recent_rejections(
    store: BaseStore, namespace: tuple[str, ...]
) -> dict[str, RejectedCandidate]:
    """One prefix search over `namespace`, freshness-filtered.

    Returns {store_key: RejectedCandidate}, so callers get O(1)
    membership checks. A stored record that fails RejectedCandidate
    validation raises (fail loud, per CLAUDE.md) - it is not skipped;
    schema_version exists to make that diagnosable.

    limit=1000 rather than the store default of 10: every rejection in
    the window is needed for filtering, not just a page of them - this
    node calls this once per run, so the cost is one local SQLite read.

    `store.asearch`'s first argument is a namespace PREFIX, not an
    exact match - e.g. searching REJECTED_MUTUAL_FUND_NS =
    ("rejected_candidates", "mutual_fund") also returns everything
    stored under the nested REJECTED_MUTUAL_FUND_BY_NAME_NS =
    ("rejected_candidates", "mutual_fund", "by_name"). Results are
    therefore filtered to items whose namespace is an EXACT match
    before processing, so a call for one namespace never pulls in
    records that actually live in a nested one.
    """
    items = await store.asearch(namespace, limit=1000)
    rejections: dict[str, RejectedCandidate] = {}
    for item in items:
        if item.namespace != namespace:
            continue
        record = RejectedCandidate.model_validate(item.value)
        if _is_fresh(record):
            rejections[item.key] = record
    return rejections


def _filter_previously_rejected(candidates: list, rejections: dict[str, RejectedCandidate], key) -> list:
    """Drop any candidate whose identity key is already a fresh rejection.

    Pure function, mirrors `_specialist_helpers._filter_to_gap_sectors`'s
    `key=` shape exactly: `key(candidate)` extracts the same identity
    string used as the rejections dict's keys (upper-cased ticker,
    scheme_code string, or normalized scheme name, depending on caller).
    """
    return [candidate for candidate in candidates if key(candidate) not in rejections]


async def _record_rejection(store: BaseStore, namespace: tuple[str, ...], record: RejectedCandidate) -> None:
    """Persist a non-Buy verdict, keyed by `record.identifier`.

    A TTL is written as belt-and-braces garbage collection (minutes,
    per AsyncSqliteStore's TTLConfig) - nothing correctness-relevant
    depends on it, since `_is_fresh` enforces staleness on read
    regardless (the installed version's `supports_ttl` MRO quirk means
    TTL alone would not filter reads - see design spec Background).
    """
    await store.aput(
        namespace,
        record.identifier,
        record.model_dump(),
        ttl=REJECTION_MEMORY_DAYS * 24 * 60,
    )


async def _load_resolved_scheme(store: BaseStore, normalized_name: str) -> dict | None:
    """Read a cached (scheme_code, category, scheme_name) resolution for
    a normalized proposed fund name, or None on a cache miss.

    A fund's real identity (scheme_code/category/name) doesn't change
    once resolved, so this cache carries no TTL/staleness concern -
    unlike rejection memory, a hit here is never "wrong", only
    "not yet known".
    """
    item = await store.aget(RESOLVED_SCHEMES_MUTUAL_FUND_NS, normalized_name)
    return item.value if item is not None else None


async def _cache_resolved_scheme(
    store: BaseStore, normalized_name: str, scheme_code: int, category: str, scheme_name: str
) -> None:
    """Write a fund-name resolution to the cache, keyed by normalized name.

    No TTL: see `_load_resolved_scheme`.
    """
    await store.aput(
        RESOLVED_SCHEMES_MUTUAL_FUND_NS,
        normalized_name,
        {"scheme_code": scheme_code, "category": category, "scheme_name": scheme_name},
    )
