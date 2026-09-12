"""Cap-enforcement logic shared across all new-buy specialist nodes.

Moved out of us_equity_etf_specialist.py once the India Equity
Specialist needed the identical logic - kept in one place so the two
specialists' cap behavior can never silently drift apart.
"""

from __future__ import annotations

from agentic_frameworks.graph.schemas import ProposedCandidate

MAX_CANDIDATES_PER_SECTOR = 3
MAX_CANDIDATES_TOTAL = 5


def _apply_candidate_cap(candidates: list[ProposedCandidate], key=lambda c: c.sector) -> list[ProposedCandidate]:
    """Enforce the hard cap in code - never trust the model's own count.

    key: extracts the grouping value from a candidate (default: .sector,
    used by the US/India equity specialists). The Mutual Fund Specialist
    passes key=lambda c: c.category, since its candidates group by fund
    category rather than GICS sector - the cap logic itself is identical
    either way, only the grouping key differs.
    """
    per_group_counts: dict[str, int] = {}
    capped: list[ProposedCandidate] = []
    for candidate in candidates:
        group = key(candidate)
        per_group_counts.setdefault(group, 0)
        if per_group_counts[group] < MAX_CANDIDATES_PER_SECTOR:
            capped.append(candidate)
            per_group_counts[group] += 1
    return capped[:MAX_CANDIDATES_TOTAL]


def _filter_to_gap_sectors(candidates: list[ProposedCandidate], gap_sectors: list[str], key=lambda c: c.sector) -> list[ProposedCandidate]:
    """Drop any candidate whose group isn't one of this run's live-computed gaps.

    Necessary because ProposedCandidate.sector (and ProposedFundCandidate.category)
    validate against a full taxonomy, not just this run's actual gaps - a
    model can legally propose a candidate for a group the portfolio
    already covers. That candidate must never reach a real verification
    or Conviction Gate call. key: see _apply_candidate_cap.
    """
    gap_set = set(gap_sectors)
    return [candidate for candidate in candidates if key(candidate) in gap_set]
