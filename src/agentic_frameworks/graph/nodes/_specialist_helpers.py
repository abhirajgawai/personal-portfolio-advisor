"""Cap-enforcement logic shared across all new-buy specialist nodes.

Moved out of us_equity_etf_specialist.py once the India Equity
Specialist needed the identical logic - kept in one place so the two
specialists' cap behavior can never silently drift apart.
"""

from __future__ import annotations

from agentic_frameworks.graph.schemas import ProposedCandidate

MAX_CANDIDATES_PER_SECTOR = 3
MAX_CANDIDATES_TOTAL = 5


def _apply_candidate_cap(candidates: list[ProposedCandidate]) -> list[ProposedCandidate]:
    """Enforce the hard cap in code - never trust the model's own count."""
    per_sector_counts: dict[str, int] = {}
    capped: list[ProposedCandidate] = []
    for candidate in candidates:
        per_sector_counts.setdefault(candidate.sector, 0)
        if per_sector_counts[candidate.sector] < MAX_CANDIDATES_PER_SECTOR:
            capped.append(candidate)
            per_sector_counts[candidate.sector] += 1
    return capped[:MAX_CANDIDATES_TOTAL]
