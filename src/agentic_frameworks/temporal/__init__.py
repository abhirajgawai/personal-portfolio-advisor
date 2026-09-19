"""The Temporal durable daily monitor: a Workflow (pure orchestration,
no I/O) plus three Activities that give the advisor graph a durable,
crash/restart-safe daily trigger.

See docs/superpowers/specs/2026-09-19-temporal-design.md for the full
design, including the "Decisions" section that resolved this phase's
open questions (notification is local-only, no live Schedule gets
activated by this codebase, the OS lock lives inside
graph/checkpointer.py and graph/store.py, not here).
"""

from __future__ import annotations
