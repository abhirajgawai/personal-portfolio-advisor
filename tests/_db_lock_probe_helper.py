"""Helper for db_lock cross-process tests (tests/test_db_lock.py).

Run directly: `python -m tests._db_lock_probe_helper acquire <path> <seconds>`
holds `advisor_memory_lock(<path>)` for `<seconds>`, printing "ACQUIRED"
as soon as it gets the lock and "RELEASED" right before exiting.

`python -m tests._db_lock_probe_helper checkpointer` opens the REAL
production `get_checkpointer()` (no path argument - it always targets
`AGENT_MEMORY_DB_PATH`) and prints "ACQUIRED" on success or
"FAILED:<ExceptionType>" if `get_checkpointer()` raises.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

from agentic_frameworks.db_lock import advisor_memory_lock


def _acquire_and_hold(path_str: str, seconds: str) -> None:
    with advisor_memory_lock(Path(path_str)):
        print("ACQUIRED", flush=True)
        time.sleep(float(seconds))
    print("RELEASED", flush=True)


async def _open_real_checkpointer() -> None:
    from agentic_frameworks.graph.checkpointer import get_checkpointer

    try:
        async with get_checkpointer():
            print("ACQUIRED", flush=True)
    except Exception as exc:  # noqa: BLE001 - reporting to the parent test process, not swallowing.
        print(f"FAILED:{type(exc).__name__}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "acquire":
        _acquire_and_hold(sys.argv[2], sys.argv[3])
    elif mode == "checkpointer":
        asyncio.run(_open_real_checkpointer())
    else:
        raise SystemExit(f"unknown mode: {mode}")
