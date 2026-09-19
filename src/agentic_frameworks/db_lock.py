"""Process-level OS advisory lock making concurrent writers to
agent_memory.db impossible, structurally rather than by convention (see
docs/superpowers/specs/2026-09-19-temporal-design.md, Architecture 3).

Stdlib only - `msvcrt` on win32, `fcntl` on posix, selected once at
import. Chosen over a sentinel/PID file or a `BEGIN EXCLUSIVE`
transaction precisely because it is released by the OS the instant the
holding process dies (crash included), so a stale lock requiring manual
cleanup can never exist - see the design spec's "Why an advisory OS
lock and not a sentinel/PID file or a SQLite lock row".

Re-entrant within one process by refcount: one process opening both
`get_checkpointer()` and `get_memory_store()` (as every real run does,
since they share `AGENT_MEMORY_DB_PATH`) acquires the real OS lock once
and releases it when the last holder exits. A *second* process trying
the same path fails to acquire and, after a short bounded backoff, gets
`AdvisorMemoryLockedError` - never a silent skip, never "run anyway"
(the standing fail-loud rule).

Scoped to whatever `db_path` a caller actually passes - `checkpointer.py`
and `store.py` are the only two call sites in this codebase, and they
only ever pass the real `AGENT_MEMORY_DB_PATH`. Tests that build their
own `AsyncSqliteSaver`/`AsyncSqliteStore` directly against a `tmp_path`
file never call this module at all, so they are unaffected by design,
not by a special case here.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

_ACQUIRE_RETRY_INTERVAL_SECONDS = 0.25
_ACQUIRE_TIMEOUT_SECONDS = 5.0


class AdvisorMemoryLockedError(RuntimeError):
    """Raised when the OS advisory lock on `agent_memory.db` can't be
    acquired within the bounded backoff - fail loud, never a silent skip
    or "run anyway" (design spec Error Handling). Temporal's retry
    policy treats this as retryable: an overlapping manual run will
    eventually end and release the lock (design spec, Activity timeouts
    and retries)."""


def _lock_region(file: IO[bytes]) -> None:
    """Lock byte 0 of `file`, non-blocking. Raises `OSError` on
    contention (both `msvcrt.locking` and `fcntl.flock` do)."""
    file.seek(0)
    if sys.platform == "win32":
        msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_region(file: IO[bytes]) -> None:
    file.seek(0)
    if sys.platform == "win32":
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)


class _RefcountedLock:
    """One instance per distinct lock-file path. Guarded by its own
    `threading.Lock` (not just relying on the GIL) since this project's
    async nodes can in principle be driven from different threads (an
    asyncio event loop plus, e.g., a Temporal worker's own thread)."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._guard = threading.Lock()
        self._refcount = 0
        self._file: IO[bytes] | None = None

    def acquire(self) -> None:
        with self._guard:
            if self._refcount > 0:
                # Already held by this process - re-entrant, no second
                # OS-level lock attempt (which could self-deadlock).
                self._refcount += 1
                return

            self._path.parent.mkdir(parents=True, exist_ok=True)
            file = open(self._path, "a+b")
            if file.seek(0, 2) == 0:
                # msvcrt.locking needs at least one real byte to lock.
                file.write(b"\0")
                file.flush()

            deadline = time.monotonic() + _ACQUIRE_TIMEOUT_SECONDS
            while True:
                try:
                    _lock_region(file)
                except OSError:
                    if time.monotonic() >= deadline:
                        file.close()
                        raise AdvisorMemoryLockedError(
                            f"Could not acquire the advisor memory lock on {self._path} "
                            f"within {_ACQUIRE_TIMEOUT_SECONDS}s - another process is "
                            "already holding it (a scheduled run and a manual run "
                            "overlapped, or a prior run is stuck). Refusing to proceed: "
                            "two concurrent writers to agent_memory.db are never allowed."
                        ) from None
                    time.sleep(_ACQUIRE_RETRY_INTERVAL_SECONDS)
                else:
                    self._file = file
                    self._refcount = 1
                    return

    def release(self) -> None:
        with self._guard:
            if self._refcount == 0:
                return
            self._refcount -= 1
            if self._refcount == 0 and self._file is not None:
                _unlock_region(self._file)
                self._file.close()
                self._file = None


_locks_guard = threading.Lock()
_locks: dict[Path, _RefcountedLock] = {}


def _lock_for(db_path: Path) -> _RefcountedLock:
    """One `_RefcountedLock` per resolved db path, memoized process-wide
    so every caller locking the same real file shares the same refcount
    - the property that makes checkpointer.py + store.py opening
    together re-entrant instead of racing each other."""
    resolved = db_path.resolve()
    lock_file_path = resolved.with_name(resolved.name + ".lock")
    with _locks_guard:
        lock = _locks.get(resolved)
        if lock is None:
            lock = _RefcountedLock(lock_file_path)
            _locks[resolved] = lock
        return lock


@contextmanager
def advisor_memory_lock(db_path: Path) -> Iterator[None]:
    """Re-entrant, process-wide OS advisory lock scoped to `db_path`.

    Intended to be held for the whole time `db_path` is open - callers
    acquire before opening the database and release only after closing
    it, so the lock's lifetime always covers the database connection's.
    """
    lock = _lock_for(db_path)
    lock.acquire()
    try:
        yield
    finally:
        lock.release()
