"""Tests for db_lock.py - the OS advisory lock that makes concurrent
writers to agent_memory.db impossible (design spec Architecture 3,
Testing items 1-2).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_probe(*args: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "tests._db_lock_probe_helper", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_lock_is_reentrant_within_one_process(tmp_path):
    """Design spec Testing item 2: get_checkpointer() and
    get_memory_store() open together (as every real run does) without
    deadlocking - modeled here directly against advisor_memory_lock,
    the mechanism both of those functions share."""
    from agentic_frameworks.db_lock import advisor_memory_lock

    db_path = tmp_path / "reentrant.db"
    with advisor_memory_lock(db_path):
        with advisor_memory_lock(db_path):
            pass  # nested acquire must not deadlock or raise.

    # Released after both context managers exit - proven by a fresh
    # process acquiring it immediately (well within the timeout).
    result = _run_probe("acquire", str(db_path), "0")
    assert "ACQUIRED" in result.stdout, result.stderr


def test_lock_blocks_a_second_process_from_opening_the_real_checkpointer():
    """Design spec Testing item 1 - the test that actually retires Phase
    7's flagged SQLite-concurrency risk: hold the lock on the REAL
    AGENT_MEMORY_DB_PATH in this test process, then a genuine second OS
    process calling get_checkpointer() (which always targets that same
    real path) must raise AdvisorMemoryLockedError and never open the
    database - then, once released, the same subprocess call succeeds.
    """
    from agentic_frameworks.db_lock import advisor_memory_lock
    from agentic_frameworks.graph.checkpointer import AGENT_MEMORY_DB_PATH

    with advisor_memory_lock(AGENT_MEMORY_DB_PATH):
        blocked = _run_probe("checkpointer", timeout=30.0)
        assert "FAILED:AdvisorMemoryLockedError" in blocked.stdout, (blocked.stdout, blocked.stderr)

    # Lock released - the same subprocess call now succeeds.
    unblocked = _run_probe("checkpointer", timeout=30.0)
    assert "ACQUIRED" in unblocked.stdout, (unblocked.stdout, unblocked.stderr)


def test_lock_is_released_when_the_holding_process_is_killed(tmp_path):
    """The property a PID-file design would fail (design spec Testing
    item 2 / "Why an advisory OS lock and not a sentinel/PID file"): a
    process that dies while holding the lock (crash, not clean exit)
    does not leave a stale lock behind - the OS releases it."""
    from agentic_frameworks.db_lock import advisor_memory_lock

    db_path = tmp_path / "crash.db"

    proc = subprocess.Popen(
        [sys.executable, "-m", "tests._db_lock_probe_helper", "acquire", str(db_path), "30"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10.0
        line = ""
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if "ACQUIRED" in line:
                break
        assert "ACQUIRED" in line, "subprocess never reported holding the lock"

        proc.kill()
        proc.wait(timeout=10.0)

        # The OS released the lock the instant the process died - a
        # fresh acquire in this process succeeds well within the
        # bounded backoff, not because anyone cleaned up a lock file.
        with advisor_memory_lock(db_path):
            pass
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10.0)


def test_acquire_raises_advisor_memory_locked_error_after_bounded_backoff(monkeypatch, tmp_path):
    """Fail loud after a bounded backoff, never a silent skip or 'run
    anyway' (design spec Error Handling).

    Genuine contention has to be simulated with two independent
    `_RefcountedLock` objects on the same lock-file path (bypassing the
    process-wide memoization in `_lock_for`) rather than two nested
    `advisor_memory_lock()` calls - those are the *same* memoized lock
    object and would just increment its refcount (the re-entrant
    behaviour proven by the test above), never actually contend.
    """
    import agentic_frameworks.db_lock as db_lock_module

    monkeypatch.setattr(db_lock_module, "_ACQUIRE_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(db_lock_module, "_ACQUIRE_RETRY_INTERVAL_SECONDS", 0.1)

    lock_file_path = tmp_path / "contended.db.lock"
    holder = db_lock_module._RefcountedLock(lock_file_path)
    contender = db_lock_module._RefcountedLock(lock_file_path)

    holder.acquire()
    try:
        with pytest.raises(db_lock_module.AdvisorMemoryLockedError):
            contender.acquire()
    finally:
        holder.release()
