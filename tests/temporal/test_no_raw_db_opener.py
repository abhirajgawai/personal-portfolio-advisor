"""Design spec Testing item 3: no module under src/ opens
AGENT_MEMORY_DB_PATH outside checkpointer.py / store.py - those are the
only two places the OS lock (db_lock.py) is taken, so a raw opener
anywhere else would bypass it silently.

Same AST-walk style as test_no_self_approval.py, for the same reason: a
substring search is defeated by a reformatted import; walking for any
reference to the name `AGENT_MEMORY_DB_PATH` is strictly stronger.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "agentic_frameworks"
ALLOWED_FILES = {"graph/checkpointer.py", "graph/store.py"}


def _references_agent_memory_db_path(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "AGENT_MEMORY_DB_PATH":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "AGENT_MEMORY_DB_PATH":
            return True
    return False


def test_no_module_under_src_opens_the_real_db_path_outside_checkpointer_and_store():
    offending_files = []
    for path in SRC_ROOT.rglob("*.py"):
        relative = str(path.relative_to(SRC_ROOT)).replace("\\", "/")
        if relative in ALLOWED_FILES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if _references_agent_memory_db_path(tree):
            offending_files.append(relative)

    assert offending_files == [], (
        "Only graph/checkpointer.py and graph/store.py may reference AGENT_MEMORY_DB_PATH - "
        f"every other path into agent_memory.db must go through get_checkpointer()/get_memory_store() "
        f"so the OS lock in db_lock.py can never be bypassed. Found in: {offending_files}"
    )
