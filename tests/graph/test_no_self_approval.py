"""The binding prohibition (design spec's "why nothing can auto-approve
it", reason 3): no module under `src/agentic_frameworks/` may construct
`Command(resume=...)`. A structural guarantee enforced by this test, not
a comment - resumption must only ever be invoked by a caller outside the
graph (a human, or the Studio/Temporal integration layers this project
hasn't built yet), never by any node or helper inside the fleet itself.

Enforced via an AST walk for any reference to the name `Command` at all
(import, attribute access, or bare name), not a literal "Command(resume"
substring search. A substring check is defeated by anything a formatter
or a refactor could produce without changing behavior - a multi-line
call (`Command(\\n    resume=payload\\n)`), reordered keywords
(`Command(goto=x, resume=payload)`), building it via
`cmd = Command(); cmd.resume = payload`, or `getattr(langgraph.types,
"Command")(...)`. Banning the name entirely is strictly stronger and
still passes today, since nothing under src/ has a legitimate reason to
reference `Command` - only external callers (a human, Studio, Temporal)
resume a pending graph.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "agentic_frameworks"


def _references_command(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(
            alias.name == "Command" for alias in node.names
        ):
            return True
        if isinstance(node, ast.Name) and node.id == "Command":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "Command":
            return True
    return False


def test_no_module_under_src_references_command():
    offending_files = []
    for path in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if _references_command(tree):
            offending_files.append(str(path.relative_to(SRC_ROOT)))

    assert offending_files == [], (
        "No module under src/agentic_frameworks/ may reference langgraph.types.Command "
        f"in any form - resumption must only ever come from outside the graph. Found in: {offending_files}"
    )
