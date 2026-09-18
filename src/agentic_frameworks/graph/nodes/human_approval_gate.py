"""The human-approval-gate node: a real LangGraph interrupt() that
suspends the run whenever the finished report contains anything
consequential (a new-buy suggestion, or a Sell/Reduce/AddTo on an
existing holding).

No LLM, no tools, no store, no network - cheap by construction (design
spec Architecture point 5). Nothing in this system can construct the
resume value itself: see the design spec's "Why nothing in this system
can auto-approve it" and this repo's binding-prohibition grep test
(test_no_self_approval.py) - no module under src/agentic_frameworks/
may construct LangGraph's resume-command primitive.
"""

from __future__ import annotations

from langgraph.types import interrupt

from agentic_frameworks.graph.schemas import HumanApproval
from agentic_frameworks.graph.state import ApprovalState, GraphState

_ACTIONS_REQUIRING_APPROVAL = {"Sell", "Reduce", "AddTo"}


def _requires_human_approval(report: dict) -> bool:
    """Pure, deterministic predicate over the finished report - never the
    model's own opinion about whether approval is needed (design spec
    Architecture point 5). AddTo is included because it commits fresh
    capital, same as a new buy (design spec Decision 2).
    """
    if report["new_buy_suggestions"]:
        return True
    return any(
        guidance["action"] in _ACTIONS_REQUIRING_APPROVAL
        for guidance in report["existing_holdings_guidance"]
    )


def _build_decision_request(report: dict) -> dict:
    """A compact, JSON-serializable decision-request payload (must be, for
    checkpointing - model_dump()'d dicts only, never a model instance) -
    the new-buy suggestions, the sell/reduce/add-to items, the assessment,
    and the coverage note, per design spec Data Flow step 2."""
    consequential_holdings_guidance = [
        guidance
        for guidance in report["existing_holdings_guidance"]
        if guidance["action"] in _ACTIONS_REQUIRING_APPROVAL
    ]
    return {
        "portfolio_assessment": report["portfolio_assessment"],
        "coverage_note": report["coverage"]["coverage_note"],
        "new_buy_suggestions": report["new_buy_suggestions"],
        "consequential_holdings_guidance": consequential_holdings_guidance,
    }


async def human_approval_gate_node(state: GraphState, config) -> ApprovalState:
    """Suspend the run pending human approval of `state["recommendation"]`.

    Fail-loud checkpointer guard: without a real checkpointer,
    interrupt() silently produces an unresumable pause rather than
    raising (design spec Background, live-verified) - this node would
    look like it ran while actually being a dead end, so this checks and
    raises rather than letting that happen quietly.

    `config["configurable"]["__pregel_checkpointer"]` is how a node sees
    the live checkpointer by signature injection when the graph was
    compiled with one, and None when it wasn't - the same LangGraph
    injection mechanism Phase 7 relied on for `store`.

    HumanApproval.model_validate(...) fails loud on anything malformed,
    missing, or extra - no lenient parsing, no default-approve path
    (design spec Error Handling / "why nothing can auto-approve", reason 2).
    """
    if config["configurable"].get("__pregel_checkpointer") is None:
        raise ValueError(
            "human_approval_gate_node requires the graph to be compiled with a real "
            "checkpointer (compile with checkpointer=... - see build_compiled_graph in "
            "graph.py); without one, interrupt() would silently produce an unresumable pause."
        )

    report = state["recommendation"]
    decision_request = _build_decision_request(report)
    resume_value = interrupt(decision_request)
    approval = HumanApproval.model_validate(resume_value)
    return {"approval": approval.model_dump()}
