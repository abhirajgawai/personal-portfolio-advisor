"""Helper for the cross-process resume test (test_human_approval_gate.py).

A minimal two-node graph - a stub node that writes a fixed, real-shaped
`AdvisorReport` dict with a Sell action (so `_requires_human_approval`
is True) straight into state, then the REAL production
`human_approval_gate_node` - compiled with a real `AsyncSqliteSaver`
pointed at a file path given on argv.

Deliberately avoids any LLM/network call: the point of the cross-process
probe (design spec Background) is proving `interrupt()`/`Command(resume=
...)` survive a real process boundary, not re-exercising the specialist
pipeline (that's covered by the real fan-in test in test_graph.py and the
same-process resume test in test_human_approval_gate.py). The stub node
uses a real, currently-held India mutual fund ISIN (INF179K01WN9 - see
test_synthesizer.py's REAL_INDEX_FUND_ISIN) as the subject of its Sell
guidance, not a fabricated identifier.

Run directly (`python -m tests.graph._cross_process_resume_helper
<db_path> <thread_id>`) as process 2: opens the same checkpoint file
process 1 (the pytest process) wrote, resumes the pending interrupt with
a real `HumanApproval` payload, and prints the final state as one JSON
line to stdout for the parent process to parse.
"""

from __future__ import annotations

import asyncio
import json
import sys

from langgraph.graph import END, START, StateGraph

from agentic_frameworks.graph.nodes.human_approval_gate import human_approval_gate_node
from agentic_frameworks.graph.schemas import (
    AdvisorReport,
    HumanApproval,
    ReportCoverage,
    SynthesizedRecommendation,
)
from agentic_frameworks.graph.state import GraphState

REAL_INDEX_FUND_ISIN = "INF179K01WN9"
REAL_INDEX_FUND_NAME = "HDFC BSE Sensex Index Fund - Direct Plan - Growth Option"


def _stub_recommendation() -> dict:
    report = AdvisorReport.from_synthesis(
        SynthesizedRecommendation(
            portfolio_assessment="Real-estate heavy; one fund flagged for exit.",
            allocation_notes=[],
            existing_holdings_guidance=[
                {
                    "identifier": REAL_INDEX_FUND_ISIN,
                    "action": "Sell",
                    "rationale": "Redundant with a newer, lower-cost index fund holding.",
                    "risk_flags": ["redundant"],
                    "confidence": 0.65,
                }
            ],
            new_buy_suggestions=[],
            no_new_buy_reason="No Buy-verdict candidates this run.",
        ),
        subject_index={REAL_INDEX_FUND_ISIN: ("isin", REAL_INDEX_FUND_NAME, "existing", None)},
        coverage=ReportCoverage(
            holdings_total=1,
            holdings_individually_assessed=1,
            unassessed_buckets=[],
            coverage_note="Stub coverage for the cross-process resume probe.",
        ),
    )
    return report.model_dump()


async def _stub_producer_node(state: dict) -> dict:
    return {"recommendation": _stub_recommendation()}


def build_mini_graph() -> StateGraph:
    graph = StateGraph(GraphState)
    graph.add_node("produce", _stub_producer_node)
    graph.add_node("human_approval_gate", human_approval_gate_node)
    graph.add_edge(START, "produce")
    graph.add_edge("produce", "human_approval_gate")
    graph.add_edge("human_approval_gate", END)
    return graph


async def _resume(db_path: str, thread_id: str) -> dict:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.types import Command

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        compiled = build_mini_graph().compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        state_before = await compiled.aget_state(config)
        resume_payload = HumanApproval(
            decision="approved",
            approved_by="abhi",
            excluded_identifiers=[],
            note="Approved from the cross-process resume probe.",
            decided_at="2026-09-19T00:00:00+00:00",
        ).model_dump()

        result = await compiled.ainvoke(Command(resume=resume_payload), config=config)
        state_after = await compiled.aget_state(config)

        return {
            "next_before": list(state_before.next),
            "recommendation_present_before": "recommendation" in state_before.values,
            "approval": result.get("approval"),
            "next_after": list(state_after.next),
        }


if __name__ == "__main__":
    db_path_arg, thread_id_arg = sys.argv[1], sys.argv[2]
    payload = asyncio.run(_resume(db_path_arg, thread_id_arg))
    print(json.dumps(payload))
