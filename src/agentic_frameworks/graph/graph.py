"""Builds the Portfolio Advisor Fleet graph: wires existing nodes together.

Research -> Risk Analyst -> {US Equity/ETF Specialist, India Equity
Specialist, Mutual Fund Holdings Analyzer, Mutual Fund Specialist} in
parallel -> Synthesizer -> (human_approval_gate | END), per design spec
docs/superpowers/specs/2026-09-19-synthesizer-design.md.
"""

from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from agentic_frameworks.graph.nodes.human_approval_gate import (
    _requires_human_approval,
    human_approval_gate_node,
)
from agentic_frameworks.graph.nodes.india_equity_specialist import (
    india_equity_specialist_node,
)
from agentic_frameworks.graph.nodes.mutual_fund_holdings_analyzer import (
    mutual_fund_holdings_analyzer_node,
)
from agentic_frameworks.graph.nodes.mutual_fund_specialist import (
    mutual_fund_specialist_node,
)
from agentic_frameworks.graph.nodes.research import research_node
from agentic_frameworks.graph.nodes.risk_analyst import risk_analyst_node
from agentic_frameworks.graph.nodes.synthesizer import synthesizer_node
from agentic_frameworks.graph.nodes.us_equity_etf_specialist import (
    us_equity_etf_specialist_node,
)
from agentic_frameworks.graph.state import GraphState


def _route_after_synthesis(state: GraphState) -> Literal["human_approval_gate", "__end__"]:
    """A pure, deterministic predicate over the finished report - not the
    model's own opinion about whether approval is needed (design spec
    Architecture point 5). Delegates to the same
    `_requires_human_approval` the gate node's own tests exercise
    standalone, so routing and the gate's own reasoning can never drift
    apart.
    """
    if _requires_human_approval(state["recommendation"]):
        return "human_approval_gate"
    return END


def build_graph() -> StateGraph:
    """Wire every node into an uncompiled StateGraph.

    Stays uncompiled (unchanged contract) so Studio and any call site
    that wants a different checkpointer/store backing can compile it
    themselves - see build_compiled_graph for the one call site meant to
    produce a real, resumable graph.
    """
    graph = StateGraph(GraphState)
    graph.add_node("research", research_node)
    graph.add_node("risk_analyst", risk_analyst_node)
    graph.add_node("us_equity_etf_specialist", us_equity_etf_specialist_node)
    graph.add_node("india_equity_specialist", india_equity_specialist_node)
    graph.add_node("mutual_fund_holdings_analyzer", mutual_fund_holdings_analyzer_node)
    graph.add_node("mutual_fund_specialist", mutual_fund_specialist_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("human_approval_gate", human_approval_gate_node)
    graph.add_edge(START, "research")
    graph.add_edge("research", "risk_analyst")
    graph.add_edge("risk_analyst", "us_equity_etf_specialist")
    graph.add_edge("risk_analyst", "india_equity_specialist")
    graph.add_edge("risk_analyst", "mutual_fund_holdings_analyzer")
    graph.add_edge("risk_analyst", "mutual_fund_specialist")
    graph.add_edge("us_equity_etf_specialist", "synthesizer")
    graph.add_edge("india_equity_specialist", "synthesizer")
    graph.add_edge("mutual_fund_holdings_analyzer", "synthesizer")
    graph.add_edge("mutual_fund_specialist", "synthesizer")
    graph.add_conditional_edges("synthesizer", _route_after_synthesis, ["human_approval_gate", END])
    graph.add_edge("human_approval_gate", END)
    return graph


def build_compiled_graph(
    *, checkpointer: BaseCheckpointSaver, store: BaseStore
) -> CompiledStateGraph:
    """The one factory that produces a real, resumable advisor graph.

    Both `checkpointer` and `store` are keyword-only and required (design
    spec Decision 5) - so there is no way to compile the advisor graph
    without the two things its guardrails structurally depend on:
    `human_approval_gate_node`'s interrupt() needs a real checkpointer to
    be resumable at all (see its own fail-loud guard, which still applies
    even if a caller bypassed this factory), and every new-buy specialist
    plus the Synthesizer need a real store for rejection memory.
    `build_graph()` itself is unchanged and stays uncompiled, for Studio
    and any call site that wants a different backing.
    """
    return build_graph().compile(checkpointer=checkpointer, store=store)
