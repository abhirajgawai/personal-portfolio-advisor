"""Builds the Portfolio Advisor Fleet graph: wires existing nodes together.

Linear for now (Research -> Risk Analyst) - parallel specialist
branches and the Synthesizer arrive in later phases.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agentic_frameworks.graph.nodes.research import research_node
from agentic_frameworks.graph.nodes.risk_analyst import risk_analyst_node
from agentic_frameworks.graph.state import GraphState


def build_graph() -> StateGraph:
    """Wire research_node and risk_analyst_node into an uncompiled StateGraph."""
    graph = StateGraph(GraphState)
    graph.add_node("research", research_node)
    graph.add_node("risk_analyst", risk_analyst_node)
    graph.add_edge(START, "research")
    graph.add_edge("research", "risk_analyst")
    graph.add_edge("risk_analyst", END)
    return graph
