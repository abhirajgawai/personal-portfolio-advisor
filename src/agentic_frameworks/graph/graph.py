"""Builds the Portfolio Advisor Fleet graph: wires existing nodes together.

Research -> Risk Analyst -> {US Equity/ETF Specialist, India Equity
Specialist, Mutual Fund Holdings Analyzer, Mutual Fund Specialist} in
parallel. All four branches currently dead-end at END - fan-in to a
Synthesizer arrives in a later phase.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

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
from agentic_frameworks.graph.nodes.us_equity_etf_specialist import (
    us_equity_etf_specialist_node,
)
from agentic_frameworks.graph.state import GraphState


def build_graph() -> StateGraph:
    """Wire research, risk_analyst, and all four new-buy/holdings-analysis nodes into an uncompiled StateGraph."""
    graph = StateGraph(GraphState)
    graph.add_node("research", research_node)
    graph.add_node("risk_analyst", risk_analyst_node)
    graph.add_node("us_equity_etf_specialist", us_equity_etf_specialist_node)
    graph.add_node("india_equity_specialist", india_equity_specialist_node)
    graph.add_node("mutual_fund_holdings_analyzer", mutual_fund_holdings_analyzer_node)
    graph.add_node("mutual_fund_specialist", mutual_fund_specialist_node)
    graph.add_edge(START, "research")
    graph.add_edge("research", "risk_analyst")
    graph.add_edge("risk_analyst", "us_equity_etf_specialist")
    graph.add_edge("risk_analyst", "india_equity_specialist")
    graph.add_edge("risk_analyst", "mutual_fund_holdings_analyzer")
    graph.add_edge("risk_analyst", "mutual_fund_specialist")
    graph.add_edge("us_equity_etf_specialist", END)
    graph.add_edge("india_equity_specialist", END)
    graph.add_edge("mutual_fund_holdings_analyzer", END)
    graph.add_edge("mutual_fund_specialist", END)
    return graph
