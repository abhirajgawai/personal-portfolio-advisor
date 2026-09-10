"""Tests for the graph's node/edge wiring and end-to-end execution."""

from __future__ import annotations

import pytest


def test_build_graph_wires_research_and_risk_analyst():
    from agentic_frameworks.graph.graph import build_graph

    graph = build_graph()

    assert set(graph.nodes.keys()) == {"research", "risk_analyst"}
