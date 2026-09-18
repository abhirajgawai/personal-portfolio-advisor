"""Tests for the India Equity Specialist node and its stages."""

from __future__ import annotations


async def test_compute_india_equity_gaps_finds_only_energy_given_real_holdings_and_real_estate_ownership():
    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _compute_india_equity_gaps,
    )
    from agentic_frameworks.graph.nodes.research import research_node

    research_result = await research_node({})

    gaps = _compute_india_equity_gaps(research_result["holdings_by_bucket"])

    assert set(gaps) == {"Energy"}


def test_compute_india_equity_gaps_raises_on_unmapped_sector():
    import pytest

    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _compute_india_equity_gaps,
    )

    holdings_by_bucket = {
        "equity": {
            "IN": [{"ticker": "ZZZ.NS", "name": "Fictional Co", "sector": "Not A Real Sector"}],
        }
    }

    with pytest.raises(KeyError):
        _compute_india_equity_gaps(holdings_by_bucket)


def test_compute_india_equity_gaps_skips_null_sector_rows():
    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _compute_india_equity_gaps,
    )

    holdings_by_bucket = {
        "equity": {
            "IN": [{"ticker": "MANUAL_TOTAL", "name": "Manual Total", "sector": None}],
        }
    }

    gaps = _compute_india_equity_gaps(holdings_by_bucket)

    assert len(gaps) == 11


def test_compute_india_equity_gaps_treats_real_estate_holding_as_covered():
    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _compute_india_equity_gaps,
    )

    holdings_by_bucket = {
        "equity": {"IN": []},
        "real_estate": {"IN": [{"ticker": "PROPERTY", "name": "Residential Property", "sector": "Residential"}]},
    }

    gaps = _compute_india_equity_gaps(holdings_by_bucket)

    assert "Real Estate" not in gaps


def test_compute_india_equity_gaps_still_flags_real_estate_when_no_real_estate_holding():
    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _compute_india_equity_gaps,
    )

    holdings_by_bucket = {"equity": {"IN": []}}

    gaps = _compute_india_equity_gaps(holdings_by_bucket)

    assert "Real Estate" in gaps


def test_summarize_india_equity_holdings_lists_real_in_equity():
    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _summarize_india_equity_holdings,
    )

    holdings_by_bucket = {
        "equity": {
            "IN": [{"ticker": "TCS", "name": "Tata Consultancy Services", "sector": "Information Technology"}],
            "US": [{"ticker": "GOOGL", "name": "Alphabet Inc", "sector": "Technology"}],
        },
    }

    summary = _summarize_india_equity_holdings(holdings_by_bucket)

    assert "TCS" in summary
    assert "GOOGL" not in summary


def test_summarize_india_equity_holdings_handles_no_in_holdings():
    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _summarize_india_equity_holdings,
    )

    summary = _summarize_india_equity_holdings({})

    assert "no India equities" in summary


async def test_india_candidate_generator_produces_proposed_candidates():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _build_candidate_generator,
    )

    generator = _build_candidate_generator(TestModel())
    result = await generator.run("propose candidates for: Energy")

    assert isinstance(result.output, list)
    assert all(hasattr(c, "ticker") and hasattr(c, "sector") for c in result.output)


async def test_india_conviction_gate_evaluator_produces_candidate_evaluation():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset
    from agentic_frameworks.graph.nodes.india_equity_specialist import (
        _build_conviction_gate_evaluator,
    )

    toolset = get_yfinance_toolset()
    async with toolset:
        evaluator = _build_conviction_gate_evaluator(
            TestModel(), toolset, "The user currently holds no India equities."
        )
        result = await evaluator.run("Evaluate ticker: RELIANCE.NS, gap sector: Energy")

        assert result.output.action in {"Buy", "Hold", "Sell", "Reduce"}
        assert isinstance(result.output.confidence, float)


async def test_india_equity_specialist_node_returns_candidate_evaluations(memory_store):
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes import india_equity_specialist as specialist_module

    def fake_get_specialist_model():
        return TestModel()

    state = {
        "holdings_by_bucket": {
            "equity": {
                "IN": [{"ticker": "TCS", "name": "Tata Consultancy Services", "sector": "Information Technology"}],
            },
        }
    }

    original = specialist_module.get_specialist_model
    specialist_module.get_specialist_model = fake_get_specialist_model
    try:
        result = await specialist_module.india_equity_specialist_node(state, store=memory_store)
    finally:
        specialist_module.get_specialist_model = original

    assert "india_equity_candidates" in result
    assert isinstance(result["india_equity_candidates"], list)
    assert result["india_equity_candidates"]
    for candidate in result["india_equity_candidates"]:
        assert candidate["action"] in {"Buy", "Hold", "Sell", "Reduce"}
        assert candidate["identifier_kind"] == "ticker"
        assert candidate["identifier"]
        assert candidate["display_name"]


async def test_india_equity_specialist_node_binds_verified_identity_not_llm_claims(memory_store):
    """Same guarantee as the US specialist: identity comes from the
    verification step, not from the model or from list position."""
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes import india_equity_specialist as specialist_module

    verified_names = {}

    async def fake_verify_ticker(symbol, toolset):
        name = f"{symbol.upper()} Verified Instrument Name"
        verified_names[symbol] = name
        return name

    def fake_get_specialist_model():
        return TestModel()

    state = {
        "holdings_by_bucket": {
            "equity": {
                "IN": [{"ticker": "TCS", "name": "Tata Consultancy Services", "sector": "Information Technology"}],
            },
        }
    }

    original_model = specialist_module.get_specialist_model
    original_verify = specialist_module._verify_ticker
    specialist_module.get_specialist_model = fake_get_specialist_model
    specialist_module._verify_ticker = fake_verify_ticker
    try:
        result = await specialist_module.india_equity_specialist_node(state, store=memory_store)
    finally:
        specialist_module.get_specialist_model = original_model
        specialist_module._verify_ticker = original_verify

    assert result["india_equity_candidates"]
    for candidate in result["india_equity_candidates"]:
        assert candidate["identifier"] in verified_names
        assert candidate["display_name"] == verified_names[candidate["identifier"]]


async def test_india_equity_specialist_node_returns_empty_list_when_no_gaps_and_skips_model_setup():
    from agentic_frameworks.graph.nodes import india_equity_specialist as specialist_module

    def compute_no_gaps(holdings_by_bucket):
        return []

    def fail_if_called():
        raise AssertionError("get_specialist_model should not be called when there are no gaps")

    original_compute = specialist_module._compute_india_equity_gaps
    original_model_getter = specialist_module.get_specialist_model
    specialist_module._compute_india_equity_gaps = compute_no_gaps
    specialist_module.get_specialist_model = fail_if_called
    try:
        result = await specialist_module.india_equity_specialist_node({"holdings_by_bucket": {}})
    finally:
        specialist_module._compute_india_equity_gaps = original_compute
        specialist_module.get_specialist_model = original_model_getter

    assert result == {"india_equity_candidates": []}


async def test_india_equity_specialist_node_raises_without_store():
    import pytest

    from agentic_frameworks.graph.nodes import india_equity_specialist as specialist_module

    state = {
        "holdings_by_bucket": {
            "equity": {"IN": [{"ticker": "TCS", "name": "Tata Consultancy Services", "sector": "Information Technology"}]},
        }
    }

    with pytest.raises(ValueError, match="requires a store"):
        await specialist_module.india_equity_specialist_node(state, store=None)


async def test_india_equity_specialist_node_suppresses_previously_rejected_candidate_across_runs(
    memory_store,
):
    """Parity with the US specialist's identical test: cross-run
    suppression, counted - run 1 evaluates and records a non-Buy
    verdict; run 2 proposes the same ticker and must make zero
    evaluator calls for it - counting calls, not just the output shape,
    is what actually proves money was saved (design spec Testing #6).

    The evaluator itself is replaced by a deterministic fake returning a
    fixed "Hold" verdict (TestModel would otherwise deterministically
    emit "Buy" - the one action this design never suppresses - which
    would make this test exercise the wrong branch every time)."""
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes import india_equity_specialist as specialist_module
    from agentic_frameworks.graph.schemas import ConvictionVerdict

    evaluator_call_count = {"n": 0}

    async def fake_verify_ticker(symbol, toolset):
        return f"{symbol.upper()} Verified Instrument Name"

    def fake_get_specialist_model():
        return TestModel()

    class _FakeEvalResult:
        def __init__(self, output):
            self.output = output

    class _FakeEvaluator:
        async def run(self, prompt):
            evaluator_call_count["n"] += 1
            return _FakeEvalResult(
                ConvictionVerdict(
                    action="Hold",
                    confidence=0.5,
                    reasoning="Fairly valued, no clear catalyst.",
                    risk_flags=["no_catalyst"],
                    entry_approach=None,
                    suggested_platform=None,
                )
            )

    def fake_build_evaluator(*args, **kwargs):
        return _FakeEvaluator()

    state = {
        "holdings_by_bucket": {
            "equity": {
                "IN": [{"ticker": "TCS", "name": "Tata Consultancy Services", "sector": "Information Technology"}],
            },
        }
    }

    original_model = specialist_module.get_specialist_model
    original_verify = specialist_module._verify_ticker
    original_build_evaluator = specialist_module._build_conviction_gate_evaluator
    specialist_module.get_specialist_model = fake_get_specialist_model
    specialist_module._verify_ticker = fake_verify_ticker
    specialist_module._build_conviction_gate_evaluator = fake_build_evaluator
    try:
        result_1 = await specialist_module.india_equity_specialist_node(state, store=memory_store)
        assert result_1["india_equity_candidates"]
        assert result_1["india_equity_candidates"][0]["action"] == "Hold"
        calls_after_run_1 = evaluator_call_count["n"]
        assert calls_after_run_1 > 0

        result_2 = await specialist_module.india_equity_specialist_node(state, store=memory_store)
        calls_after_run_2 = evaluator_call_count["n"]

        assert result_2["india_equity_candidates"] == []
        assert calls_after_run_2 == calls_after_run_1
    finally:
        specialist_module.get_specialist_model = original_model
        specialist_module._verify_ticker = original_verify
        specialist_module._build_conviction_gate_evaluator = original_build_evaluator


async def test_india_equity_specialist_node_never_suppresses_a_buy_verdict(memory_store):
    """Parity with the US specialist's identical test: a Buy verdict
    writes no rejection record and is re-evaluated on the next run
    (design spec Architecture #2)."""
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes import india_equity_specialist as specialist_module
    from agentic_frameworks.graph.schemas import ConvictionVerdict

    evaluator_call_count = {"n": 0}

    async def fake_verify_ticker(symbol, toolset):
        return f"{symbol.upper()} Verified Instrument Name"

    def fake_get_specialist_model():
        return TestModel()

    class _FakeEvalResult:
        def __init__(self, output):
            self.output = output

    class _FakeEvaluator:
        async def run(self, prompt):
            evaluator_call_count["n"] += 1
            return _FakeEvalResult(
                ConvictionVerdict(
                    action="Buy",
                    confidence=0.7,
                    reasoning="Fills a real gap with strong fundamentals.",
                    risk_flags=[],
                    entry_approach="Staggered",
                    suggested_platform="Zerodha",
                )
            )

    def fake_build_evaluator(*args, **kwargs):
        return _FakeEvaluator()

    state = {
        "holdings_by_bucket": {
            "equity": {
                "IN": [{"ticker": "TCS", "name": "Tata Consultancy Services", "sector": "Information Technology"}],
            },
        }
    }

    original_model = specialist_module.get_specialist_model
    original_verify = specialist_module._verify_ticker
    original_build_evaluator = specialist_module._build_conviction_gate_evaluator
    specialist_module.get_specialist_model = fake_get_specialist_model
    specialist_module._verify_ticker = fake_verify_ticker
    specialist_module._build_conviction_gate_evaluator = fake_build_evaluator
    try:
        result_1 = await specialist_module.india_equity_specialist_node(state, store=memory_store)
        assert result_1["india_equity_candidates"][0]["action"] == "Buy"
        calls_after_run_1 = evaluator_call_count["n"]

        result_2 = await specialist_module.india_equity_specialist_node(state, store=memory_store)
        calls_after_run_2 = evaluator_call_count["n"]

        # A Buy verdict is never suppressed - run 2 evaluates the same
        # candidate again (same evaluator call count growth as run 1),
        # not zero, since no rejection record was ever written for it.
        assert result_2["india_equity_candidates"]
        assert result_2["india_equity_candidates"][0]["action"] == "Buy"
        assert calls_after_run_2 == calls_after_run_1 * 2
    finally:
        specialist_module.get_specialist_model = original_model
        specialist_module._verify_ticker = original_verify
        specialist_module._build_conviction_gate_evaluator = original_build_evaluator
