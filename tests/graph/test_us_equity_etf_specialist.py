"""Tests for the US Equity/ETF Specialist node and its stages."""

from __future__ import annotations


async def test_verify_ticker_returns_real_verified_name():
    """Verification now returns the instrument's real longName (not just a
    bool) so the node can attach a verified display_name to the
    evaluation without a second lookup or an LLM-supplied name."""
    from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset
    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import _verify_ticker

    toolset = get_yfinance_toolset()
    async with toolset:
        name = await _verify_ticker("AAPL", toolset)

    assert isinstance(name, str)
    assert "Apple" in name


async def test_verify_ticker_rejects_fake_ticker():
    from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset
    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import _verify_ticker

    toolset = get_yfinance_toolset()
    async with toolset:
        assert await _verify_ticker("NOTAREALTICKER999", toolset) is None


async def test_candidate_generator_produces_proposed_candidates():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import (
        _build_candidate_generator,
    )

    generator = _build_candidate_generator(TestModel())
    result = await generator.run("propose candidates for: Energy")

    assert isinstance(result.output, list)
    assert all(hasattr(c, "ticker") and hasattr(c, "sector") for c in result.output)


def test_summarize_us_holdings_lists_real_equity_and_etf_holdings():
    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import (
        _summarize_us_holdings,
    )

    holdings_by_bucket = {
        "equity": {
            "US": [{"ticker": "GOOGL", "name": "Alphabet Inc", "sector": "Technology"}],
            "IN": [{"ticker": "TCS", "name": "Tata Consultancy Services", "sector": "IT"}],
        },
        "etf": {
            "US": [{"ticker": "VOO", "name": "Vanguard S&P 500 ETF", "sector": "US Large Cap"}],
        },
    }

    summary = _summarize_us_holdings(holdings_by_bucket)

    assert "GOOGL" in summary
    assert "VOO" in summary
    assert "TCS" not in summary


def test_summarize_us_holdings_handles_no_us_holdings():
    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import (
        _summarize_us_holdings,
    )

    summary = _summarize_us_holdings({})

    assert "no US equities or ETFs" in summary


async def test_conviction_gate_evaluator_produces_candidate_evaluation():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.mcp_clients import get_yfinance_toolset
    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import (
        _build_conviction_gate_evaluator,
    )

    toolset = get_yfinance_toolset()
    async with toolset:
        evaluator = _build_conviction_gate_evaluator(
            TestModel(), toolset, "The user currently holds no US equities or ETFs."
        )
        result = await evaluator.run("Evaluate ticker: XOM, gap sector: Energy")

        assert result.output.action in {"Buy", "Hold", "Sell", "Reduce"}
        assert isinstance(result.output.confidence, float)


async def test_compute_us_gaps_excludes_information_technology_given_real_holdings():
    from agentic_frameworks.graph.nodes.research import research_node
    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import (
        _compute_us_gaps,
    )

    research_result = await research_node({})

    gaps = _compute_us_gaps(research_result["holdings_by_bucket"])

    assert "Information Technology" not in gaps
    assert "Energy" in gaps
    assert "Health Care" in gaps
    assert len(gaps) == 10


def test_compute_us_gaps_raises_on_unmapped_sector():
    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import _compute_us_gaps

    holdings_by_bucket = {
        "equity": {
            "US": [{"ticker": "ZZZ", "name": "Fictional Co", "sector": "Not A Real Sector"}],
        }
    }

    with __import__("pytest").raises(KeyError):
        _compute_us_gaps(holdings_by_bucket)


def test_compute_us_gaps_skips_null_sector_rows():
    from agentic_frameworks.graph.nodes.us_equity_etf_specialist import _compute_us_gaps

    holdings_by_bucket = {
        "equity": {
            "US": [{"ticker": "MANUAL_TOTAL", "name": "Manual Total", "sector": None}],
        }
    }

    gaps = _compute_us_gaps(holdings_by_bucket)

    assert len(gaps) == 11


async def test_us_equity_etf_specialist_node_returns_candidate_evaluations(memory_store):
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes import us_equity_etf_specialist as specialist_module

    def fake_get_specialist_model():
        return TestModel()

    state = {
        "holdings_by_bucket": {
            "equity": {
                "US": [{"ticker": "GOOGL", "name": "Alphabet Inc", "sector": "Technology"}],
            },
            "etf": {
                "US": [{"ticker": "VOO", "name": "Vanguard S&P 500 ETF", "sector": "US Large Cap"}],
            },
        }
    }

    original = specialist_module.get_specialist_model
    specialist_module.get_specialist_model = fake_get_specialist_model
    try:
        result = await specialist_module.us_equity_etf_specialist_node(state, store=memory_store)
    finally:
        specialist_module.get_specialist_model = original

    assert "us_equity_etf_candidates" in result
    assert isinstance(result["us_equity_etf_candidates"], list)
    assert result["us_equity_etf_candidates"]
    for candidate in result["us_equity_etf_candidates"]:
        assert candidate["action"] in {"Buy", "Hold", "Sell", "Reduce"}
        assert candidate["identifier_kind"] == "ticker"
        assert candidate["identifier"]
        assert candidate["display_name"]


async def test_us_equity_etf_specialist_node_binds_verified_identity_not_llm_claims(memory_store):
    """Every evaluation must carry the ticker that was actually verified
    and the name yfinance returned for it - never a name the model made
    up, and never an identity implied by list position."""
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes import us_equity_etf_specialist as specialist_module

    verified_names = {}

    async def fake_verify_ticker(symbol, toolset):
        # A deterministic fake standing in for _verify_ticker's real
        # yfinance-mcp lookup, at this substituted test seam - NOT real
        # data (no such longName exists), just a synthesized string keyed
        # by the symbol actually checked, so this test can assert the
        # verified identity round-trips into the candidate output.
        name = f"{symbol.upper()} Verified Instrument Name"
        verified_names[symbol] = name
        return name

    def fake_get_specialist_model():
        return TestModel()

    state = {
        "holdings_by_bucket": {
            "equity": {"US": [{"ticker": "GOOGL", "name": "Alphabet Inc", "sector": "Technology"}]},
        }
    }

    original_model = specialist_module.get_specialist_model
    original_verify = specialist_module._verify_ticker
    specialist_module.get_specialist_model = fake_get_specialist_model
    specialist_module._verify_ticker = fake_verify_ticker
    try:
        result = await specialist_module.us_equity_etf_specialist_node(state, store=memory_store)
    finally:
        specialist_module.get_specialist_model = original_model
        specialist_module._verify_ticker = original_verify

    assert result["us_equity_etf_candidates"]
    for candidate in result["us_equity_etf_candidates"]:
        assert candidate["identifier"] in verified_names
        assert candidate["display_name"] == verified_names[candidate["identifier"]]


async def test_us_equity_etf_specialist_node_returns_empty_list_when_no_gaps_and_skips_model_setup():
    from agentic_frameworks.graph.nodes import us_equity_etf_specialist as specialist_module

    def compute_no_gaps(holdings_by_bucket):
        return []

    def fail_if_called():
        raise AssertionError("get_specialist_model should not be called when there are no gaps")

    original_compute = specialist_module._compute_us_gaps
    original_model_getter = specialist_module.get_specialist_model
    specialist_module._compute_us_gaps = compute_no_gaps
    specialist_module.get_specialist_model = fail_if_called
    try:
        # store=None is fine here: the no-gap reject-early return happens
        # before the store-is-None guard, so this also incidentally proves
        # the reject-early ordering (no store access on an empty gap list).
        result = await specialist_module.us_equity_etf_specialist_node({"holdings_by_bucket": {}})
    finally:
        specialist_module._compute_us_gaps = original_compute
        specialist_module.get_specialist_model = original_model_getter

    assert result == {"us_equity_etf_candidates": []}


async def test_us_equity_etf_specialist_node_raises_without_store():
    import pytest

    from agentic_frameworks.graph.nodes import us_equity_etf_specialist as specialist_module

    state = {
        "holdings_by_bucket": {
            "equity": {"US": [{"ticker": "GOOGL", "name": "Alphabet Inc", "sector": "Technology"}]},
        }
    }

    with pytest.raises(ValueError, match="requires a store"):
        await specialist_module.us_equity_etf_specialist_node(state, store=None)


async def test_us_equity_etf_specialist_node_suppresses_previously_rejected_candidate_across_runs(
    memory_store,
):
    """Cross-run suppression, counted: run 1 evaluates and records a
    non-Buy verdict; run 2 proposes the same ticker and must make zero
    evaluator calls for it - counting calls, not just the output shape,
    is what actually proves money was saved (design spec Testing #6).

    The evaluator itself is replaced by a deterministic fake returning a
    fixed "Hold" verdict (TestModel would otherwise deterministically
    emit "Buy" - the one action this design never suppresses - which
    would make this test exercise the wrong branch every time)."""
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes import us_equity_etf_specialist as specialist_module
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
            "equity": {"US": [{"ticker": "GOOGL", "name": "Alphabet Inc", "sector": "Technology"}]},
        }
    }

    original_model = specialist_module.get_specialist_model
    original_verify = specialist_module._verify_ticker
    original_build_evaluator = specialist_module._build_conviction_gate_evaluator
    specialist_module.get_specialist_model = fake_get_specialist_model
    specialist_module._verify_ticker = fake_verify_ticker
    specialist_module._build_conviction_gate_evaluator = fake_build_evaluator
    try:
        result_1 = await specialist_module.us_equity_etf_specialist_node(state, store=memory_store)
        assert result_1["us_equity_etf_candidates"]
        assert result_1["us_equity_etf_candidates"][0]["action"] == "Hold"
        calls_after_run_1 = evaluator_call_count["n"]
        assert calls_after_run_1 > 0

        result_2 = await specialist_module.us_equity_etf_specialist_node(state, store=memory_store)
        calls_after_run_2 = evaluator_call_count["n"]

        assert result_2["us_equity_etf_candidates"] == []
        assert calls_after_run_2 == calls_after_run_1
    finally:
        specialist_module.get_specialist_model = original_model
        specialist_module._verify_ticker = original_verify
        specialist_module._build_conviction_gate_evaluator = original_build_evaluator


async def test_us_equity_etf_specialist_node_never_suppresses_a_buy_verdict(memory_store):
    """A Buy verdict writes no rejection record and is re-evaluated on
    the next run (design spec Architecture #2)."""
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes import us_equity_etf_specialist as specialist_module
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
                    suggested_platform="moomoo",
                )
            )

    def fake_build_evaluator(*args, **kwargs):
        return _FakeEvaluator()

    state = {
        "holdings_by_bucket": {
            "equity": {"US": [{"ticker": "GOOGL", "name": "Alphabet Inc", "sector": "Technology"}]},
        }
    }

    original_model = specialist_module.get_specialist_model
    original_verify = specialist_module._verify_ticker
    original_build_evaluator = specialist_module._build_conviction_gate_evaluator
    specialist_module.get_specialist_model = fake_get_specialist_model
    specialist_module._verify_ticker = fake_verify_ticker
    specialist_module._build_conviction_gate_evaluator = fake_build_evaluator
    try:
        result_1 = await specialist_module.us_equity_etf_specialist_node(state, store=memory_store)
        assert result_1["us_equity_etf_candidates"][0]["action"] == "Buy"
        calls_after_run_1 = evaluator_call_count["n"]

        result_2 = await specialist_module.us_equity_etf_specialist_node(state, store=memory_store)
        calls_after_run_2 = evaluator_call_count["n"]

        assert result_2["us_equity_etf_candidates"]
        assert calls_after_run_2 > calls_after_run_1
    finally:
        specialist_module.get_specialist_model = original_model
        specialist_module._verify_ticker = original_verify
        specialist_module._build_conviction_gate_evaluator = original_build_evaluator
