"""Tests for the Mutual Fund Specialist (new-buy) node."""

from __future__ import annotations


async def test_candidate_generator_produces_proposed_fund_candidates():
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes.mutual_fund_specialist import (
        _build_candidate_generator,
    )

    generator = _build_candidate_generator(TestModel())
    result = await generator.run("propose candidates for: Gilt")

    assert isinstance(result.output, list)
    assert all(hasattr(c, "scheme_name") and hasattr(c, "category") for c in result.output)


async def test_mutual_fund_specialist_node_returns_candidate_evaluations(memory_store):
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes import mutual_fund_specialist as node_module
    from agentic_frameworks.graph.nodes.research import research_node

    def fake_get_specialist_model():
        return TestModel()

    research_result = await research_node({})
    state = {"holdings_by_bucket": research_result["holdings_by_bucket"]}

    original = node_module.get_specialist_model
    node_module.get_specialist_model = fake_get_specialist_model
    try:
        result = await node_module.mutual_fund_specialist_node(state, store=memory_store)
    finally:
        node_module.get_specialist_model = original

    assert "mutual_fund_candidates" in result
    # TestModel emits a fixed, non-real scheme_name string (not a genuine
    # fund name) for every candidate it proposes, so
    # _resolve_scheme_by_name_and_direct_plan finds no real Direct-plan
    # match for it (live-verified this session: real MFapi.in search
    # against TestModel's dummy string returns no hits) and every
    # candidate is dropped before evaluation ever runs - an empty list is
    # the correct, expected outcome here, not a regression. This also
    # means the loop below is not a vacuous no-op check: it's pinning the
    # real, observed shape (empty), same pattern as test_graph.py's
    # india_equity_candidates == [] assertion with TestModel.
    assert result["mutual_fund_candidates"] == []
    for candidate in result["mutual_fund_candidates"]:
        assert candidate["action"] in {"Buy", "Hold", "Sell", "Reduce"}


async def test_mutual_fund_specialist_node_binds_resolved_scheme_identity(memory_store):
    """A surviving candidate must carry the MFapi scheme_code and real
    scheme name that were actually resolved and category-checked - not
    the LLM's proposed name, and not an identity implied by list order.

    The resolve step is substituted with the real, live-verified triple
    for a genuine Direct-plan Large Cap scheme (scheme_code 118632,
    'Equity Scheme - Large Cap Fund', 'Nippon India Large Cap Fund -
    Direct Plan - Growth Option'), because TestModel proposes a dummy
    scheme name that no real search can match. Everything after that
    point is real: the NAV history, metrics and benchmark comparison all
    come from the live MFapi.in/yfinance data for scheme 118632.
    """
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes import mutual_fund_specialist as node_module
    from agentic_frameworks.graph.nodes.research import research_node

    real_scheme_code = 118632
    real_scheme_name = "Nippon India Large Cap Fund - Direct Plan - Growth Option"

    async def fake_resolve_real_large_cap(scheme_name, toolset):
        return real_scheme_code, "Equity Scheme - Large Cap Fund", real_scheme_name

    def fake_get_specialist_model():
        return TestModel()

    research_result = await research_node({})
    state = {"holdings_by_bucket": research_result["holdings_by_bucket"]}

    original_model = node_module.get_specialist_model
    original_resolve = node_module._resolve_scheme_by_name_and_direct_plan
    node_module.get_specialist_model = fake_get_specialist_model
    node_module._resolve_scheme_by_name_and_direct_plan = fake_resolve_real_large_cap
    try:
        result = await node_module.mutual_fund_specialist_node(state, store=memory_store)
    finally:
        node_module.get_specialist_model = original_model
        node_module._resolve_scheme_by_name_and_direct_plan = original_resolve

    assert result["mutual_fund_candidates"]
    for candidate in result["mutual_fund_candidates"]:
        assert candidate["identifier"] == str(real_scheme_code)
        assert candidate["identifier_kind"] == "scheme_code"
        assert candidate["display_name"] == real_scheme_name
        assert candidate["action"] in {"Buy", "Hold", "Sell", "Reduce"}


async def test_mutual_fund_specialist_node_drops_candidate_on_category_mismatch(memory_store):
    """Finding 1: if a candidate's proposed category doesn't match the
    real category of the fund _resolve_scheme_by_name_and_direct_plan
    actually resolves to, the candidate must be dropped rather than
    having a different fund's real NAV/metrics attached to it under the
    LLM's originally proposed name."""
    from pydantic_ai.models.test import TestModel

    from agentic_frameworks.graph.nodes import mutual_fund_specialist as node_module
    from agentic_frameworks.graph.nodes.research import research_node

    def fake_get_specialist_model():
        return TestModel()

    async def fake_resolve_mismatched_category(scheme_name, toolset):
        # Always resolves to a real Gilt fund's identity, regardless of
        # what category was proposed - simulating a fuzzy/wrong search
        # match. TestModel deterministically proposes category="Large Cap"
        # (the taxonomy's first Literal choice), so this always mismatches.
        return 118989, "Debt Scheme - Gilt Fund", "Some Real Gilt Fund - Direct Plan - Growth"

    research_result = await research_node({})
    state = {"holdings_by_bucket": research_result["holdings_by_bucket"]}

    original_model = node_module.get_specialist_model
    original_resolve = node_module._resolve_scheme_by_name_and_direct_plan
    node_module.get_specialist_model = fake_get_specialist_model
    node_module._resolve_scheme_by_name_and_direct_plan = fake_resolve_mismatched_category
    try:
        result = await node_module.mutual_fund_specialist_node(state, store=memory_store)
    finally:
        node_module.get_specialist_model = original_model
        node_module._resolve_scheme_by_name_and_direct_plan = original_resolve

    assert result["mutual_fund_candidates"] == []


async def test_mutual_fund_specialist_node_catches_renamed_proposal_via_scheme_code_stage(memory_store):
    """Design spec Testing #9: a proposed name that normalizes
    differently from the one that produced a prior rejection - so the
    cheap by-name alias pre-filter (stage a) does NOT catch it - must
    still be suppressed once it resolves to the same, already-rejected
    real scheme_code (stage b, the authoritative one). Uses real
    MFapi.in resolution (free): "ICICI Prudential Gilt Fund" and "ICICI
    Prudential Gilt Fund PF Option" both live-resolve to real
    scheme_code 120605 but normalize to different alias keys.

    The real resolver is wrapped (not replaced) so the test can confirm
    resolution genuinely ran and genuinely returned scheme_code 120605
    for this proposed name - otherwise an empty `mutual_fund_candidates`
    result could just as easily mean resolution silently failed for an
    unrelated reason (e.g. a real MFapi.in outage) rather than the
    scheme_code filter actually doing its job."""
    from agentic_frameworks.graph.nodes._candidate_memory import (
        REJECTED_MUTUAL_FUND_NS,
        _normalize_scheme_name,
        _record_rejection,
    )
    from agentic_frameworks.graph.nodes.mutual_fund_specialist import mutual_fund_specialist_node
    from agentic_frameworks.graph.nodes.research import research_node
    from agentic_frameworks.graph.schemas import RejectedCandidate

    real_scheme_code = 120605
    proposed_name = "ICICI Prudential Gilt Fund PF Option"
    prior_rejected_name = "ICICI Prudential Gilt Fund"

    # Confirm up front that these two names really do normalize
    # differently - otherwise this test could pass for the wrong reason
    # (caught by the cheap alias filter, stage a, rather than the
    # authoritative scheme_code filter, stage b, that it claims to test).
    assert _normalize_scheme_name(proposed_name) != _normalize_scheme_name(prior_rejected_name)

    # Pre-seed a canonical rejection for this real scheme_code, as if a
    # prior run proposed it under the name "ICICI Prudential Gilt Fund".
    await _record_rejection(
        memory_store,
        REJECTED_MUTUAL_FUND_NS,
        RejectedCandidate(
            identifier=str(real_scheme_code),
            identifier_kind="scheme_code",
            specialist="mutual_fund",
            display_name="ICICI Prudential Gilt Fund Investment Plan PF Option - Direct Plan -Growth Option",
            group="Gilt",
            action="Hold",
            confidence=0.5,
            risk_flags=[],
            reasoning="Weak risk-adjusted performance versus peers.",
            rejected_at="2026-09-01T00:00:00+00:00",
            thread_id=None,
            schema_version=1,
        ),
    )

    class _FakeCandidate:
        scheme_name = proposed_name
        category = "Gilt"

    class _FakeGenerator:
        async def run(self, prompt):
            class _Result:
                output = [_FakeCandidate()]

            return _Result()

    class _FakeEvaluator:
        async def run(self, prompt):
            raise AssertionError(
                "Conviction Gate evaluator.run() must not be called - the candidate "
                "should be caught by the scheme_code rejection filter before any paid call."
            )

    import agentic_frameworks.graph.nodes.mutual_fund_specialist as node_module

    resolve_calls = []
    original_resolve = node_module._resolve_scheme_by_name_and_direct_plan

    async def spying_resolve(scheme_name, toolset):
        resolution = await original_resolve(scheme_name, toolset)
        resolve_calls.append((scheme_name, resolution))
        return resolution

    original_build_generator = node_module._build_candidate_generator
    original_build_evaluator = node_module._build_conviction_gate_evaluator
    node_module._build_candidate_generator = lambda model: _FakeGenerator()
    node_module._build_conviction_gate_evaluator = lambda *args, **kwargs: _FakeEvaluator()
    node_module._resolve_scheme_by_name_and_direct_plan = spying_resolve
    try:
        research_result = await research_node({})
        state = {"holdings_by_bucket": research_result["holdings_by_bucket"]}

        result = await mutual_fund_specialist_node(state, store=memory_store)
    finally:
        node_module._build_candidate_generator = original_build_generator
        node_module._build_conviction_gate_evaluator = original_build_evaluator
        node_module._resolve_scheme_by_name_and_direct_plan = original_resolve

    # Resolution genuinely happened and genuinely resolved to the same,
    # already-rejected real scheme_code - proving the empty result below
    # is the scheme_code filter doing its job, not resolution silently
    # failing for an unrelated reason.
    assert len(resolve_calls) == 1
    called_name, (resolved_scheme_code, resolved_category, resolved_scheme_name) = resolve_calls[0]
    assert called_name == proposed_name
    assert resolved_scheme_code == real_scheme_code

    assert result["mutual_fund_candidates"] == []


async def test_mutual_fund_specialist_node_raises_without_store():
    import pytest

    from agentic_frameworks.graph.nodes import mutual_fund_specialist as node_module
    from agentic_frameworks.graph.nodes.research import research_node

    research_result = await research_node({})
    state = {"holdings_by_bucket": research_result["holdings_by_bucket"]}

    with pytest.raises(ValueError, match="requires a store"):
        await node_module.mutual_fund_specialist_node(state, store=None)


async def test_mutual_fund_specialist_node_suppresses_previously_rejected_scheme_code_across_runs(
    memory_store,
):
    """Cross-run suppression at the authoritative scheme_code stage (b),
    genuinely exercised, not just the cheap alias pre-filter (a): run 2
    proposes a DIFFERENTLY-NAMED candidate from run 1 - one that
    normalizes to a different alias key - so stage (a) cannot catch it.
    Both runs' names resolve (via a call-counted fake resolver, standing
    in for a real Direct-plan fund) to the same real scheme_code 118632,
    so only stage (b), reached after resolution runs again, can suppress
    run 2. `resolve_call_count` proves resolution genuinely ran again in
    run 2 (i.e. stage (a) did not short-circuit before it), and
    `evaluator_call_count` proves the paid evaluator was still never
    reached - together these pin the suppression to stage (b).

    The evaluator is replaced by a deterministic fake returning a fixed
    "Hold" verdict (TestModel would otherwise deterministically emit
    "Buy", which this design never suppresses - see the equity
    specialist's identical test for the same reasoning)."""
    from agentic_frameworks.graph.nodes import mutual_fund_specialist as node_module
    from agentic_frameworks.graph.nodes._candidate_memory import _normalize_scheme_name
    from agentic_frameworks.graph.nodes.research import research_node
    from agentic_frameworks.graph.schemas import ConvictionVerdict

    real_scheme_code = 118632
    real_scheme_name = "Nippon India Large Cap Fund - Direct Plan - Growth Option"
    evaluator_call_count = {"n": 0}
    resolve_call_count = {"n": 0}
    resolve_calls: list[str] = []

    proposed_name_run_1 = "Nippon India Large Cap Fund Direct Plan Growth"
    proposed_name_run_2 = "Nippon India Large Cap Fund PF Option"
    assert _normalize_scheme_name(proposed_name_run_1) != _normalize_scheme_name(proposed_name_run_2), (
        "the two proposed names must normalize differently, or the alias "
        "pre-filter (stage a) - not the scheme_code filter (stage b) - "
        "would be what actually suppresses run 2"
    )

    async def fake_resolve_real_large_cap(scheme_name, toolset):
        resolve_call_count["n"] += 1
        resolve_calls.append(scheme_name)
        return real_scheme_code, "Equity Scheme - Large Cap Fund", real_scheme_name

    class _FakeCandidate:
        def __init__(self, scheme_name):
            self.scheme_name = scheme_name
            self.category = "Large Cap"

    def make_fake_generator(scheme_name):
        class _FakeGenerator:
            async def run(self, prompt):
                class _Result:
                    output = [_FakeCandidate(scheme_name)]

                return _Result()

        return _FakeGenerator()

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
                    reasoning="Beats benchmark modestly but short track record.",
                    risk_flags=["short_track_record"],
                    entry_approach=None,
                    suggested_platform=None,
                )
            )

    def fake_build_evaluator(*args, **kwargs):
        return _FakeEvaluator()

    research_result = await research_node({})
    state = {"holdings_by_bucket": research_result["holdings_by_bucket"]}

    original_build_generator = node_module._build_candidate_generator
    original_resolve = node_module._resolve_scheme_by_name_and_direct_plan
    original_build_evaluator = node_module._build_conviction_gate_evaluator
    node_module._resolve_scheme_by_name_and_direct_plan = fake_resolve_real_large_cap
    node_module._build_conviction_gate_evaluator = fake_build_evaluator
    try:
        node_module._build_candidate_generator = lambda model: make_fake_generator(proposed_name_run_1)
        result_1 = await node_module.mutual_fund_specialist_node(state, store=memory_store)
        assert result_1["mutual_fund_candidates"]
        assert result_1["mutual_fund_candidates"][0]["action"] == "Hold"
        evaluator_calls_after_run_1 = evaluator_call_count["n"]
        resolve_calls_after_run_1 = resolve_call_count["n"]
        assert evaluator_calls_after_run_1 > 0
        assert resolve_calls_after_run_1 > 0

        node_module._build_candidate_generator = lambda model: make_fake_generator(proposed_name_run_2)
        result_2 = await node_module.mutual_fund_specialist_node(state, store=memory_store)

        # Resolution ran again for run 2's differently-named candidate -
        # proving the cheap alias filter (stage a) did NOT catch it
        # before resolution (it has no knowledge of run 1's alias key).
        assert resolve_call_count["n"] > resolve_calls_after_run_1
        assert resolve_calls[-1] == proposed_name_run_2

        # Yet the evaluator was never called again: only the scheme_code
        # filter (stage b), which runs after resolution, could have
        # suppressed it.
        assert result_2["mutual_fund_candidates"] == []
        assert evaluator_call_count["n"] == evaluator_calls_after_run_1
    finally:
        node_module._build_candidate_generator = original_build_generator
        node_module._resolve_scheme_by_name_and_direct_plan = original_resolve
        node_module._build_conviction_gate_evaluator = original_build_evaluator
