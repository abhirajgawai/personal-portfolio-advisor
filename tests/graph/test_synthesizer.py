"""Tests for the Recommendation Synthesizer node.

Real identifiers only (no fabricated tickers/ISINs/scheme codes), per
CLAUDE.md and the design spec's Testing section: XLV/JPM/RELIANCE.NS are
real, currently-traded instruments already used elsewhere in this
project's real-model tests; INF179K01WN9 is a real, currently-held India
mutual fund ISIN (see test_mutual_fund_helpers.py); 120605/118989 are
real MFapi.in Gilt-fund scheme codes already resolved live elsewhere in
this suite (test_mutual_fund_specialist.py).

`real_research_risk_state` (conftest.py) supplies real holdings/
allocation/risk data; the four specialist branches are hand-built
`CandidateEvaluation`-shaped dicts (real identifiers, fabricated verdicts)
rather than real LLM runs, since a real specialist run would cost money -
same reject-early-on-cost approach the rest of this suite already takes.
"""

from __future__ import annotations

import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

REAL_INDEX_FUND_ISIN = "INF179K01WN9"
REAL_INDEX_FUND_NAME = "HDFC BSE Sensex Index Fund - Direct Plan - Growth Option"


def _existing_fund_evaluation(**overrides) -> dict:
    defaults = dict(
        identifier=REAL_INDEX_FUND_ISIN,
        identifier_kind="isin",
        display_name=REAL_INDEX_FUND_NAME,
        action="Hold",
        confidence=0.6,
        reasoning="Tracks its index closely; no clear reason to change allocation.",
        risk_flags=["short_track_record"],
        entry_approach=None,
        suggested_platform=None,
    )
    defaults.update(overrides)
    return defaults


def _us_buy_candidate(**overrides) -> dict:
    defaults = dict(
        identifier="XLV",
        identifier_kind="ticker",
        display_name="Health Care Select Sector SPDR Fund",
        action="Buy",
        confidence=0.7,
        reasoning="Fills the Health Care gap with reasonable valuation.",
        risk_flags=[],
        entry_approach="Staggered",
        suggested_platform="moomoo",
    )
    defaults.update(overrides)
    return defaults


def _us_hold_candidate(**overrides) -> dict:
    defaults = dict(
        identifier="JPM",
        identifier_kind="ticker",
        display_name="JPMorgan Chase & Co.",
        action="Hold",
        confidence=0.4,
        reasoning="Richly valued right now relative to its own history.",
        risk_flags=["valuation_rich"],
        entry_approach=None,
        suggested_platform=None,
    )
    defaults.update(overrides)
    return defaults


def _india_hold_candidate(**overrides) -> dict:
    defaults = dict(
        identifier="RELIANCE.NS",
        identifier_kind="ticker",
        display_name="Reliance Industries Limited",
        action="Hold",
        confidence=0.5,
        reasoning="No clear gap fit this cycle.",
        risk_flags=[],
        entry_approach=None,
        suggested_platform=None,
    )
    defaults.update(overrides)
    return defaults


def _mf_hold_candidate(**overrides) -> dict:
    defaults = dict(
        identifier="118989",
        identifier_kind="scheme_code",
        display_name="Some Real Gilt Fund - Direct Plan - Growth",
        action="Hold",
        confidence=0.5,
        reasoning="Weak risk-adjusted performance versus peers.",
        risk_flags=["short_track_record"],
        entry_approach=None,
        suggested_platform=None,
    )
    defaults.update(overrides)
    return defaults


def _full_state(
    real_research_risk_state: dict,
    *,
    us_equity_etf_candidates=None,
    india_equity_candidates=None,
    mutual_fund_candidates=None,
    existing_fund_evaluations=None,
) -> dict:
    return {
        **real_research_risk_state,
        "us_equity_etf_candidates": (
            [_us_buy_candidate(), _us_hold_candidate()]
            if us_equity_etf_candidates is None
            else us_equity_etf_candidates
        ),
        "india_equity_candidates": (
            [_india_hold_candidate()] if india_equity_candidates is None else india_equity_candidates
        ),
        "mutual_fund_candidates": (
            [_mf_hold_candidate()] if mutual_fund_candidates is None else mutual_fund_candidates
        ),
        "existing_fund_evaluations": (
            [_existing_fund_evaluation()] if existing_fund_evaluations is None else existing_fund_evaluations
        ),
    }


def _happy_path_test_model() -> TestModel:
    return TestModel(
        custom_output_args={
            "portfolio_assessment": "Diversified but real-estate heavy.",
            "allocation_notes": ["Real estate is a large share of the portfolio."],
            "existing_holdings_guidance": [
                {
                    "identifier": REAL_INDEX_FUND_ISIN,
                    "action": "Hold",
                    "rationale": "Tracks its index closely.",
                    "risk_flags": ["short_track_record"],
                    "confidence": 0.6,
                }
            ],
            "new_buy_suggestions": [
                {
                    "identifier": "XLV",
                    "rationale": "Fills the Health Care gap.",
                    "gap_addressed": "Health Care",
                    "position_size_pct_of_portfolio": 1.5,
                    "entry_approach": "Staggered",
                    "suggested_platform": "moomoo",
                    "risk_flags": [],
                    "confidence": 0.7,
                }
            ],
            "no_new_buy_reason": None,
        }
    )


def _function_model_returning(output_args: dict) -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_name = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=output_args)])

    return FunctionModel(fn)


def _capturing_function_model(output_args: dict, captured_prompts: list[str]) -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for message in messages:
            for part in message.parts:
                text = getattr(part, "content", None)
                if isinstance(text, str):
                    captured_prompts.append(text)
        tool_name = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=output_args)])

    return FunctionModel(fn)


# --- pure helpers ---


def test_select_buy_candidates_filters_to_buy_only_and_tags_specialist(real_research_risk_state):
    from agentic_frameworks.graph.nodes.synthesizer import _select_buy_candidates

    state = _full_state(real_research_risk_state)
    buy_candidates = _select_buy_candidates(state)

    assert len(buy_candidates) == 1
    assert buy_candidates[0]["identifier"] == "XLV"
    assert buy_candidates[0]["specialist"] == "us_equity"


def test_select_buy_candidates_returns_empty_when_no_buy_verdicts(real_research_risk_state):
    from agentic_frameworks.graph.nodes.synthesizer import _select_buy_candidates

    state = _full_state(real_research_risk_state, us_equity_etf_candidates=[_us_hold_candidate()])
    assert _select_buy_candidates(state) == []


def test_build_subject_index_includes_existing_and_new_buy_sides(real_research_risk_state):
    from agentic_frameworks.graph.nodes.synthesizer import _build_subject_index, _select_buy_candidates

    state = _full_state(real_research_risk_state)
    buy_candidates = _select_buy_candidates(state)
    index = _build_subject_index(state, buy_candidates)

    assert index[REAL_INDEX_FUND_ISIN].side == "existing"
    assert index[REAL_INDEX_FUND_ISIN].specialist is None
    assert index["XLV"].side == "new_buy"
    assert index["XLV"].specialist == "us_equity"
    # Non-Buy candidates never enter the index at all - they must not be
    # nameable as either an existing subject or a new-buy one.
    assert "JPM" not in index
    assert "RELIANCE.NS" not in index


def test_compute_coverage_against_real_holdings(real_research_risk_state):
    from agentic_frameworks.graph.nodes.synthesizer import _build_subject_index, _compute_coverage

    state = _full_state(real_research_risk_state)
    index = _build_subject_index(state, [])
    coverage = _compute_coverage(state["holdings"], index)

    assert coverage.holdings_total == len(state["holdings"])
    assert coverage.holdings_individually_assessed == 1
    assert coverage.holdings_individually_assessed < coverage.holdings_total
    assert "real_estate/IN" not in coverage.unassessed_buckets
    assert coverage.unassessed_buckets


def test_derive_platforms_uses_real_holdings(real_research_risk_state):
    from agentic_frameworks.graph.nodes.synthesizer import _derive_platforms

    platforms = _derive_platforms(real_research_risk_state["holdings"])

    # Real platform names, live-derived - see design spec Background.
    assert "moomoo" in platforms
    assert "groww" in platforms


# --- node happy path ---


async def test_synthesizer_node_happy_path_binds_identity_and_persists_recommendation(
    real_research_risk_state, memory_store
):
    from agentic_frameworks.graph.nodes import synthesizer as synth_module

    state = _full_state(real_research_risk_state)

    original = synth_module.get_synthesizer_model
    synth_module.get_synthesizer_model = _happy_path_test_model
    try:
        result = await synth_module.synthesizer_node(state, store=memory_store)
    finally:
        synth_module.get_synthesizer_model = original

    report = result["recommendation"]
    assert report["existing_holdings_guidance"][0]["identifier"] == REAL_INDEX_FUND_ISIN
    assert report["existing_holdings_guidance"][0]["display_name"] == REAL_INDEX_FUND_NAME
    assert report["existing_holdings_guidance"][0]["identifier_kind"] == "isin"
    assert report["new_buy_suggestions"][0]["identifier"] == "XLV"
    assert report["new_buy_suggestions"][0]["specialist"] == "us_equity"
    assert report["no_new_buy_reason"] is None
    assert report["coverage"]["holdings_total"] == len(state["holdings"])
    assert report["schema_version"] == 1


async def test_synthesizer_node_raises_without_store(real_research_risk_state):
    from agentic_frameworks.graph.nodes.synthesizer import synthesizer_node

    state = _full_state(real_research_risk_state)

    with pytest.raises(ValueError, match="requires a store"):
        await synthesizer_node(state, store=None)


# --- identifier contract (design spec Architecture point 3 / Testing item 4) ---


async def test_synthesizer_node_retries_then_raises_on_unknown_identifier(
    real_research_risk_state, memory_store
):
    from agentic_frameworks.graph.nodes import synthesizer as synth_module

    state = _full_state(real_research_risk_state)
    bad_output = {
        "portfolio_assessment": "x",
        "allocation_notes": [],
        "existing_holdings_guidance": [],
        "new_buy_suggestions": [
            {
                "identifier": "NOTAREALTICKER",
                "rationale": "hallucinated",
                "gap_addressed": "Health Care",
                "position_size_pct_of_portfolio": 1.0,
                "entry_approach": "Staggered",
                "suggested_platform": "moomoo",
                "risk_flags": [],
                "confidence": 0.5,
            }
        ],
        "no_new_buy_reason": None,
    }

    original = synth_module.get_synthesizer_model
    synth_module.get_synthesizer_model = lambda: _function_model_returning(bad_output)
    try:
        with pytest.raises(UnexpectedModelBehavior):
            await synth_module.synthesizer_node(state, store=memory_store)
    finally:
        synth_module.get_synthesizer_model = original


async def test_synthesizer_node_raises_on_cross_side_contamination(real_research_risk_state, memory_store):
    """An existing-holdings identifier must never validate as a new-buy
    suggestion, even though it's a real, known identifier this run saw."""
    from agentic_frameworks.graph.nodes import synthesizer as synth_module

    state = _full_state(real_research_risk_state)
    bad_output = {
        "portfolio_assessment": "x",
        "allocation_notes": [],
        "existing_holdings_guidance": [],
        "new_buy_suggestions": [
            {
                "identifier": REAL_INDEX_FUND_ISIN,
                "rationale": "wrong side",
                "gap_addressed": "Health Care",
                "position_size_pct_of_portfolio": 1.0,
                "entry_approach": "Staggered",
                "suggested_platform": "moomoo",
                "risk_flags": [],
                "confidence": 0.5,
            }
        ],
        "no_new_buy_reason": None,
    }

    original = synth_module.get_synthesizer_model
    synth_module.get_synthesizer_model = lambda: _function_model_returning(bad_output)
    try:
        with pytest.raises(UnexpectedModelBehavior):
            await synth_module.synthesizer_node(state, store=memory_store)
    finally:
        synth_module.get_synthesizer_model = original


async def test_synthesizer_node_raises_on_unknown_platform(real_research_risk_state, memory_store):
    from agentic_frameworks.graph.nodes import synthesizer as synth_module

    state = _full_state(real_research_risk_state)
    bad_output = {
        "portfolio_assessment": "x",
        "allocation_notes": [],
        "existing_holdings_guidance": [],
        "new_buy_suggestions": [
            {
                "identifier": "XLV",
                "rationale": "Fills the Health Care gap.",
                "gap_addressed": "Health Care",
                "position_size_pct_of_portfolio": 1.0,
                "entry_approach": "Staggered",
                "suggested_platform": "robinhood",
                "risk_flags": [],
                "confidence": 0.5,
            }
        ],
        "no_new_buy_reason": None,
    }

    original = synth_module.get_synthesizer_model
    synth_module.get_synthesizer_model = lambda: _function_model_returning(bad_output)
    try:
        with pytest.raises(UnexpectedModelBehavior):
            await synth_module.synthesizer_node(state, store=memory_store)
    finally:
        synth_module.get_synthesizer_model = original


async def test_synthesizer_node_raises_when_aggregate_capital_cap_exceeded(
    real_research_risk_state, memory_store
):
    """MAX_NEW_CAPITAL_PCT (aggregate, code-enforced) rejects a set whose
    total exceeds it even though every individual suggestion is within
    MAX_NEW_POSITION_PCT (schema-enforced)."""
    from agentic_frameworks.graph.nodes import synthesizer as synth_module
    from agentic_frameworks.graph.schemas import MAX_NEW_CAPITAL_PCT, MAX_NEW_POSITION_PCT

    state = _full_state(
        real_research_risk_state,
        us_equity_etf_candidates=[
            _us_buy_candidate(identifier="XLV"),
            _us_buy_candidate(identifier="JPM", display_name="JPMorgan Chase & Co."),
            _us_buy_candidate(identifier="IBM", display_name="International Business Machines Corporation"),
        ],
    )
    # Three suggestions, each individually within MAX_NEW_POSITION_PCT, whose
    # sum exceeds MAX_NEW_CAPITAL_PCT.
    per_suggestion_pct = min(MAX_NEW_POSITION_PCT, MAX_NEW_CAPITAL_PCT) - 0.01
    assert per_suggestion_pct * 3 > MAX_NEW_CAPITAL_PCT
    bad_output = {
        "portfolio_assessment": "x",
        "allocation_notes": [],
        "existing_holdings_guidance": [],
        "new_buy_suggestions": [
            {
                "identifier": identifier,
                "rationale": "Also worth adding.",
                "gap_addressed": "Health Care",
                "position_size_pct_of_portfolio": per_suggestion_pct,
                "entry_approach": "Staggered",
                "suggested_platform": "moomoo",
                "risk_flags": [],
                "confidence": 0.5,
            }
            for identifier in ("XLV", "JPM", "IBM")
        ],
        "no_new_buy_reason": None,
    }

    original = synth_module.get_synthesizer_model
    synth_module.get_synthesizer_model = lambda: _function_model_returning(bad_output)
    try:
        with pytest.raises(UnexpectedModelBehavior):
            await synth_module.synthesizer_node(state, store=memory_store)
    finally:
        synth_module.get_synthesizer_model = original


# --- avoid-is-the-default (design spec Testing item 5) ---


async def test_synthesizer_node_avoid_is_default_when_every_candidate_is_hold(
    real_research_risk_state, memory_store
):
    from agentic_frameworks.graph.nodes import synthesizer as synth_module

    state = _full_state(
        real_research_risk_state,
        us_equity_etf_candidates=[_us_hold_candidate()],
        india_equity_candidates=[_india_hold_candidate()],
        mutual_fund_candidates=[_mf_hold_candidate()],
    )
    captured_prompts: list[str] = []
    all_hold_output = {
        "portfolio_assessment": "Nothing compelling this cycle.",
        "allocation_notes": [],
        "existing_holdings_guidance": [],
        "new_buy_suggestions": [],
        "no_new_buy_reason": "placeholder - overwritten deterministically in code",
    }

    original = synth_module.get_synthesizer_model
    synth_module.get_synthesizer_model = lambda: _capturing_function_model(all_hold_output, captured_prompts)
    try:
        result = await synth_module.synthesizer_node(state, store=memory_store)
    finally:
        synth_module.get_synthesizer_model = original

    report = result["recommendation"]
    assert report["new_buy_suggestions"] == []
    assert report["no_new_buy_reason"] is not None
    assert "No new-buy candidate cleared the Conviction Gate" in report["no_new_buy_reason"]

    full_prompt = "\n".join(captured_prompts)
    assert "New-buy candidates that cleared the Conviction Gate" not in full_prompt


# --- minimal context (design spec Testing item 6) ---


async def test_synthesizer_node_prompt_excludes_raw_holdings_and_non_buy_full_reasoning(
    real_research_risk_state, memory_store
):
    from agentic_frameworks.graph.nodes import synthesizer as synth_module

    state = _full_state(real_research_risk_state)
    captured_prompts: list[str] = []

    original = synth_module.get_synthesizer_model
    synth_module.get_synthesizer_model = lambda: _capturing_function_model(
        {
            "portfolio_assessment": "x",
            "allocation_notes": [],
            "existing_holdings_guidance": [],
            "new_buy_suggestions": [],
            "no_new_buy_reason": "placeholder",
        },
        captured_prompts,
    )
    try:
        await synth_module.synthesizer_node(state, store=memory_store)
    finally:
        synth_module.get_synthesizer_model = original

    full_prompt = "\n".join(captured_prompts)
    # The non-Buy candidates' short one-line summary appears...
    assert "JPMorgan Chase & Co." in full_prompt
    # ...but their full reasoning text does not (cost rule 4 - one line
    # each, not the full verdict).
    assert "Richly valued right now relative to its own history." not in full_prompt
    assert "No clear gap fit this cycle." not in full_prompt
    # No raw per-holding dump of the ~45+ row holdings list: the raw
    # dict's own field names never appear verbatim, since every number
    # that reaches the prompt goes through a derived summary instead.
    assert "market_value_base" not in full_prompt
    assert "updated_at" not in full_prompt
    assert "quantity" not in full_prompt
