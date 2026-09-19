"""LLM model wiring for specialist/Synthesizer nodes.

Named constants per CLAUDE.md's "LLM cost control" rule 1 (explicit
per-node choice, never a silent default). BACKUP_MODEL_NAME exists
because deepseek/deepseek-v4.1-flash's shared free-tier OpenRouter
pool returned repeated 429 "engine_overloaded" errors during Phase 4's
real-model verification (2026-09-10) - a genuine upstream capacity
issue, not a bug in this code. FallbackModel's default fallback_on
(ModelAPIError) covers this: ModelHTTPError (what a 429 raises as) is
a ModelAPIError subclass, confirmed live against the installed
pydantic-ai version.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

load_dotenv()

SPECIALIST_MODEL_NAME = "deepseek/deepseek-v4.1-flash"
BACKUP_MODEL_NAME = "z-ai/glm-5.3-flash"


def get_specialist_model() -> FallbackModel:
    """Build the OpenRouter-backed model used by specialist nodes.

    Tries SPECIALIST_MODEL_NAME first; falls back to BACKUP_MODEL_NAME
    only on a real API error (rate limit, provider outage, etc.) - not
    a silent default, an explicit two-model list per cost control rule 1.
    """
    api_key = os.environ["OPENROUTER_API_KEY"]
    provider = OpenRouterProvider(api_key=api_key)
    primary = OpenRouterModel(SPECIALIST_MODEL_NAME, provider=provider)
    backup = OpenRouterModel(BACKUP_MODEL_NAME, provider=provider)
    return FallbackModel(primary, backup)


# Synthesizer's own, separately-chosen model list (never inherited from the
# specialists' FallbackModel by convenience - CLAUDE.md cost control rule 1).
#
# REVISED 2026-09-19 after the real-model gate failed twice. The original
# chain was free-first: nvidia/nemotron-3-super-120b-a12b:free ->
# deepseek/deepseek-v4-flash-0731:free -> z-ai/glm-5.3-flash (paid tail).
# It never once reached that paid tail, and could not have, for a reason
# confirmed by reading the installed FallbackModel's source rather than
# assuming:
#
#   FallbackModel.request() only advances to the next model when
#   model.request() itself raises an exception its fallback_on handler
#   accepts (default: ModelAPIError), or when an explicit response handler
#   rejects a returned ModelResponse. Output-validation retries live a
#   layer ABOVE that: pydantic-ai's agent graph counts them in
#   _agent_graph.py's consume_output_retry() and, on exhaustion, raises
#   UnexpectedModelBehavior("Exceeded maximum output retries (N)") - which
#   subclasses AgentRunError, NOT ModelAPIError, and is raised outside
#   FallbackModel entirely. So a model that answers HTTP-200 with JSON that
#   won't validate burns Agent(retries=2) alone and kills the run; the rest
#   of the chain is never consulted.
#
# That makes "put a reliable model later in the chain" worthless as a
# reliability strategy for this failure mode - the ONLY thing that matters
# is which model sits in the primary slot. Hence the revision below.
#
# The tradeoff, stated explicitly per cost control rule 1 (Abhi delegated
# this specific call, so the reasoning is recorded here rather than left
# implicit): the free reasoning model costs $0/run but produced malformed,
# unparseable structured output twice out of two real attempts on this
# schema (9 nested existing-holdings entries + new-buy entries) - it spends
# most of its budget on internal reasoning and emits a compressed, garbled
# final answer under the schema's brevity caps. z-ai/glm-5.3-flash costs
# ~$0.002/run (~$0.06/month at one Temporal-scheduled run a day) and has
# been the proven structured-output workhorse in this project since the
# Phase 4/5 specialists. Paying ~6 paise a run to make the terminal node of
# the graph actually return a report is the right trade: a $0 model that
# fails is infinitely more expensive than a $0.002 model that works,
# because every upstream specialist call in that run was paid for and
# thrown away with it.
SYNTHESIZER_MODEL_NAME = "z-ai/glm-5.3-flash"
SYNTHESIZER_BACKUP_MODEL_NAME = "deepseek/deepseek-v4.1-flash"


def get_synthesizer_model() -> FallbackModel:
    """Build the OpenRouter-backed model used by the Synthesizer node.

    Two-model FallbackModel chain, both paid and both already proven on
    structured output in this project: SYNTHESIZER_MODEL_NAME (primary,
    ~$0.002/run) -> SYNTHESIZER_BACKUP_MODEL_NAME (~$0.004/run). The
    fallback here buys exactly one thing and no more - survival of a real
    API-level outage on the primary (ModelAPIError, e.g. a 429), the same
    pattern Phase 4 proved live. It deliberately does NOT try to paper over
    a model that returns unvalidatable output, because FallbackModel
    structurally cannot do that (see the module comment above); schema
    reliability is bought by the primary-slot choice instead.
    """
    api_key = os.environ["OPENROUTER_API_KEY"]
    provider = OpenRouterProvider(api_key=api_key)
    primary = OpenRouterModel(SYNTHESIZER_MODEL_NAME, provider=provider)
    backup = OpenRouterModel(SYNTHESIZER_BACKUP_MODEL_NAME, provider=provider)
    return FallbackModel(primary, backup)
