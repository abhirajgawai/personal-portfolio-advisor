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
