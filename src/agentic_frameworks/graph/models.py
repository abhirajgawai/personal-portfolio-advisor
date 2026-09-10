"""LLM model wiring for specialist/Synthesizer nodes.

Single default model, no fallback list yet - kept as an easy-to-swap
named constant per CLAUDE.md's "LLM cost control" rule 1 (explicit
per-node choice, never a silent default).
"""

from __future__ import annotations

import os

from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

SPECIALIST_MODEL_NAME = "deepseek/deepseek-v4.1-flash"


def get_specialist_model() -> OpenRouterModel:
    """Build the OpenRouter-backed model used by specialist nodes."""
    api_key = os.environ["OPENROUTER_API_KEY"]
    return OpenRouterModel(SPECIALIST_MODEL_NAME, provider=OpenRouterProvider(api_key=api_key))
