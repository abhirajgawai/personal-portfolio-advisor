"""Tests for LLM model wiring."""

from __future__ import annotations


def test_get_specialist_model_uses_configured_model_name(monkeypatch):
    from agentic_frameworks.graph import models

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

    model = models.get_specialist_model()

    assert models.SPECIALIST_MODEL_NAME == "deepseek/deepseek-v4.1-flash"
    assert models.BACKUP_MODEL_NAME == "z-ai/glm-5.3-flash"
    wrapped_model_names = [m.model_name for m in model.models]
    assert wrapped_model_names == [models.SPECIALIST_MODEL_NAME, models.BACKUP_MODEL_NAME]
