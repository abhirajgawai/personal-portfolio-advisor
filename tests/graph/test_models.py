"""Tests for LLM model wiring."""

from __future__ import annotations


def test_get_specialist_model_uses_configured_model_name(monkeypatch):
    from agentic_frameworks.graph import models

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

    model = models.get_specialist_model()

    assert model.model_name == models.SPECIALIST_MODEL_NAME
    assert models.SPECIALIST_MODEL_NAME == "deepseek/deepseek-v4.1-flash"
