"""Tests for cost tier metadata in LiteLLM provider.

Validates:
1. list_models() adds metadata={"cost_tier": "medium"} as safe default
2. Default model also gets cost_tier
"""

import os

import pytest

from amplifier_module_provider_litellm.provider import LiteLLMProvider


class TestLiteLLMCostTier:
    """Verify all LiteLLM models get medium cost tier as default."""

    @pytest.fixture
    def provider(self):
        return LiteLLMProvider(config={"model": "anthropic/claude-opus-4-6"})

    @pytest.mark.asyncio
    async def test_default_model_has_cost_tier(self, provider):
        """Default model (always included) should have cost_tier."""
        # Clear env vars so we only get the default model
        env_vars = [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "XAI_API_KEY",
            "GROQ_API_KEY",
            "OPENROUTER_API_KEY",
        ]
        saved = {}
        for var in env_vars:
            saved[var] = os.environ.pop(var, None)

        try:
            models = await provider.list_models()
            assert len(models) >= 1
            for model in models:
                assert "cost_tier" in model.metadata, f"{model.id} missing cost_tier"
                assert model.metadata["cost_tier"] == "medium"
        finally:
            for var, val in saved.items():
                if val is not None:
                    os.environ[var] = val

    @pytest.mark.asyncio
    async def test_env_models_have_cost_tier(self, provider):
        """When env vars are set, listed models should have cost_tier."""
        # Set a fake env var to trigger model listing
        saved = os.environ.get("ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = "test-key"

        try:
            models = await provider.list_models()
            anthropic_models = [m for m in models if "anthropic" in m.id]
            assert len(anthropic_models) >= 1
            for model in anthropic_models:
                assert model.metadata["cost_tier"] == "medium"
        finally:
            if saved is not None:
                os.environ["ANTHROPIC_API_KEY"] = saved
            else:
                os.environ.pop("ANTHROPIC_API_KEY", None)
