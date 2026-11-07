"""Truthful configured/manual LiteLLM inventory; no env-key model invention."""

from unittest.mock import AsyncMock, patch

import pytest
from amplifier_core.llm_errors import ProviderUnavailableError
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_litellm.provider import LiteLLMProvider
from amplifier_module_provider_litellm._model_listing import ModelFilters


@pytest.mark.asyncio
async def test_configured_router_names_defaults_and_overrides():
    entries = [
        {
            "model_name": "Team/Custom:Q4",
            "litellm_params": {"model": "hidden-backend", "api_key": "SECRET"},
        },
        {"model_name": "production-chat"},
        "legacy",
    ]
    filters = {"include": ["Team/*"]}
    p = LiteLLMProvider({"model_list": entries, "model_filters": filters})
    entries.clear()
    filters["include"].clear()
    assert [m.id for m in await p.list_models()] == ["Team/Custom:Q4"]
    assert [m.id for m in await p.list_models(filters=None)] == ["Team/Custom:Q4"]
    assert [m.id for m in await p.list_models(filters={})] == [
        "Team/Custom:Q4",
        "legacy",
        "production-chat",
    ]
    assert [
        m.id
        for m in await p.list_models(
            filters={"include": ["*"], "exclude": ["TEAM/*", "legacy"]}
        )
    ] == ["production-chat"]
    assert await p.list_models(filters={"include": []}) == []
    info = p.get_model_list_info()
    assert info["source"] == "configured" and info["override"] == "replace"
    assert "SECRET" not in str(info)
    info["default_filters"]["include"].clear()
    info["filters_schema"]["properties"].clear()
    assert p.get_model_list_info()["default_filters"]["include"] == ["Team/*"]
    assert "include" in p.get_model_list_info()["filters_schema"]["properties"]
    assert all(
        m.context_window == 0 and m.max_output_tokens == 0 and not m.capabilities
        for m in await p.list_models()
    )


@pytest.mark.asyncio
async def test_keys_api_base_default_and_empty_are_not_false_inventory(monkeypatch):
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "XAI_API_KEY",
    ):
        monkeypatch.setenv(key, "not-a-real-key")
    p = LiteLLMProvider({"api_base": "https://proxy.invalid"})
    with pytest.raises(ProviderUnavailableError):
        await p.list_models()
    assert p.get_model_list_info()["source"] == "unsupported"
    assert await LiteLLMProvider({"model_list": []}).list_models() == []
    p = LiteLLMProvider({"default_model": "exact-manual"})
    assert [m.id for m in await p.list_models()] == ["exact-manual"]
    assert p.get_model_list_info()["source"] == "configured"


@pytest.mark.asyncio
async def test_hidden_explicit_inference_unchanged():
    from litellm import ModelResponse

    p = LiteLLMProvider(
        {
            "model_list": ["recommended"],
            "model_filters": {"include": []},
            "use_streaming": False,
            "extra_request_params": {"top_p": 0.4},
        }
    )
    response = ModelResponse(
        model="openai/hidden-alias",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )
    with patch(
        "amplifier_module_provider_litellm.provider.litellm.acompletion",
        AsyncMock(return_value=response),
    ) as call:
        assert await p.list_models() == []
        await p.complete(
            ChatRequest(
                model="openai/hidden-alias",
                messages=[Message(role="user", content="test")],
            )
        )
        assert call.await_args.kwargs["model"] == "openai/hidden-alias"
        assert call.await_args.kwargs["top_p"] == 0.4


@pytest.mark.parametrize("value", [None, "x", [{}], [1], [" "]])
def test_invalid_model_list(value):
    with pytest.raises((TypeError, ValueError)):
        LiteLLMProvider({"model_list": value})


@pytest.mark.parametrize(
    "value",
    [None, [], {"include": "*"}, {"exclude": [1]}, {"hide_dated": 1}, {"unknown": []}],
)
def test_invalid_filters(value):
    with pytest.raises((TypeError, ValueError)):
        LiteLLMProvider({"model_filters": value})


def test_common_vectors():
    fixture = ["Custom:Q4", "Custom:Q8", "future", "future-2099-01-01"]
    assert ModelFilters.parse({}).select(fixture) == sorted(fixture)
    assert ModelFilters.parse({"include": ["CUSTOM:*"], "exclude": ["*:q8"]}).select(
        fixture
    ) == ["Custom:Q4"]
    assert ModelFilters.parse({"hide_dated": True}).select(fixture) == [
        "Custom:Q4",
        "Custom:Q8",
        "future",
    ]
    assert ModelFilters.parse({"include": []}).select(fixture) == []
