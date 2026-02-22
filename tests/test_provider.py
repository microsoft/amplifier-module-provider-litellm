"""Tests for the LiteLLM provider."""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from amplifier_module_provider_litellm.provider import (
    LiteLLMProvider,
    _from_litellm_response,
    _to_litellm_messages,
    _to_litellm_tools,
)


# ---------------------------------------------------------------------------
# Provider basics
# ---------------------------------------------------------------------------

class TestProviderInit:
    def test_defaults(self):
        p = LiteLLMProvider()
        assert p.name == "litellm"
        assert p.default_model == "anthropic/claude-opus-4-6"
        assert p._timeout == 300.0
        assert p._drop_params is True
        assert p.coordinator is None

    def test_custom_config(self):
        p = LiteLLMProvider({"model": "openai/gpt-4o", "timeout": 60, "drop_params": False})
        assert p.default_model == "openai/gpt-4o"
        assert p._timeout == 60.0
        assert p._drop_params is False

    def test_coordinator_passed(self):
        coord = MagicMock()
        p = LiteLLMProvider(coordinator=coord)
        assert p.coordinator is coord


class TestProviderInfo:
    def test_get_info(self):
        p = LiteLLMProvider()
        info = p.get_info()
        assert info.id == "litellm"
        assert "tools" in info.capabilities
        assert "streaming" in info.capabilities
        assert info.credential_env_vars == []
        assert len(info.config_fields) >= 1


class TestListModels:
    @pytest.mark.asyncio
    async def test_lists_models_for_set_env_vars(self):
        controlled_env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "")}
        controlled_env["ANTHROPIC_API_KEY"] = "test"
        controlled_env["OPENAI_API_KEY"] = "test"
        with patch.dict(os.environ, controlled_env, clear=True):
            p = LiteLLMProvider()
            models = await p.list_models()
            ids = [m.id for m in models]
            assert any("anthropic" in mid for mid in ids)
            assert any("openai" in mid for mid in ids)
            assert not any("gemini" in mid for mid in ids)

    @pytest.mark.asyncio
    async def test_always_includes_default_model(self):
        with patch.dict(os.environ, {}, clear=False):
            p = LiteLLMProvider({"model": "ollama/llama3.2"})
            models = await p.list_models()
            ids = [m.id for m in models]
            assert "ollama/llama3.2" in ids


# ---------------------------------------------------------------------------
# Message serialization
# ---------------------------------------------------------------------------

class TestToLiteLLMMessages:
    def test_simple_user_message(self):
        request = MagicMock()
        msg = MagicMock()
        msg.role = "user"
        msg.content = "Hello"
        msg.tool_call_id = None
        msg.tool_calls = None
        request.messages = [msg]

        result = _to_litellm_messages(request)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_tool_message(self):
        request = MagicMock()
        msg = MagicMock()
        msg.role = "tool"
        msg.content = "result"
        msg.tool_call_id = "tc_123"
        request.messages = [msg]

        result = _to_litellm_messages(request)
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "tc_123"

    def test_assistant_with_tool_calls(self):
        request = MagicMock()
        msg = MagicMock()
        msg.role = "assistant"
        msg.content = ""
        msg.tool_call_id = None

        tc = MagicMock()
        tc.id = "tc_1"
        tc.name = "search"
        tc.tool = "search"
        tc.arguments = {"query": "test"}
        msg.tool_calls = [tc]
        request.messages = [msg]

        result = _to_litellm_messages(request)
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["function"]["name"] == "search"


class TestToLiteLLMTools:
    def test_converts_tools(self):
        request = MagicMock()
        tool = MagicMock()
        tool.name = "web_search"
        tool.description = "Search the web"
        tool.parameters = {"type": "object", "properties": {"q": {"type": "string"}}}
        tool.input_schema = None
        request.tools = [tool]

        result = _to_litellm_tools(request)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "web_search"

    def test_no_tools(self):
        request = MagicMock()
        request.tools = None
        assert _to_litellm_tools(request) is None


# ---------------------------------------------------------------------------
# Response deserialization
# ---------------------------------------------------------------------------

class TestFromLiteLLMResponse:
    def test_text_response(self):
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = "Hello world"
        choice.message.tool_calls = None
        choice.finish_reason = "stop"
        response.choices = [choice]
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 5
        response.usage.total_tokens = 15
        response.usage.prompt_tokens_details = None
        response.model = "anthropic/claude-opus-4-6"

        result = _from_litellm_response(response)
        assert result.content[0].text == "Hello world"
        assert result.metadata["model"] == "anthropic/claude-opus-4-6"
        assert result.finish_reason == "stop"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    def test_tool_call_response(self):
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = ""
        tc = MagicMock()
        tc.id = "call_123"
        tc.function.name = "web_search"
        tc.function.arguments = '{"query": "test"}'
        choice.message.tool_calls = [tc]
        choice.finish_reason = "tool_calls"
        response.choices = [choice]
        response.usage.prompt_tokens = 20
        response.usage.completion_tokens = 10
        response.usage.total_tokens = 30
        response.usage.prompt_tokens_details = None
        response.model = "openai/gpt-4o"

        result = _from_litellm_response(response)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "web_search"
        assert result.tool_calls[0].arguments == {"query": "test"}

    def test_empty_response(self):
        response = MagicMock()
        response.choices = []
        response.usage = None
        response.model = ""

        result = _from_litellm_response(response)
        assert result.content == []


# ---------------------------------------------------------------------------
# Complete (mocked litellm call)
# ---------------------------------------------------------------------------

class TestComplete:
    @pytest.mark.asyncio
    async def test_complete_calls_litellm(self):
        provider = LiteLLMProvider({"model": "openai/gpt-4o"})

        mock_response = MagicMock()
        choice = MagicMock()
        choice.message.content = "4"
        choice.message.tool_calls = None
        choice.finish_reason = "stop"
        mock_response.choices = [choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 1
        mock_response.usage.total_tokens = 11
        mock_response.usage.prompt_tokens_details = None
        mock_response.model = "openai/gpt-4o"

        request = MagicMock()
        request.model = "openai/gpt-4o"
        request.messages = []
        request.tools = None
        request.max_output_tokens = 100
        request.temperature = 0.0

        with patch("amplifier_module_provider_litellm.provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            # Patch error classes to exist on the mock
            mock_litellm.AuthenticationError = Exception
            mock_litellm.RateLimitError = Exception
            mock_litellm.ContextWindowExceededError = Exception
            mock_litellm.ContentPolicyViolationError = Exception
            mock_litellm.BadRequestError = Exception
            mock_litellm.ServiceUnavailableError = Exception
            mock_litellm.Timeout = Exception
            result = await provider.complete(request)

            mock_litellm.acompletion.assert_called_once()
            call_kwargs = mock_litellm.acompletion.call_args[1]
            assert call_kwargs["model"] == "openai/gpt-4o"
            assert call_kwargs["max_tokens"] == 100
            assert call_kwargs["temperature"] == 0.0
            assert result.content[0].text == "4"


# ---------------------------------------------------------------------------
# Parse tool calls
# ---------------------------------------------------------------------------

class TestParseToolCalls:
    def test_filters_none_arguments(self):
        provider = LiteLLMProvider()
        response = MagicMock()
        tc_good = MagicMock()
        tc_good.name = "search"
        tc_good.arguments = {"q": "test"}
        tc_bad = MagicMock()
        tc_bad.name = "broken"
        tc_bad.arguments = None
        response.tool_calls = [tc_good, tc_bad]

        result = provider.parse_tool_calls(response)
        assert len(result) == 1
        assert result[0].name == "search"

    def test_empty_tool_calls(self):
        provider = LiteLLMProvider()
        response = MagicMock()
        response.tool_calls = []
        assert provider.parse_tool_calls(response) == []

    def test_none_tool_calls(self):
        provider = LiteLLMProvider()
        response = MagicMock()
        response.tool_calls = None
        assert provider.parse_tool_calls(response) == []
