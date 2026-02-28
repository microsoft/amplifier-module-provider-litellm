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
        p = LiteLLMProvider(
            {"model": "openai/gpt-4o", "timeout": 60, "drop_params": False}
        )
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
        controlled_env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
        }
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

    def test_assistant_with_pydantic_content_blocks(self):
        """Content blocks stored as Pydantic objects must be serialized to dicts.

        Regression test: the loop module stores assistant message content as
        TextBlock/ToolCallBlock Pydantic models. Without serialization, litellm
        crashes with "'TextBlock' object has no attribute 'get'".
        """
        from amplifier_core.message_models import TextBlock

        request = MagicMock()
        msg = MagicMock()
        msg.role = "assistant"
        msg.content = [TextBlock(text="Hello world")]
        msg.tool_call_id = None
        msg.tool_calls = None
        request.messages = [msg]

        result = _to_litellm_messages(request)
        content = result[0]["content"]
        assert isinstance(content, list)
        assert isinstance(content[0], dict)
        assert content[0]["text"] == "Hello world"


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

        with patch(
            "amplifier_module_provider_litellm.provider.litellm"
        ) as mock_litellm:
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


# ---------------------------------------------------------------------------
# Error message uses json.dumps(e.body) instead of str(e)
# ---------------------------------------------------------------------------


class _FakeError(Exception):
    """Exception with a .body attribute, mimicking litellm error structure."""

    def __init__(self, message, body=None):
        super().__init__(message)
        self.body = body


def _make_provider_and_request():
    """Helper: build a LiteLLMProvider and a minimal ChatRequest mock."""
    provider = LiteLLMProvider({"model": "openai/gpt-4o"})
    request = MagicMock()
    request.model = "openai/gpt-4o"
    request.messages = []
    request.tools = None
    request.max_output_tokens = 100
    request.temperature = 0.0
    return provider, request


def _patch_litellm_error_classes(mock_litellm):
    """Assign all litellm error classes so the except chain works.

    Every class is set to a unique type that will never match, so the
    caller can then override the one it actually wants to trigger.
    """

    class _Never(Exception):
        pass

    mock_litellm.AuthenticationError = _Never
    mock_litellm.RateLimitError = _Never
    mock_litellm.ContextWindowExceededError = _Never
    mock_litellm.ContentPolicyViolationError = _Never
    mock_litellm.BadRequestError = _Never
    mock_litellm.ServiceUnavailableError = _Never
    mock_litellm.Timeout = _Never
    return mock_litellm


class TestErrorMessageUsesJsonBody:
    """Verify that all 7 targeted except blocks use json.dumps(e.body)."""

    @pytest.mark.asyncio
    async def test_auth_error_uses_json_body(self):
        provider, request = _make_provider_and_request()
        body = {"message": "invalid api key", "type": "auth_error", "code": 401}

        with patch(
            "amplifier_module_provider_litellm.provider.litellm"
        ) as mock_litellm:
            _patch_litellm_error_classes(mock_litellm)
            AuthErr = type("AuthenticationError", (_FakeError,), {})
            mock_litellm.AuthenticationError = AuthErr
            mock_litellm.acompletion = AsyncMock(
                side_effect=AuthErr("str repr", body=body)
            )

            from amplifier_core.llm_errors import (
                AuthenticationError as KernelAuthenticationError,
            )

            with pytest.raises(KernelAuthenticationError) as exc_info:
                await provider.complete(request)

            assert json.dumps(body) in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_rate_limit_error_uses_json_body(self):
        provider, request = _make_provider_and_request()
        body = {"message": "rate limited", "type": "rate_limit", "code": 429}

        with patch(
            "amplifier_module_provider_litellm.provider.litellm"
        ) as mock_litellm:
            _patch_litellm_error_classes(mock_litellm)
            RateErr = type("RateLimitError", (_FakeError,), {})
            mock_litellm.RateLimitError = RateErr
            mock_litellm.acompletion = AsyncMock(
                side_effect=RateErr("str repr", body=body)
            )

            from amplifier_core.llm_errors import RateLimitError as KernelRateLimitError

            with pytest.raises(KernelRateLimitError) as exc_info:
                await provider.complete(request)

            assert json.dumps(body) in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_context_window_error_uses_json_body(self):
        provider, request = _make_provider_and_request()
        body = {"message": "context window exceeded", "code": 400}

        with patch(
            "amplifier_module_provider_litellm.provider.litellm"
        ) as mock_litellm:
            _patch_litellm_error_classes(mock_litellm)
            CtxErr = type("ContextWindowExceededError", (_FakeError,), {})
            mock_litellm.ContextWindowExceededError = CtxErr
            mock_litellm.acompletion = AsyncMock(
                side_effect=CtxErr("str repr", body=body)
            )

            from amplifier_core.llm_errors import (
                ContextLengthError as KernelContextLengthError,
            )

            with pytest.raises(KernelContextLengthError) as exc_info:
                await provider.complete(request)

            assert json.dumps(body) in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_content_policy_error_uses_json_body(self):
        provider, request = _make_provider_and_request()
        body = {"message": "content policy violation", "code": 400}

        with patch(
            "amplifier_module_provider_litellm.provider.litellm"
        ) as mock_litellm:
            _patch_litellm_error_classes(mock_litellm)
            PolicyErr = type("ContentPolicyViolationError", (_FakeError,), {})
            mock_litellm.ContentPolicyViolationError = PolicyErr
            mock_litellm.acompletion = AsyncMock(
                side_effect=PolicyErr("str repr", body=body)
            )

            from amplifier_core.llm_errors import (
                ContentFilterError as KernelContentFilterError,
            )

            with pytest.raises(KernelContentFilterError) as exc_info:
                await provider.complete(request)

            assert json.dumps(body) in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_bad_request_error_uses_json_body(self):
        provider, request = _make_provider_and_request()
        body = {"message": "bad request", "code": 400}

        with patch(
            "amplifier_module_provider_litellm.provider.litellm"
        ) as mock_litellm:
            _patch_litellm_error_classes(mock_litellm)
            BadReqErr = type("BadRequestError", (_FakeError,), {})
            mock_litellm.BadRequestError = BadReqErr
            mock_litellm.acompletion = AsyncMock(
                side_effect=BadReqErr("str repr", body=body)
            )

            from amplifier_core.llm_errors import (
                InvalidRequestError as KernelInvalidRequestError,
            )

            with pytest.raises(KernelInvalidRequestError) as exc_info:
                await provider.complete(request)

            assert json.dumps(body) in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_service_unavailable_error_uses_json_body(self):
        provider, request = _make_provider_and_request()
        body = {"message": "service unavailable", "code": 503}

        with patch(
            "amplifier_module_provider_litellm.provider.litellm"
        ) as mock_litellm:
            _patch_litellm_error_classes(mock_litellm)
            SvcErr = type("ServiceUnavailableError", (_FakeError,), {})
            mock_litellm.ServiceUnavailableError = SvcErr
            mock_litellm.acompletion = AsyncMock(
                side_effect=SvcErr("str repr", body=body)
            )

            from amplifier_core.llm_errors import (
                ProviderUnavailableError as KernelProviderUnavailableError,
            )

            with pytest.raises(KernelProviderUnavailableError) as exc_info:
                await provider.complete(request)

            assert json.dumps(body) in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generic_exception_uses_json_body(self):
        provider, request = _make_provider_and_request()
        body = {"message": "unexpected error", "code": 500}

        with patch(
            "amplifier_module_provider_litellm.provider.litellm"
        ) as mock_litellm:
            _patch_litellm_error_classes(mock_litellm)
            exc = _FakeError("str repr", body=body)
            mock_litellm.acompletion = AsyncMock(side_effect=exc)

            from amplifier_core.llm_errors import LLMError as KernelLLMError

            with pytest.raises(KernelLLMError) as exc_info:
                await provider.complete(request)

            assert json.dumps(body) in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_auth_error_falls_back_to_str_when_no_body(self):
        """When e.body is None, falls back to str(e)."""
        provider, request = _make_provider_and_request()

        with patch(
            "amplifier_module_provider_litellm.provider.litellm"
        ) as mock_litellm:
            _patch_litellm_error_classes(mock_litellm)
            AuthErr = type("AuthenticationError", (_FakeError,), {})
            mock_litellm.AuthenticationError = AuthErr
            mock_litellm.acompletion = AsyncMock(
                side_effect=AuthErr("fallback string msg", body=None)
            )

            from amplifier_core.llm_errors import (
                AuthenticationError as KernelAuthenticationError,
            )

            with pytest.raises(KernelAuthenticationError) as exc_info:
                await provider.complete(request)

            assert "fallback string msg" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generic_exception_falls_back_to_str_when_no_body_attr(self):
        """When e has no .body attribute at all, falls back to str(e)."""
        provider, request = _make_provider_and_request()

        with patch(
            "amplifier_module_provider_litellm.provider.litellm"
        ) as mock_litellm:
            _patch_litellm_error_classes(mock_litellm)
            mock_litellm.acompletion = AsyncMock(
                side_effect=RuntimeError("plain error")
            )

            from amplifier_core.llm_errors import LLMError as KernelLLMError

            with pytest.raises(KernelLLMError) as exc_info:
                await provider.complete(request)

            assert "plain error" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Fake helpers for retry-after extraction tests
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Mimics an httpx.Response with status_code and headers."""

    def __init__(self, status_code=200, headers=None):
        self.status_code = status_code
        self.headers = headers or {}


class _FakeErrorWithResponse(_FakeError):
    """Exception with .response.headers, mimicking litellm exceptions backed by httpx."""

    def __init__(self, message, body=None, status_code=None, response=None):
        super().__init__(message, body=body)
        self.status_code = status_code
        self.response = response


# ---------------------------------------------------------------------------
# Retry-After extraction from response headers
# ---------------------------------------------------------------------------


def _make_no_retry_provider_and_request():
    """Helper: build a LiteLLMProvider with retries disabled and a minimal ChatRequest mock."""
    provider = LiteLLMProvider({"model": "openai/gpt-4o", "max_retries": 0})
    request = MagicMock()
    request.model = "openai/gpt-4o"
    request.messages = []
    request.tools = None
    request.max_output_tokens = 100
    request.temperature = 0.0
    return provider, request


class TestRetryAfterExtraction:
    """Verify retry_after is extracted from litellm exception response headers."""

    @pytest.mark.asyncio
    async def test_rate_limit_with_retry_after_header(self):
        """RateLimitError with Retry-After header -> retry_after == 30.0."""
        provider, request = _make_no_retry_provider_and_request()
        resp = _FakeResponse(status_code=429, headers={"retry-after": "30"})
        body = {"message": "rate limited"}

        with patch(
            "amplifier_module_provider_litellm.provider.litellm"
        ) as mock_litellm:
            _patch_litellm_error_classes(mock_litellm)
            RateErr = type("RateLimitError", (_FakeErrorWithResponse,), {})
            mock_litellm.RateLimitError = RateErr
            mock_litellm.acompletion = AsyncMock(
                side_effect=RateErr(
                    "rate limited", body=body, status_code=429, response=resp
                )
            )

            from amplifier_core.llm_errors import RateLimitError as KernelRateLimitError

            with pytest.raises(KernelRateLimitError) as exc_info:
                await provider.complete(request)

            assert exc_info.value.retry_after == 30.0

    @pytest.mark.asyncio
    async def test_rate_limit_without_retry_after_header(self):
        """RateLimitError without header -> retry_after is None."""
        provider, request = _make_no_retry_provider_and_request()
        resp = _FakeResponse(status_code=429, headers={})
        body = {"message": "rate limited"}

        with patch(
            "amplifier_module_provider_litellm.provider.litellm"
        ) as mock_litellm:
            _patch_litellm_error_classes(mock_litellm)
            RateErr = type("RateLimitError", (_FakeErrorWithResponse,), {})
            mock_litellm.RateLimitError = RateErr
            mock_litellm.acompletion = AsyncMock(
                side_effect=RateErr(
                    "rate limited", body=body, status_code=429, response=resp
                )
            )

            from amplifier_core.llm_errors import RateLimitError as KernelRateLimitError

            with pytest.raises(KernelRateLimitError) as exc_info:
                await provider.complete(request)

            assert exc_info.value.retry_after is None

    @pytest.mark.asyncio
    async def test_rate_limit_no_response_attribute(self):
        """RateLimitError with no .response attribute -> retry_after is None (no crash)."""
        provider, request = _make_no_retry_provider_and_request()
        body = {"message": "rate limited"}

        with patch(
            "amplifier_module_provider_litellm.provider.litellm"
        ) as mock_litellm:
            _patch_litellm_error_classes(mock_litellm)
            # Use _FakeError which has no .response attribute
            RateErr = type("RateLimitError", (_FakeError,), {})
            mock_litellm.RateLimitError = RateErr
            mock_litellm.acompletion = AsyncMock(
                side_effect=RateErr("rate limited", body=body)
            )

            from amplifier_core.llm_errors import RateLimitError as KernelRateLimitError

            with pytest.raises(KernelRateLimitError) as exc_info:
                await provider.complete(request)

            assert exc_info.value.retry_after is None

    @pytest.mark.asyncio
    async def test_service_unavailable_with_retry_after_header(self):
        """ServiceUnavailableError with Retry-After header -> retry_after == 60.0."""
        provider, request = _make_no_retry_provider_and_request()
        resp = _FakeResponse(status_code=503, headers={"retry-after": "60"})
        body = {"message": "service unavailable"}

        with patch(
            "amplifier_module_provider_litellm.provider.litellm"
        ) as mock_litellm:
            _patch_litellm_error_classes(mock_litellm)
            SvcErr = type("ServiceUnavailableError", (_FakeErrorWithResponse,), {})
            mock_litellm.ServiceUnavailableError = SvcErr
            mock_litellm.acompletion = AsyncMock(
                side_effect=SvcErr(
                    "service unavailable",
                    body=body,
                    status_code=503,
                    response=resp,
                )
            )

            from amplifier_core.llm_errors import (
                ProviderUnavailableError as KernelProviderUnavailableError,
            )

            with pytest.raises(KernelProviderUnavailableError) as exc_info:
                await provider.complete(request)

            assert exc_info.value.retry_after == 60.0

    @pytest.mark.asyncio
    async def test_service_unavailable_without_retry_after_header(self):
        """ServiceUnavailableError without header -> retry_after is None."""
        provider, request = _make_no_retry_provider_and_request()
        resp = _FakeResponse(status_code=503, headers={})
        body = {"message": "service unavailable"}

        with patch(
            "amplifier_module_provider_litellm.provider.litellm"
        ) as mock_litellm:
            _patch_litellm_error_classes(mock_litellm)
            SvcErr = type("ServiceUnavailableError", (_FakeErrorWithResponse,), {})
            mock_litellm.ServiceUnavailableError = SvcErr
            mock_litellm.acompletion = AsyncMock(
                side_effect=SvcErr(
                    "service unavailable",
                    body=body,
                    status_code=503,
                    response=resp,
                )
            )

            from amplifier_core.llm_errors import (
                ProviderUnavailableError as KernelProviderUnavailableError,
            )

            with pytest.raises(KernelProviderUnavailableError) as exc_info:
                await provider.complete(request)

            assert exc_info.value.retry_after is None

    @pytest.mark.asyncio
    async def test_rate_limit_with_non_numeric_retry_after_header(self):
        """RateLimitError with non-numeric Retry-After header -> retry_after is None."""
        provider, request = _make_no_retry_provider_and_request()
        resp = _FakeResponse(status_code=429, headers={"retry-after": "not-a-number"})
        body = {"message": "rate limited"}

        with patch(
            "amplifier_module_provider_litellm.provider.litellm"
        ) as mock_litellm:
            _patch_litellm_error_classes(mock_litellm)
            RateErr = type("RateLimitError", (_FakeErrorWithResponse,), {})
            mock_litellm.RateLimitError = RateErr
            mock_litellm.acompletion = AsyncMock(
                side_effect=RateErr(
                    "rate limited", body=body, status_code=429, response=resp
                )
            )

            from amplifier_core.llm_errors import RateLimitError as KernelRateLimitError

            with pytest.raises(KernelRateLimitError) as exc_info:
                await provider.complete(request)

            assert exc_info.value.retry_after is None


# ---------------------------------------------------------------------------
# Overloaded (529) delay multiplier
# ---------------------------------------------------------------------------


class TestOverloadedDelayMultiplier:
    """Verify delay_multiplier is applied for overloaded (529) errors routed through ServiceUnavailableError."""

    @pytest.mark.asyncio
    async def test_status_529_gets_overloaded_delay_multiplier(self):
        """Status 529 -> delay_multiplier == 10.0 and status_code == 529."""
        provider, request = _make_no_retry_provider_and_request()
        resp = _FakeResponse(status_code=529, headers={})
        body = {"message": "overloaded"}

        with patch(
            "amplifier_module_provider_litellm.provider.litellm"
        ) as mock_litellm:
            _patch_litellm_error_classes(mock_litellm)
            SvcErr = type("ServiceUnavailableError", (_FakeErrorWithResponse,), {})
            mock_litellm.ServiceUnavailableError = SvcErr
            mock_litellm.acompletion = AsyncMock(
                side_effect=SvcErr(
                    "overloaded", body=body, status_code=529, response=resp
                )
            )

            from amplifier_core.llm_errors import (
                ProviderUnavailableError as KernelProviderUnavailableError,
            )

            with pytest.raises(KernelProviderUnavailableError) as exc_info:
                await provider.complete(request)

            assert exc_info.value.delay_multiplier == 10.0
            assert exc_info.value.status_code == 529

    @pytest.mark.asyncio
    async def test_overloaded_in_message_with_status_503(self):
        """'overloaded' in message with status 503 -> delay_multiplier == 10.0."""
        provider, request = _make_no_retry_provider_and_request()
        resp = _FakeResponse(status_code=503, headers={})
        body = {"message": "API is overloaded"}

        with patch(
            "amplifier_module_provider_litellm.provider.litellm"
        ) as mock_litellm:
            _patch_litellm_error_classes(mock_litellm)
            SvcErr = type("ServiceUnavailableError", (_FakeErrorWithResponse,), {})
            mock_litellm.ServiceUnavailableError = SvcErr
            mock_litellm.acompletion = AsyncMock(
                side_effect=SvcErr(
                    "API is overloaded", body=body, status_code=503, response=resp
                )
            )

            from amplifier_core.llm_errors import (
                ProviderUnavailableError as KernelProviderUnavailableError,
            )

            with pytest.raises(KernelProviderUnavailableError) as exc_info:
                await provider.complete(request)

            assert exc_info.value.delay_multiplier == 10.0

    @pytest.mark.asyncio
    async def test_genuine_503_no_overloaded_multiplier(self):
        """Genuine 503 (not overloaded) -> delay_multiplier == 1.0."""
        provider, request = _make_no_retry_provider_and_request()
        resp = _FakeResponse(status_code=503, headers={})
        body = {"message": "service unavailable"}

        with patch(
            "amplifier_module_provider_litellm.provider.litellm"
        ) as mock_litellm:
            _patch_litellm_error_classes(mock_litellm)
            SvcErr = type("ServiceUnavailableError", (_FakeErrorWithResponse,), {})
            mock_litellm.ServiceUnavailableError = SvcErr
            mock_litellm.acompletion = AsyncMock(
                side_effect=SvcErr(
                    "service unavailable",
                    body=body,
                    status_code=503,
                    response=resp,
                )
            )

            from amplifier_core.llm_errors import (
                ProviderUnavailableError as KernelProviderUnavailableError,
            )

            with pytest.raises(KernelProviderUnavailableError) as exc_info:
                await provider.complete(request)

            assert exc_info.value.delay_multiplier == 1.0

    @pytest.mark.asyncio
    async def test_custom_overloaded_delay_multiplier_config(self):
        """Custom overloaded_delay_multiplier config of 5.0 -> delay_multiplier == 5.0."""
        provider = LiteLLMProvider(
            {
                "model": "openai/gpt-4o",
                "overloaded_delay_multiplier": 5.0,
                "max_retries": 0,
            }
        )
        request = MagicMock()
        request.model = "openai/gpt-4o"
        request.messages = []
        request.tools = None
        request.max_output_tokens = 100
        request.temperature = 0.0

        resp = _FakeResponse(status_code=529, headers={})
        body = {"message": "overloaded"}

        with patch(
            "amplifier_module_provider_litellm.provider.litellm"
        ) as mock_litellm:
            _patch_litellm_error_classes(mock_litellm)
            SvcErr = type("ServiceUnavailableError", (_FakeErrorWithResponse,), {})
            mock_litellm.ServiceUnavailableError = SvcErr
            mock_litellm.acompletion = AsyncMock(
                side_effect=SvcErr(
                    "overloaded", body=body, status_code=529, response=resp
                )
            )

            from amplifier_core.llm_errors import (
                ProviderUnavailableError as KernelProviderUnavailableError,
            )

            with pytest.raises(KernelProviderUnavailableError) as exc_info:
                await provider.complete(request)

            assert exc_info.value.delay_multiplier == 5.0
