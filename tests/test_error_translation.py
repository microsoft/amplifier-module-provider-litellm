"""Tests for 403 error translation in LiteLLM provider."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from amplifier_core.llm_errors import (
    AccessDeniedError as KernelAccessDeniedError,
    ProviderUnavailableError as KernelProviderUnavailableError,
)
from conftest import _FakeError, _patch_litellm_error_classes


def _make_no_retry_provider_and_request():
    """Build a LiteLLMProvider with retries disabled and a minimal ChatRequest mock."""
    from amplifier_module_provider_litellm.provider import LiteLLMProvider

    provider = LiteLLMProvider({"model": "openai/gpt-4o", "max_retries": 0})
    request = MagicMock()
    request.model = "openai/gpt-4o"
    request.messages = []
    request.tools = None
    request.max_output_tokens = 100
    request.temperature = 0.0
    return provider, request


class TestCloudflare403Detection:
    """Verify that LiteLLM 403 errors are correctly classified."""

    @pytest.mark.asyncio
    async def test_403_with_body_none_raises_provider_unavailable(self):
        """403 with body=None (CDN/proxy challenge) -> ProviderUnavailableError (retryable)."""
        provider, request = _make_no_retry_provider_and_request()

        with patch(
            "amplifier_module_provider_litellm.provider.litellm"
        ) as mock_litellm:
            _patch_litellm_error_classes(mock_litellm)
            PermErr = type("PermissionDeniedError", (_FakeError,), {})
            mock_litellm.PermissionDeniedError = PermErr
            mock_litellm.acompletion = AsyncMock(
                side_effect=PermErr("forbidden", body=None, status_code=403)
            )

            with pytest.raises(KernelProviderUnavailableError) as exc_info:
                await provider.complete(request)

            assert exc_info.value.status_code == 403
            assert exc_info.value.retryable is True

    @pytest.mark.asyncio
    async def test_403_with_body_raises_access_denied(self):
        """403 with body (real denial) -> AccessDeniedError (not retryable)."""
        provider, request = _make_no_retry_provider_and_request()
        body = {
            "message": "You don't have access to this model",
            "type": "permission_denied",
        }

        with patch(
            "amplifier_module_provider_litellm.provider.litellm"
        ) as mock_litellm:
            _patch_litellm_error_classes(mock_litellm)
            PermErr = type("PermissionDeniedError", (_FakeError,), {})
            mock_litellm.PermissionDeniedError = PermErr
            mock_litellm.acompletion = AsyncMock(
                side_effect=PermErr("permission denied", body=body, status_code=403)
            )

            with pytest.raises(KernelAccessDeniedError) as exc_info:
                await provider.complete(request)

            assert exc_info.value.status_code == 403
            assert exc_info.value.retryable is False
            assert json.dumps(body) in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_403_does_not_fall_through_to_generic_handler(self):
        """PermissionDeniedError must NOT fall through to the generic Exception handler."""
        from amplifier_core.llm_errors import LLMError as KernelLLMError

        provider, request = _make_no_retry_provider_and_request()

        with patch(
            "amplifier_module_provider_litellm.provider.litellm"
        ) as mock_litellm:
            _patch_litellm_error_classes(mock_litellm)
            PermErr = type("PermissionDeniedError", (_FakeError,), {})
            mock_litellm.PermissionDeniedError = PermErr
            mock_litellm.acompletion = AsyncMock(
                side_effect=PermErr("forbidden", body=None, status_code=403)
            )

            with pytest.raises(KernelLLMError) as exc_info:
                await provider.complete(request)

            # Must be ProviderUnavailableError, NOT generic LLMError
            assert isinstance(exc_info.value, KernelProviderUnavailableError)
