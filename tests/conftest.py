"""Shared test helpers for LiteLLM provider tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock


class _FakeError(Exception):
    """Exception with .body and .status_code attributes, mimicking litellm error structure."""

    def __init__(
        self,
        message: str,
        body: dict[str, Any] | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.body = body
        self.status_code = status_code


def _patch_litellm_error_classes(mock_litellm: MagicMock) -> MagicMock:
    """Assign all litellm error classes so the except chain works.

    Every class is set to a unique type that will never match, so the
    caller can then override the one it actually wants to trigger.
    """

    class _Never(Exception):
        pass

    mock_litellm.AuthenticationError = _Never
    mock_litellm.PermissionDeniedError = _Never
    mock_litellm.RateLimitError = _Never
    mock_litellm.ContextWindowExceededError = _Never
    mock_litellm.ContentPolicyViolationError = _Never
    mock_litellm.BadRequestError = _Never
    mock_litellm.ServiceUnavailableError = _Never
    mock_litellm.NotFoundError = _Never
    mock_litellm.APIConnectionError = _Never
    mock_litellm.Timeout = _Never
    return mock_litellm
