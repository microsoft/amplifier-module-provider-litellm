"""LiteLLM provider for Amplifier — routes LLM calls through litellm.

litellm reads API keys from standard environment variables:
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, XAI_API_KEY,
    GROQ_API_KEY, OPENROUTER_API_KEY, OLLAMA_API_BASE, etc.

Model names follow litellm conventions:
    anthropic/claude-opus-4-6, openai/gpt-4o, gemini/gemini-2.0-flash,
    ollama/llama3.2, openrouter/meta-llama/llama-3-70b, etc.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import litellm
from amplifier_core import ConfigField, ModelInfo, ModuleCoordinator, ProviderInfo
from amplifier_core.llm_errors import (
    AuthenticationError as KernelAuthenticationError,
    ContentFilterError as KernelContentFilterError,
    ContextLengthError as KernelContextLengthError,
    InvalidRequestError as KernelInvalidRequestError,
    LLMError as KernelLLMError,
    LLMTimeoutError as KernelLLMTimeoutError,
    NetworkError as KernelNetworkError,
    NotFoundError as KernelNotFoundError,
    ProviderUnavailableError as KernelProviderUnavailableError,
    RateLimitError as KernelRateLimitError,
)
from amplifier_core.message_models import (
    ChatRequest,
    ChatResponse,
    TextBlock,
    ToolCall,
    Usage,
)
from amplifier_core.utils.retry import RetryConfig, retry_with_backoff

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging by default
litellm.suppress_debug_info = True

_DEFAULT_TIMEOUT = 300.0


def _extract_retry_after(exc: Exception) -> float | None:
    """Extract retry-after seconds from a litellm exception's httpx response headers.

    Returns the parsed float value, or None if unavailable or unparseable.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


class LiteLLMProvider:
    """Amplifier provider that delegates LLM calls to litellm.

    litellm handles provider detection, API key resolution, and request
    formatting automatically based on the model name prefix.
    """

    name = "litellm"
    api_label = "LiteLLM"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        coordinator: ModuleCoordinator | None = None,
    ) -> None:
        self.config = config or {}
        self.coordinator = coordinator
        self.default_model: str = self.config.get("model", "anthropic/claude-opus-4-6")
        self._timeout: float = float(self.config.get("timeout", _DEFAULT_TIMEOUT))
        self._drop_params: bool = self.config.get("drop_params", True)
        self.debug: bool = self.config.get("debug", False)
        self._overloaded_delay_multiplier: float = float(
            self.config.get("overloaded_delay_multiplier", 10.0)
        )

        # Retry configuration — delegates to shared retry_with_backoff from amplifier-core
        jitter_val = self.config.get("retry_jitter", 0.2)
        if isinstance(jitter_val, bool):
            jitter_val = 0.2 if jitter_val else 0.0
        self._retry_config = RetryConfig(
            max_retries=int(self.config.get("max_retries", 3)),
            min_delay=float(self.config.get("min_retry_delay", 1.0)),
            max_delay=float(self.config.get("max_retry_delay", 60.0)),
            jitter=float(jitter_val),
        )

    def get_info(self) -> ProviderInfo:
        """Return provider metadata."""
        return ProviderInfo(
            id="litellm",
            display_name="LiteLLM (Multi-Provider)",
            credential_env_vars=[],  # litellm reads env vars per-provider automatically
            capabilities=["tools", "streaming"],
            defaults={
                "model": self.default_model,
                "max_tokens": 8192,
                "temperature": 0.7,
                "timeout": self._timeout,
            },
            config_fields=[
                ConfigField(
                    id="model",
                    display_name="Default Model",
                    field_type="text",
                    prompt="Default model (e.g. anthropic/claude-opus-4-6, openai/gpt-4o, gemini/gemini-2.5-flash)",
                    required=False,
                    default="anthropic/claude-opus-4-6",
                ),
            ],
        )

    async def list_models(self) -> list[ModelInfo]:
        """Return a list of known available models.

        Checks which provider env vars are set and returns representative
        models for each.  Best-effort — litellm supports far more models
        than we enumerate here.
        """
        models: list[ModelInfo] = []

        # Provider -> (env_var, [(model_id, display_name, context_window, max_output)])
        provider_models = [
            (
                "ANTHROPIC_API_KEY",
                [
                    ("anthropic/claude-opus-4-6", "Claude Opus 4", 200_000, 128_000),
                    (
                        "anthropic/claude-sonnet-4-20250514",
                        "Claude Sonnet 4",
                        200_000,
                        64_000,
                    ),
                ],
            ),
            (
                "OPENAI_API_KEY",
                [
                    ("openai/gpt-4o", "GPT-4o", 128_000, 16_384),
                    ("openai/gpt-4o-mini", "GPT-4o Mini", 128_000, 16_384),
                    ("openai/o3", "o3", 200_000, 100_000),
                ],
            ),
            (
                "GEMINI_API_KEY",
                [
                    ("gemini/gemini-2.5-pro", "Gemini 2.5 Pro", 1_000_000, 64_000),
                    ("gemini/gemini-2.5-flash", "Gemini 2.5 Flash", 1_000_000, 64_000),
                ],
            ),
            (
                "XAI_API_KEY",
                [
                    ("xai/grok-3", "Grok 3", 131_072, 16_384),
                ],
            ),
            (
                "GROQ_API_KEY",
                [
                    (
                        "groq/llama-3.3-70b-versatile",
                        "Llama 3.3 70B (Groq)",
                        128_000,
                        32_000,
                    ),
                ],
            ),
            (
                "OPENROUTER_API_KEY",
                [
                    ("openrouter/auto", "OpenRouter Auto", 200_000, 64_000),
                ],
            ),
        ]

        for env_var, model_list in provider_models:
            if os.environ.get(env_var):
                for mid, display, ctx, max_out in model_list:
                    models.append(
                        ModelInfo(
                            id=mid,
                            display_name=display,
                            context_window=ctx,
                            max_output_tokens=max_out,
                            capabilities=["tools", "streaming"],
                            metadata={"cost_tier": "medium"},
                        )
                    )

        # Always include default model if not already listed
        if not any(m.id == self.default_model for m in models):
            models.insert(
                0,
                ModelInfo(
                    id=self.default_model,
                    display_name=self.default_model,
                    context_window=200_000,
                    max_output_tokens=64_000,
                    capabilities=["tools", "streaming"],
                    metadata={"cost_tier": "medium"},
                ),
            )

        return models

    async def complete(self, request: ChatRequest, **kwargs: Any) -> ChatResponse:
        """Send a completion request through litellm with retry and error translation."""
        model = request.model or kwargs.get("model", self.default_model)
        timeout = kwargs.get("timeout", self._timeout)

        # Build litellm params
        messages = _to_litellm_messages(request)
        tools = _to_litellm_tools(request) if request.tools else None

        litellm_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "timeout": timeout,
            "drop_params": self._drop_params,
        }

        if tools:
            litellm_kwargs["tools"] = tools

        if request.max_output_tokens:
            litellm_kwargs["max_tokens"] = request.max_output_tokens

        if request.temperature is not None:
            litellm_kwargs["temperature"] = request.temperature

        # Emit llm:request event
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            await self.coordinator.hooks.emit(
                "llm:request",
                {
                    "provider": "litellm",
                    "model": model,
                    "message_count": len(messages),
                    "tool_count": len(tools or []),
                },
            )

        logger.info(
            "litellm complete: model=%s, messages=%d, tools=%d",
            model,
            len(messages),
            len(tools or []),
        )

        start_time = time.time()

        async def _do_complete():
            """Single attempt with error translation."""
            try:
                return await litellm.acompletion(**litellm_kwargs)
            except litellm.AuthenticationError as e:
                body = getattr(e, "body", None)
                msg = json.dumps(body, default=str) if body is not None else str(e)
                raise KernelAuthenticationError(
                    msg,
                    provider="litellm",
                    status_code=401,
                ) from e
            except litellm.RateLimitError as e:
                body = getattr(e, "body", None)
                msg = json.dumps(body, default=str) if body is not None else str(e)
                raise KernelRateLimitError(
                    msg,
                    provider="litellm",
                    status_code=429,
                    retryable=True,
                    retry_after=_extract_retry_after(e),
                ) from e
            except litellm.ContextWindowExceededError as e:
                body = getattr(e, "body", None)
                msg = json.dumps(body, default=str) if body is not None else str(e)
                raise KernelContextLengthError(
                    msg,
                    provider="litellm",
                    status_code=400,
                ) from e
            except litellm.ContentPolicyViolationError as e:
                body = getattr(e, "body", None)
                msg = json.dumps(body, default=str) if body is not None else str(e)
                raise KernelContentFilterError(
                    msg,
                    provider="litellm",
                    status_code=400,
                ) from e
            except litellm.BadRequestError as e:
                body = getattr(e, "body", None)
                msg = json.dumps(body, default=str) if body is not None else str(e)
                raise KernelInvalidRequestError(
                    msg,
                    provider="litellm",
                    status_code=400,
                ) from e
            except litellm.ServiceUnavailableError as e:
                # Anthropic 529 "Overloaded" routes through litellm as
                # ServiceUnavailableError. Detect overloaded condition to
                # apply a longer backoff via delay_multiplier.
                body = getattr(e, "body", None)
                msg = json.dumps(body, default=str) if body is not None else str(e)
                status = getattr(e, "status_code", 503) or 503
                is_overloaded = status == 529 or "overloaded" in str(e).lower()
                multiplier = self._overloaded_delay_multiplier if is_overloaded else 1.0
                raise KernelProviderUnavailableError(
                    msg,
                    provider="litellm",
                    status_code=status,
                    retryable=True,
                    retry_after=_extract_retry_after(e),
                    delay_multiplier=multiplier,
                ) from e
            except litellm.NotFoundError as e:
                body = getattr(e, "body", None)
                msg = json.dumps(body, default=str) if body is not None else str(e)
                raise KernelNotFoundError(
                    msg,
                    provider="litellm",
                    status_code=404,
                ) from e
            except litellm.APIConnectionError as e:
                body = getattr(e, "body", None)
                msg = json.dumps(body, default=str) if body is not None else str(e)
                raise KernelNetworkError(
                    msg,
                    provider="litellm",
                    retryable=True,
                ) from e
            except litellm.Timeout as e:
                raise KernelLLMTimeoutError(
                    f"Request timed out after {timeout}s",
                    provider="litellm",
                    retryable=True,
                ) from e
            except KernelLLMError:
                raise  # Already translated
            except Exception as e:
                body = getattr(e, "body", None)
                error_msg = (
                    json.dumps(body, default=str)
                    if body is not None
                    else (str(e) or f"{type(e).__name__}: (no message)")
                )
                raise KernelLLMError(
                    error_msg,
                    provider="litellm",
                    retryable=True,
                ) from e

        async def _on_retry(attempt: int, delay: float, error: KernelLLMError):
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "provider:retry",
                    {
                        "provider": "litellm",
                        "model": model,
                        "attempt": attempt,
                        "delay": delay,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                )

        try:
            response = await retry_with_backoff(
                _do_complete,
                self._retry_config,
                on_retry=_on_retry,
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            # Emit llm:response event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                response_event: dict[str, Any] = {
                    "provider": "litellm",
                    "model": model,
                    "status": "ok",
                    "duration_ms": elapsed_ms,
                }
                if response.usage:
                    response_event["usage"] = {
                        "input": response.usage.prompt_tokens or 0,
                        "output": response.usage.completion_tokens or 0,
                    }
                await self.coordinator.hooks.emit("llm:response", response_event)

            return _from_litellm_response(response)

        except KernelLLMError as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error("[PROVIDER] litellm error: %s", str(e))
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "litellm",
                        "model": model,
                        "status": "error",
                        "duration_ms": elapsed_ms,
                        "error": str(e),
                    },
                )
            raise

    def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]:
        """Extract tool calls from a ChatResponse."""
        if not response.tool_calls:
            return []

        valid = []
        for tc in response.tool_calls:
            if tc.arguments is None:
                logger.debug("Filtering tool call '%s' with None arguments", tc.name)
                continue
            valid.append(tc)
        return valid


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _to_litellm_messages(request: ChatRequest) -> list[dict[str, Any]]:
    """Convert ChatRequest messages to litellm format (OpenAI-compatible)."""
    messages = []
    for msg in request.messages:
        role = msg.role
        content = msg.content

        if role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id or "",
                    "content": content if isinstance(content, str) else str(content),
                }
            )
            continue

        entry: dict[str, Any] = {"role": role}

        if isinstance(content, str):
            entry["content"] = content
        elif isinstance(content, list):
            # Multi-part content (text + images)
            # Serialize Pydantic models to dicts — the loop module may store
            # content blocks as TextBlock/ToolCallBlock objects, but litellm
            # expects plain dicts or strings.
            serialized = []
            for block in content:
                if hasattr(block, "model_dump"):
                    serialized.append(block.model_dump())
                elif isinstance(block, dict):
                    serialized.append(block)
                else:
                    serialized.append(str(block))
            entry["content"] = serialized
        else:
            entry["content"] = str(content) if content else ""

        # Preserve tool_calls on assistant messages
        # Tool calls may be ToolCall objects OR plain dicts (from context storage)
        if role == "assistant" and msg.tool_calls:
            normalized_tcs = []
            for tc in msg.tool_calls:
                if isinstance(tc, dict):
                    tc_id = tc.get("id", "")
                    tc_name = tc.get("name") or tc.get("tool", "")
                    tc_args = tc.get("arguments") or {}
                else:
                    tc_id = tc.id or ""
                    tc_name = getattr(tc, "name", None) or getattr(tc, "tool", "") or ""
                    tc_args = tc.arguments if tc.arguments is not None else {}
                normalized_tcs.append(
                    {
                        "id": tc_id,
                        "type": "function",
                        "function": {
                            "name": tc_name,
                            "arguments": json.dumps(tc_args)
                            if isinstance(tc_args, dict)
                            else str(tc_args or "{}"),
                        },
                    }
                )
            entry["tool_calls"] = normalized_tcs

        messages.append(entry)

    return messages


def _to_litellm_tools(request: ChatRequest) -> list[dict[str, Any]] | None:
    """Convert ChatRequest tools to litellm (OpenAI) format."""
    if not request.tools:
        return None

    tools = []
    for t in request.tools:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": t.name or "",
                    "description": t.description or "",
                    "parameters": t.parameters or t.input_schema or {"type": "object"},
                },
            }
        )
    return tools


def _from_litellm_response(response: Any) -> ChatResponse:
    """Convert litellm response to Amplifier ChatResponse."""
    choice = response.choices[0] if response.choices else None
    message = choice.message if choice else None

    text = message.content or "" if message else ""
    tool_calls = []

    if message and message.tool_calls:
        for tc in message.tool_calls:
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": args}

            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                )
            )

    # Build Usage object
    usage = None
    if response.usage:
        input_tokens = response.usage.prompt_tokens or 0
        output_tokens = response.usage.completion_tokens or 0

        usage_kwargs: dict[str, Any] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

        # Extract cache info when available
        if (
            hasattr(response.usage, "prompt_tokens_details")
            and response.usage.prompt_tokens_details
        ):
            details = response.usage.prompt_tokens_details
            if hasattr(details, "cached_tokens") and details.cached_tokens:
                usage_kwargs["cache_read_tokens"] = details.cached_tokens

        usage = Usage(**usage_kwargs)

    # Build content blocks
    content: list[TextBlock] = []
    if text:
        content = [TextBlock(text=text)]

    return ChatResponse(
        content=content,
        tool_calls=tool_calls if tool_calls else None,
        usage=usage,
        finish_reason=choice.finish_reason if choice else None,
        metadata={"model": response.model or ""} if response.model else None,
    )
