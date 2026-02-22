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
from typing import Any

import litellm
from amplifier_core import ModelInfo, ProviderInfo
from amplifier_core.message_models import ChatRequest, ChatResponse, TextBlock, ToolCall, Usage

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging by default
litellm.suppress_debug_info = True

_DEFAULT_TIMEOUT = 300.0


class LiteLLMProvider:
    """Amplifier provider that delegates LLM calls to litellm.

    litellm handles provider detection, API key resolution, and request
    formatting automatically based on the model name prefix.
    """

    name = "litellm"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.default_model: str = self.config.get("model", "anthropic/claude-opus-4-6")
        self._timeout: float = float(self.config.get("timeout", _DEFAULT_TIMEOUT))
        self._drop_params: bool = self.config.get("drop_params", True)

    def get_info(self) -> ProviderInfo:
        """Return provider metadata."""
        return ProviderInfo(
            id="litellm",
            display_name="LiteLLM (Multi-Provider)",
            credential_env_vars=[],  # litellm reads env vars per-provider automatically
            capabilities=["tools", "streaming"],
            defaults={
                "model": self.default_model,
                "timeout": self._timeout,
            },
        )

    async def list_models(self) -> list[ModelInfo]:
        """Return a list of known available models.

        Checks which provider env vars are set and returns representative
        models for each.  This is best-effort — litellm supports far more
        models than we enumerate here.
        """
        import os

        models: list[ModelInfo] = []

        # Provider -> (env_var, [(model_id, display_name, context_window, max_output)])
        provider_models = [
            ("ANTHROPIC_API_KEY", [
                ("anthropic/claude-opus-4-6", "Claude Opus 4", 200_000, 32_000),
                ("anthropic/claude-sonnet-4-20250514", "Claude Sonnet 4", 200_000, 64_000),
            ]),
            ("OPENAI_API_KEY", [
                ("openai/gpt-4o", "GPT-4o", 128_000, 16_384),
                ("openai/gpt-4o-mini", "GPT-4o Mini", 128_000, 16_384),
                ("openai/o3", "o3", 200_000, 100_000),
            ]),
            ("GEMINI_API_KEY", [
                ("gemini/gemini-2.5-pro", "Gemini 2.5 Pro", 1_000_000, 64_000),
                ("gemini/gemini-2.5-flash", "Gemini 2.5 Flash", 1_000_000, 64_000),
            ]),
            ("XAI_API_KEY", [
                ("xai/grok-3", "Grok 3", 131_072, 16_384),
            ]),
            ("GROQ_API_KEY", [
                ("groq/llama-3.3-70b-versatile", "Llama 3.3 70B (Groq)", 128_000, 32_000),
            ]),
            ("OPENROUTER_API_KEY", [
                ("openrouter/auto", "OpenRouter Auto", 200_000, 64_000),
            ]),
        ]

        for env_var, model_list in provider_models:
            if os.environ.get(env_var):
                for mid, display, ctx, max_out in model_list:
                    models.append(ModelInfo(
                        id=mid,
                        display_name=display,
                        context_window=ctx,
                        max_output_tokens=max_out,
                        capabilities=["tools", "streaming"],
                    ))

        # Always include default model if not already listed
        if not any(m.id == self.default_model for m in models):
            models.insert(0, ModelInfo(
                id=self.default_model,
                display_name=self.default_model,
                context_window=200_000,
                max_output_tokens=64_000,
                capabilities=["tools", "streaming"],
            ))

        return models

    async def complete(self, request: ChatRequest, **kwargs: Any) -> ChatResponse:
        """Send a completion request through litellm."""
        model = request.model or self.default_model
        timeout = kwargs.get("timeout", self._timeout)

        # Build litellm messages from ChatRequest
        messages = _to_litellm_messages(request)

        # Build litellm tools from ChatRequest
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

        logger.info(
            "litellm complete: model=%s, messages=%d, tools=%d",
            model, len(messages), len(tools or []),
        )

        response = await litellm.acompletion(**litellm_kwargs)

        return _from_litellm_response(response)

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
    """Convert ChatRequest messages to litellm format."""
    messages = []
    for msg in request.messages:
        role = msg.role
        content = msg.content

        if role == "tool":
            messages.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id or "",
                "content": content if isinstance(content, str) else str(content),
            })
            continue

        entry: dict[str, Any] = {"role": role}

        if isinstance(content, str):
            entry["content"] = content
        elif isinstance(content, list):
            # Multi-part content (text + images)
            entry["content"] = content
        else:
            entry["content"] = str(content) if content else ""

        # Preserve tool_calls on assistant messages
        if role == "assistant" and msg.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.id or "",
                    "type": "function",
                    "function": {
                        "name": tc.name or tc.tool or "",
                        "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else str(tc.arguments or "{}"),
                    },
                }
                for tc in msg.tool_calls
            ]

        messages.append(entry)

    return messages


def _to_litellm_tools(request: ChatRequest) -> list[dict[str, Any]] | None:
    """Convert ChatRequest tools to litellm (OpenAI) format."""
    if not request.tools:
        return None

    tools = []
    for t in request.tools:
        tools.append({
            "type": "function",
            "function": {
                "name": t.name or "",
                "description": t.description or "",
                "parameters": t.parameters or t.input_schema or {"type": "object"},
            },
        })
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

            tool_calls.append(ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=args,
            ))

    # Build Usage object
    usage = None
    if response.usage:
        input_tokens = response.usage.prompt_tokens or 0
        output_tokens = response.usage.completion_tokens or 0
        total_tokens = response.usage.total_tokens or 0

        usage_kwargs: dict[str, Any] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

        # Extract cache info when available
        if hasattr(response.usage, "prompt_tokens_details") and response.usage.prompt_tokens_details:
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
