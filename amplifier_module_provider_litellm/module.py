"""Amplifier module entry point for provider-litellm.

This file is discovered by Amplifier's module loader via the
``__amplifier_module_type__`` marker.
"""

from __future__ import annotations

import logging
from typing import Any

from amplifier_core import ModuleCoordinator

logger = logging.getLogger(__name__)

__amplifier_module_type__ = "provider"


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None) -> None:
    """Mount the LiteLLM provider on an Amplifier coordinator.

    Args:
        coordinator: Amplifier ModuleCoordinator.
        config: Optional provider config. Keys:
            model: default model (e.g. "anthropic/claude-opus-4-6")
            timeout: request timeout seconds (default: 300)
            drop_params: let litellm drop unsupported params (default: true)
            max_retries: retry count for transient errors (default: 3)
            api_base: base URL for self-hosted endpoints (e.g. "http://localhost:8080")
            api_key: API key override (defaults to env var or "not-needed" for local servers)
    """
    from amplifier_module_provider_litellm.provider import LiteLLMProvider

    provider = LiteLLMProvider(config, coordinator=coordinator)
    await coordinator.mount("providers", provider, name="litellm")
    # Declare the events this provider contributes so the ecosystem can
    # discover them (e.g. for UI wiring, documentation, or validation).
    coordinator.register_contributor(
        "events",
        "provider-litellm",
        lambda: [
            "llm:request",
            "llm:response",
            "llm:stream_block_start",
            "llm:stream_block_delta",
            "llm:stream_block_end",
            "llm:stream_aborted",
        ],
    )
    logger.info("Mounted LiteLLMProvider (default_model=%s)", provider.default_model)
