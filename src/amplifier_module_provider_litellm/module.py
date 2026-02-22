"""Amplifier module entry point for provider-litellm.

This file is discovered by Amplifier's module loader via the
``__amplifier_module_type__`` marker.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__amplifier_module_type__ = "provider"


async def mount(coordinator: Any, config: dict[str, Any] | None = None) -> None:
    """Mount the LiteLLM provider on an Amplifier coordinator.

    Args:
        coordinator: Amplifier ModuleCoordinator.
        config: Optional provider config. Keys:
            model: default model (e.g. "anthropic/claude-opus-4-6")
            timeout: request timeout seconds (default: 300)
            drop_params: let litellm drop unsupported params (default: true)
    """
    from amplifier_module_provider_litellm.provider import LiteLLMProvider

    provider = LiteLLMProvider(config)
    await coordinator.mount("providers", provider, name="litellm")
    logger.info("Mounted LiteLLMProvider (default_model=%s)", provider.default_model)
