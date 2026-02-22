"""Amplifier provider module using litellm for multi-provider LLM access.

Supports 100+ LLM providers (Anthropic, OpenAI, Google, Ollama, OpenRouter,
Azure, Bedrock, etc.) via standard environment variables.  Zero configuration
needed — if the env var is set, the provider works.

Usage as an Amplifier module:
    Module type: provider
    Config keys:
        model: default model name (e.g. "anthropic/claude-opus-4-6")
        timeout: request timeout in seconds (default: 300)
        drop_params: allow litellm to drop unsupported params (default: true)
"""

__version__ = "0.1.0"

# Module metadata and mount() must be importable from __init__ for the
# Amplifier module loader to discover them.
from amplifier_module_provider_litellm.module import mount, __amplifier_module_type__

__all__ = ["mount", "__amplifier_module_type__", "__version__"]
