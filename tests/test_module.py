"""Tests for the Amplifier module mount point."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_module_provider_litellm.module import mount, __amplifier_module_type__


class TestModuleMetadata:
    def test_module_type(self):
        assert __amplifier_module_type__ == "provider"


class TestMount:
    @pytest.mark.asyncio
    async def test_mounts_provider(self):
        coordinator = MagicMock()
        coordinator.mount = AsyncMock()

        await mount(coordinator)

        coordinator.mount.assert_called_once()
        args, kwargs = coordinator.mount.call_args
        assert args[0] == "providers"
        assert kwargs["name"] == "litellm"
        assert args[1].name == "litellm"
        # Verify coordinator is passed through
        assert args[1].coordinator is coordinator

    @pytest.mark.asyncio
    async def test_mounts_with_custom_config(self):
        coordinator = MagicMock()
        coordinator.mount = AsyncMock()

        await mount(coordinator, {"model": "gemini/gemini-2.5-pro", "timeout": 120})

        provider = coordinator.mount.call_args[0][1]
        assert provider.default_model == "gemini/gemini-2.5-pro"
        assert provider._timeout == 120.0

    @pytest.mark.asyncio
    async def test_register_contributor_called(self):
        """register_contributor is called to declare emitted events on the events channel."""
        coordinator = MagicMock()
        coordinator.mount = AsyncMock()

        await mount(coordinator)

        coordinator.register_contributor.assert_called_once()
        call_args = coordinator.register_contributor.call_args
        channel = call_args[0][0]
        contributor_name = call_args[0][1]
        callback = call_args[0][2]

        assert channel == "events"
        assert "litellm" in contributor_name

        # Callback must return the declared event list
        declared = callback()
        assert "llm:request" in declared
        assert "llm:response" in declared
        assert "llm:stream_block_start" in declared
        assert "llm:stream_block_delta" in declared
        # Contract: llm:stream_thinking_delta is removed; block_type carries the distinction.
        assert "llm:stream_thinking_delta" not in declared
        assert "llm:stream_block_end" in declared
        assert "llm:stream_aborted" in declared
