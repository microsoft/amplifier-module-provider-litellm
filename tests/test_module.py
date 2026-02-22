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
