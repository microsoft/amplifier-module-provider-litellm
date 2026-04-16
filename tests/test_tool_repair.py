"""Tests for tool result repair and infinite loop prevention."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch


from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message, ToolCallBlock
from amplifier_module_provider_litellm.provider import LiteLLMProvider

# Import conftest helpers using the module's own tests directory.
# A bare ``from tests.conftest import ...`` can resolve to the wrong package
# when multiple test packages exist on sys.path.
_tests_dir = str(Path(__file__).resolve().parent)
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)
from conftest import _patch_litellm_error_classes  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeHooks:
    """Collects emitted events for inspection."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def emit(self, name: str, payload: dict[str, Any]) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    """Minimal coordinator stub with event hooks."""

    def __init__(self) -> None:
        self.hooks = FakeHooks()


def _litellm_response(content: str = "ok") -> MagicMock:
    """Build a minimal mock for a litellm completion response."""
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=None),
        finish_reason="stop",
    )
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=5,
        prompt_tokens_details=None,
        cache_creation_input_tokens=None,
    )
    resp = SimpleNamespace(choices=[choice], usage=usage, model="test-model")
    return resp


def _make_provider(
    coordinator: FakeCoordinator | None = None,
) -> LiteLLMProvider:
    """Create a provider with a fake coordinator wired up."""
    p = LiteLLMProvider()
    if coordinator is not None:
        p.coordinator = cast(ModuleCoordinator, coordinator)
    return p


# ---------------------------------------------------------------------------
# _find_missing_tool_results — unit tests
# ---------------------------------------------------------------------------


class TestFindMissingToolResults:
    """Tests for the detection logic itself (no mocking of litellm needed)."""

    def test_no_tool_calls_returns_empty(self):
        """Fast path — no tool calls means nothing to repair."""
        provider = _make_provider()
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]
        assert provider._find_missing_tool_results(messages) == []

    def test_single_missing_result_detected(self):
        """An assistant tool call without a following tool result is detected."""
        provider = _make_provider()
        asst = Message(role="assistant", content="calling tool")
        # Simulate OpenAI-style tool_calls attribute (via Pydantic extra)
        asst.tool_calls = [{"id": "call_1", "name": "grep", "arguments": {}}]  # type: ignore[attr-defined]

        messages = [
            asst,
            Message(role="user", content="No tool result here"),
        ]
        missing = provider._find_missing_tool_results(messages)
        assert len(missing) == 1
        assert missing[0] == (0, "call_1", "grep")

    def test_present_result_not_detected(self):
        """A tool call followed by a matching tool result is NOT detected."""
        provider = _make_provider()
        asst = Message(role="assistant", content="calling tool")
        asst.tool_calls = [{"id": "call_1", "name": "grep", "arguments": {}}]  # type: ignore[attr-defined]

        messages = [
            asst,
            Message(role="tool", content="found it", tool_call_id="call_1"),
            Message(role="user", content="Thanks"),
        ]
        assert provider._find_missing_tool_results(messages) == []

    def test_mix_of_present_and_missing(self):
        """Only missing results are returned when some are present."""
        provider = _make_provider()
        asst = Message(role="assistant", content="calling tools")
        asst.tool_calls = [  # type: ignore[attr-defined]
            {"id": "call_1", "name": "grep", "arguments": {}},
            {"id": "call_2", "name": "bash", "arguments": {}},
            {"id": "call_3", "name": "read_file", "arguments": {}},
        ]

        messages = [
            asst,
            Message(role="tool", content="result 1", tool_call_id="call_1"),
            # call_2 and call_3 are missing
            Message(role="user", content="Continue"),
        ]
        missing = provider._find_missing_tool_results(messages)
        ids = {m[1] for m in missing}
        assert ids == {"call_2", "call_3"}

    def test_already_repaired_ids_skipped(self):
        """IDs in _repaired_tool_ids are not re-detected."""
        provider = _make_provider()
        provider._repaired_tool_ids.add("call_1")

        asst = Message(role="assistant", content="calling tool")
        asst.tool_calls = [{"id": "call_1", "name": "grep", "arguments": {}}]  # type: ignore[attr-defined]

        messages = [
            asst,
            Message(role="user", content="No tool result"),
        ]
        assert provider._find_missing_tool_results(messages) == []

    def test_tool_calls_as_objects(self):
        """ToolCall-like objects (with .id, .name attributes) are handled."""
        provider = _make_provider()
        tc = SimpleNamespace(id="call_obj_1", name="edit_file", arguments={"a": 1})

        asst = Message(role="assistant", content="calling")
        asst.tool_calls = [tc]  # type: ignore[attr-defined]

        messages = [
            asst,
            Message(role="user", content="No result"),
        ]
        missing = provider._find_missing_tool_results(messages)
        assert len(missing) == 1
        assert missing[0] == (0, "call_obj_1", "edit_file")

    def test_content_block_tool_calls_detected(self):
        """Tool calls inside content list as ToolCallBlock are detected."""
        provider = _make_provider()
        messages = [
            Message(
                role="assistant",
                content=[
                    ToolCallBlock(id="call_blk_1", name="bash", input={"cmd": "ls"}),
                ],
            ),
            Message(role="user", content="No result"),
        ]
        missing = provider._find_missing_tool_results(messages)
        assert len(missing) == 1
        assert missing[0] == (0, "call_blk_1", "bash")

    def test_content_block_dict_tool_calls_detected(self):
        """Tool calls as dicts in content list (ToolCallBlock form) are detected."""
        provider = _make_provider()
        messages = [
            Message(
                role="assistant",
                content=[
                    ToolCallBlock(
                        id="call_dict_1", name="write_file", input={"path": "/a.txt"}
                    ),
                ],
            ),
            Message(role="user", content="No result"),
        ]
        missing = provider._find_missing_tool_results(messages)
        assert len(missing) == 1
        assert missing[0] == (0, "call_dict_1", "write_file")

    def test_multiple_assistant_messages_correct_indices(self):
        """Missing results reference the correct assistant message index."""
        provider = _make_provider()

        asst1 = Message(role="assistant", content="first")
        asst1.tool_calls = [{"id": "call_a", "name": "grep", "arguments": {}}]  # type: ignore[attr-defined]

        asst2 = Message(role="assistant", content="second")
        asst2.tool_calls = [{"id": "call_b", "name": "bash", "arguments": {}}]  # type: ignore[attr-defined]

        messages = [
            asst1,
            Message(role="tool", content="result a", tool_call_id="call_a"),
            Message(role="user", content="next"),
            asst2,  # index 3
            # call_b has no result
            Message(role="user", content="done"),
        ]
        missing = provider._find_missing_tool_results(messages)
        assert len(missing) == 1
        assert missing[0] == (3, "call_b", "bash")

    def test_empty_tool_call_id_ignored(self):
        """Tool calls with empty or missing IDs are silently skipped."""
        provider = _make_provider()
        asst = Message(role="assistant", content="calling")
        asst.tool_calls = [  # type: ignore[attr-defined]
            {"id": "", "name": "grep", "arguments": {}},
            {"name": "bash", "arguments": {}},  # no id key at all
        ]

        messages = [asst, Message(role="user", content="done")]
        assert provider._find_missing_tool_results(messages) == []


# ---------------------------------------------------------------------------
# _create_synthetic_result — unit tests
# ---------------------------------------------------------------------------


class TestCreateSyntheticResult:
    def test_returns_tool_message(self):
        provider = _make_provider()
        result = provider._create_synthetic_result("call_42")
        assert result.role == "tool"
        assert result.tool_call_id == "call_42"
        assert "SYSTEM ERROR" in result.content
        assert "lost during context management" in result.content


# ---------------------------------------------------------------------------
# Integration: complete() with repair (full round-trip)
# ---------------------------------------------------------------------------


class TestCompleteWithRepair:
    """End-to-end tests that mock litellm.acompletion and verify repair behavior."""

    @patch("amplifier_module_provider_litellm.provider.litellm")
    def test_single_missing_result_repaired(self, mock_litellm: MagicMock):
        """Missing tool result triggers repair, event emitted, LLM call succeeds."""
        mock_litellm.acompletion = AsyncMock(return_value=_litellm_response())
        _patch_litellm_error_classes(mock_litellm)

        coord = FakeCoordinator()
        provider = _make_provider(coordinator=coord)

        asst = Message(role="assistant", content="calling tool")
        asst.tool_calls = [
            {"id": "call_1", "name": "grep", "arguments": {"pattern": "x"}}
        ]  # type: ignore[attr-defined]

        request = ChatRequest(
            messages=[
                asst,
                Message(role="user", content="No tool result"),
            ]
        )

        asyncio.run(provider.complete(request))

        # Should have injected a synthetic result after the assistant message
        assert len(request.messages) == 3  # asst + synthetic + user
        assert request.messages[1].role == "tool"
        assert request.messages[1].tool_call_id == "call_1"

        # Should have emitted repair event
        repair_events = [
            e for e in coord.hooks.events if e[0] == "provider:tool_sequence_repaired"
        ]
        assert len(repair_events) == 1
        assert repair_events[0][1]["provider"] == "litellm"
        assert repair_events[0][1]["repair_count"] == 1
        assert repair_events[0][1]["repairs"][0]["tool_call_id"] == "call_1"
        assert repair_events[0][1]["repairs"][0]["tool_name"] == "grep"

        # Should have tracked the ID
        assert "call_1" in provider._repaired_tool_ids

    @patch("amplifier_module_provider_litellm.provider.litellm")
    def test_multiple_missing_results_all_repaired(self, mock_litellm: MagicMock):
        """Multiple missing tool results from one assistant message are all repaired."""
        mock_litellm.acompletion = AsyncMock(return_value=_litellm_response())
        _patch_litellm_error_classes(mock_litellm)

        coord = FakeCoordinator()
        provider = _make_provider(coordinator=coord)

        asst = Message(role="assistant", content="calling tools")
        asst.tool_calls = [  # type: ignore[attr-defined]
            {"id": "call_1", "name": "grep", "arguments": {}},
            {"id": "call_2", "name": "bash", "arguments": {}},
            {"id": "call_3", "name": "read_file", "arguments": {}},
        ]

        request = ChatRequest(
            messages=[
                asst,
                Message(role="user", content="No results"),
            ]
        )

        asyncio.run(provider.complete(request))

        # All 3 synthetics injected after assistant
        assert len(request.messages) == 5  # asst + 3 synthetics + user
        for i in range(1, 4):
            assert request.messages[i].role == "tool"
        tool_ids = {request.messages[i].tool_call_id for i in range(1, 4)}
        assert tool_ids == {"call_1", "call_2", "call_3"}

        # All tracked
        assert provider._repaired_tool_ids == {"call_1", "call_2", "call_3"}

        # Single repair event with count 3
        repair_events = [
            e for e in coord.hooks.events if e[0] == "provider:tool_sequence_repaired"
        ]
        assert len(repair_events) == 1
        assert repair_events[0][1]["repair_count"] == 3

    @patch("amplifier_module_provider_litellm.provider.litellm")
    def test_repaired_ids_not_redetected(self, mock_litellm: MagicMock):
        """Already-repaired IDs don't trigger repair on subsequent calls."""
        mock_litellm.acompletion = AsyncMock(return_value=_litellm_response())
        _patch_litellm_error_classes(mock_litellm)

        coord = FakeCoordinator()
        provider = _make_provider(coordinator=coord)

        asst = Message(role="assistant", content="calling tool")
        asst.tool_calls = [{"id": "call_1", "name": "grep", "arguments": {}}]  # type: ignore[attr-defined]

        # First call — triggers repair
        request1 = ChatRequest(
            messages=[
                asst,
                Message(role="user", content="No result"),
            ]
        )
        asyncio.run(provider.complete(request1))
        assert "call_1" in provider._repaired_tool_ids

        # Clear events
        coord.hooks.events.clear()

        # Second call with SAME missing ID — should NOT trigger repair again
        asst2 = Message(role="assistant", content="calling tool again")
        asst2.tool_calls = [{"id": "call_1", "name": "grep", "arguments": {}}]  # type: ignore[attr-defined]

        request2 = ChatRequest(
            messages=[
                asst2,
                Message(role="user", content="Still no result"),
            ]
        )
        asyncio.run(provider.complete(request2))

        # No repair event on second call
        repair_events = [
            e for e in coord.hooks.events if e[0] == "provider:tool_sequence_repaired"
        ]
        assert len(repair_events) == 0

    @patch("amplifier_module_provider_litellm.provider.litellm")
    def test_no_repair_when_all_results_present(self, mock_litellm: MagicMock):
        """No repair when all tool calls have matching results."""
        mock_litellm.acompletion = AsyncMock(return_value=_litellm_response())
        _patch_litellm_error_classes(mock_litellm)

        coord = FakeCoordinator()
        provider = _make_provider(coordinator=coord)

        asst = Message(role="assistant", content="calling tools")
        asst.tool_calls = [  # type: ignore[attr-defined]
            {"id": "call_1", "name": "grep", "arguments": {}},
            {"id": "call_2", "name": "bash", "arguments": {}},
        ]

        request = ChatRequest(
            messages=[
                asst,
                Message(role="tool", content="result 1", tool_call_id="call_1"),
                Message(role="tool", content="result 2", tool_call_id="call_2"),
                Message(role="user", content="Great"),
            ]
        )

        asyncio.run(provider.complete(request))

        # No messages inserted
        assert len(request.messages) == 4
        # No repair event
        repair_events = [
            e for e in coord.hooks.events if e[0] == "provider:tool_sequence_repaired"
        ]
        assert len(repair_events) == 0

    @patch("amplifier_module_provider_litellm.provider.litellm")
    def test_position_sensitive_insertion(self, mock_litellm: MagicMock):
        """Synthetic results are inserted at the correct position (after assistant)."""
        mock_litellm.acompletion = AsyncMock(return_value=_litellm_response())
        _patch_litellm_error_classes(mock_litellm)

        provider = _make_provider(coordinator=FakeCoordinator())

        # Two assistant messages, each with missing results
        asst1 = Message(role="assistant", content="first call")
        asst1.tool_calls = [{"id": "call_a", "name": "grep", "arguments": {}}]  # type: ignore[attr-defined]

        asst2 = Message(role="assistant", content="second call")
        asst2.tool_calls = [{"id": "call_b", "name": "bash", "arguments": {}}]  # type: ignore[attr-defined]

        request = ChatRequest(
            messages=[
                Message(role="user", content="start"),  # idx 0
                asst1,  # idx 1
                Message(role="user", content="middle"),  # idx 2
                asst2,  # idx 3
                Message(role="user", content="end"),  # idx 4
            ]
        )

        asyncio.run(provider.complete(request))

        # After repair: user, asst1, synthetic_a, user, asst2, synthetic_b, user
        assert len(request.messages) == 7
        assert request.messages[0].role == "user"
        assert request.messages[1].role == "assistant"  # asst1
        assert request.messages[2].role == "tool"  # synthetic for call_a
        assert request.messages[2].tool_call_id == "call_a"
        assert request.messages[3].role == "user"  # middle
        assert request.messages[4].role == "assistant"  # asst2
        assert request.messages[5].role == "tool"  # synthetic for call_b
        assert request.messages[5].tool_call_id == "call_b"
        assert request.messages[6].role == "user"  # end

    @patch("amplifier_module_provider_litellm.provider.litellm")
    def test_no_tool_calls_fast_path(self, mock_litellm: MagicMock):
        """Conversations with no tool calls at all skip repair entirely."""
        mock_litellm.acompletion = AsyncMock(return_value=_litellm_response())
        _patch_litellm_error_classes(mock_litellm)

        coord = FakeCoordinator()
        provider = _make_provider(coordinator=coord)

        request = ChatRequest(
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there"),
                Message(role="user", content="How are you?"),
            ]
        )

        asyncio.run(provider.complete(request))

        # No messages inserted, no events
        assert len(request.messages) == 3
        repair_events = [
            e for e in coord.hooks.events if e[0] == "provider:tool_sequence_repaired"
        ]
        assert len(repair_events) == 0
