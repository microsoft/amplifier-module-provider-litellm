"""Unit tests for the litellm provider streaming path (TDD).

Uses fake litellm.acompletion (monkeypatched) returning a fake async chunk
generator (OpenAI-style ModelResponse chunks).  All tests must pass after the
streaming implementation is added to provider.py.

Contract reference: docs/provider-streaming-contract.md
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from amplifier_module_provider_litellm.provider import LiteLLMProvider


# ---------------------------------------------------------------------------
# Fake chunk types  (mimic litellm streaming chunks / OpenAI ModelResponse)
# ---------------------------------------------------------------------------


class _FakeDelta:
    """Mimic a streaming delta object from litellm.

    reasoning_content is intentionally absent from some instances to match
    litellm's real behaviour: non-thinking backends don't include the attribute
    at all, so ``getattr(delta, "reasoning_content", None)`` returns None.
    """

    _MISSING = object()

    def __init__(
        self,
        content=None,
        reasoning_content=_MISSING,
        tool_calls=None,
    ):
        self.content = content
        self.tool_calls = tool_calls
        if reasoning_content is not _FakeDelta._MISSING:
            self.reasoning_content = reasoning_content


class _FakeChoice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class _FakeUsageDetails:
    def __init__(self, reasoning_tokens=None):
        self.reasoning_tokens = reasoning_tokens


class _FakeUsage:
    def __init__(self, prompt_tokens=10, completion_tokens=5, reasoning_tokens=None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        self.prompt_tokens_details = None
        self.completion_tokens_details = (
            _FakeUsageDetails(reasoning_tokens) if reasoning_tokens is not None else None
        )


class _FakeChunk:
    def __init__(self, choices=None, usage=None, model="openai/gpt-4o"):
        self.choices = choices or []
        self.usage = usage
        self.model = model


class _FakeTCFunction:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class _FakeTCDelta:
    """Mimic a tool_call entry inside a streaming delta."""

    def __init__(self, index, id=None, function_name=None, function_args=None):
        self.index = index
        self.id = id
        self.function = _FakeTCFunction(name=function_name, arguments=function_args)


# ---------------------------------------------------------------------------
# Chunk factory helpers
# ---------------------------------------------------------------------------


def _text_chunk(text, finish_reason=None):
    delta = _FakeDelta(content=text)
    return _FakeChunk(choices=[_FakeChoice(delta, finish_reason=finish_reason)])


def _thinking_chunk(thinking_text):
    delta = _FakeDelta(reasoning_content=thinking_text)
    return _FakeChunk(choices=[_FakeChoice(delta)])


def _usage_chunk(prompt_tokens=10, completion_tokens=5, reasoning_tokens=None):
    """Usage-only final chunk (empty choices list)."""
    usage = _FakeUsage(prompt_tokens, completion_tokens, reasoning_tokens)
    return _FakeChunk(choices=[], usage=usage)


def _tool_chunk(index, id=None, name=None, args_fragment=None, finish_reason=None):
    tc = _FakeTCDelta(index=index, id=id, function_name=name, function_args=args_fragment)
    delta = _FakeDelta(tool_calls=[tc])
    return _FakeChunk(choices=[_FakeChoice(delta, finish_reason=finish_reason)])


def _fake_acompletion(*items):
    """
    Create a fake ``litellm.acompletion`` coroutine that returns an async generator.

    *items* may be _FakeChunk objects or Exception instances.
    Exceptions are raised when encountered during iteration (mid-stream simulation).
    """

    async def _acompletion(**kwargs):
        async def _gen():
            for item in items:
                if isinstance(item, BaseException):
                    raise item
                yield item

        return _gen()

    return _acompletion


# ---------------------------------------------------------------------------
# Fake coordinator / hooks — captures all emitted events
# ---------------------------------------------------------------------------


class _FakeHooks:
    def __init__(self):
        self.events = []

    async def emit(self, name, payload):
        self.events.append((name, payload))


class _FakeCoordinator:
    def __init__(self):
        self.hooks = _FakeHooks()


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _events(coordinator, name):
    """Return payloads for all events with the given name."""
    return [p for n, p in coordinator.hooks.events if n == name]


def _event_names(coordinator):
    return [n for n, _ in coordinator.hooks.events]


def _make_provider(config=None):
    coordinator = _FakeCoordinator()
    cfg = {"model": "openai/gpt-4o", **(config or {})}
    provider = LiteLLMProvider(cfg, coordinator=coordinator)
    return provider, coordinator


def _make_request(metadata=None):
    """Minimal ChatRequest mock with explicit metadata (not an auto-MagicMock value)."""
    request = MagicMock()
    request.model = "openai/gpt-4o"
    request.messages = []
    request.tools = None
    request.max_output_tokens = 100
    request.temperature = 0.0
    request.metadata = metadata  # explicit — prevents MagicMock truthy interference
    del request.reasoning_effort  # getattr(..., None) returns None
    del request.response_format   # same
    return request


def _patch_litellm_errors(mock_litellm):
    """Set all litellm error classes to a non-matching sentinel."""

    class _Never(Exception):
        pass

    for name in (
        "AuthenticationError",
        "PermissionDeniedError",
        "RateLimitError",
        "ContextWindowExceededError",
        "ContentPolicyViolationError",
        "BadRequestError",
        "ServiceUnavailableError",
        "NotFoundError",
        "APIConnectionError",
        "Timeout",
    ):
        setattr(mock_litellm, name, _Never)


# ===========================================================================
# Tests
# ===========================================================================


class TestStreamingConfig:
    """Provider configuration for streaming."""

    def test_use_streaming_defaults_to_true(self):
        provider = LiteLLMProvider()
        assert provider.use_streaming is True

    def test_use_streaming_false_via_config(self):
        provider = LiteLLMProvider({"use_streaming": False})
        assert provider.use_streaming is False

    def test_streaming_in_capabilities(self):
        provider = LiteLLMProvider()
        info = provider.get_info()
        assert "streaming" in info.capabilities


class TestStreamingKwargs:
    """Correct kwargs forwarded to litellm.acompletion in streaming mode."""

    @pytest.mark.asyncio
    async def test_stream_true_and_include_usage_forwarded(self):
        """stream=True and stream_options={"include_usage": True} are passed."""
        captured = {}

        async def capturing_acompletion(**kwargs):
            captured.update(kwargs)

            async def gen():
                yield _text_chunk("hi")
                yield _usage_chunk()

            return gen()

        provider, _ = _make_provider()
        request = _make_request()

        with patch("amplifier_module_provider_litellm.provider.litellm") as m:
            _patch_litellm_errors(m)
            m.acompletion = capturing_acompletion
            await provider.complete(request)

        assert captured.get("stream") is True
        assert captured.get("stream_options") == {"include_usage": True}


class TestTextStream:
    """Basic text streaming: block_start -> deltas -> block_end."""

    @pytest.mark.asyncio
    async def test_text_stream_event_sequence(self):
        """block_start(text) -> stream_block_delta*3 -> block_end, then llm:response."""
        chunks = [
            _text_chunk("Hello"),
            _text_chunk(", "),
            _text_chunk("world!"),
            _usage_chunk(prompt_tokens=10, completion_tokens=3),
        ]
        provider, coordinator = _make_provider()
        request = _make_request()

        with patch("amplifier_module_provider_litellm.provider.litellm") as m:
            _patch_litellm_errors(m)
            m.acompletion = _fake_acompletion(*chunks)
            result = await provider.complete(request)

        names = _event_names(coordinator)
        assert names == [
            "llm:request",
            "llm:stream_block_start",
            "llm:stream_block_delta",
            "llm:stream_block_delta",
            "llm:stream_block_delta",
            "llm:stream_block_end",
            "llm:response",
        ]

        starts = _events(coordinator, "llm:stream_block_start")
        assert len(starts) == 1
        assert starts[0]["block_type"] == "text"
        assert starts[0]["block_index"] == 0

        deltas = _events(coordinator, "llm:stream_block_delta")
        assert [d["text"] for d in deltas] == ["Hello", ", ", "world!"]
        assert [d["sequence"] for d in deltas] == [0, 1, 2]
        assert all(d["block_index"] == 0 for d in deltas)
        assert all(d["block_type"] == "text" for d in deltas)

        ends = _events(coordinator, "llm:stream_block_end")
        assert len(ends) == 1
        assert ends[0]["block_type"] == "text"
        assert ends[0]["block_index"] == 0

        assert len(result.content) == 1
        assert result.content[0].text == "Hello, world!"

    @pytest.mark.asyncio
    async def test_empty_and_none_fragments_not_emitted(self):
        """Empty string and None content produce no delta events."""
        chunks = [
            _text_chunk(None),
            _text_chunk(""),
            _text_chunk("x"),
            _usage_chunk(),
        ]
        provider, coordinator = _make_provider()
        request = _make_request()

        with patch("amplifier_module_provider_litellm.provider.litellm") as m:
            _patch_litellm_errors(m)
            m.acompletion = _fake_acompletion(*chunks)
            await provider.complete(request)

        deltas = _events(coordinator, "llm:stream_block_delta")
        assert len(deltas) == 1
        assert deltas[0]["text"] == "x"

    @pytest.mark.asyncio
    async def test_single_request_id_across_all_stream_events(self):
        """All stream events for one call share exactly one request_id (uuid4)."""
        chunks = [_text_chunk("Hi"), _usage_chunk()]
        provider, coordinator = _make_provider()
        request = _make_request()

        with patch("amplifier_module_provider_litellm.provider.litellm") as m:
            _patch_litellm_errors(m)
            m.acompletion = _fake_acompletion(*chunks)
            await provider.complete(request)

        stream_events = [(n, p) for n, p in coordinator.hooks.events if n.startswith("llm:stream_")]
        assert len(stream_events) > 0

        ids = {p["request_id"] for _, p in stream_events}
        assert len(ids) == 1, f"Expected one request_id, got {ids}"
        uuid.UUID(next(iter(ids)), version=4)

    @pytest.mark.asyncio
    async def test_llm_request_and_response_emitted_for_streaming_calls(self):
        """llm:request and llm:response are emitted in streaming mode."""
        chunks = [_text_chunk("hi"), _usage_chunk()]
        provider, coordinator = _make_provider()
        request = _make_request()

        with patch("amplifier_module_provider_litellm.provider.litellm") as m:
            _patch_litellm_errors(m)
            m.acompletion = _fake_acompletion(*chunks)
            await provider.complete(request)

        assert "llm:request" in _event_names(coordinator)
        assert "llm:response" in _event_names(coordinator)
        resp = _events(coordinator, "llm:response")[0]
        assert resp["status"] == "ok"
        assert resp["provider"] == "litellm"
        assert "duration_ms" in resp


class TestThinkingStream:
    """Thinking (reasoning) blocks via reasoning_content."""

    @pytest.mark.asyncio
    async def test_thinking_only_stream(self):
        """reasoning_content chunks -> thinking block_start + block_delta(block_type=thinking) events."""
        chunks = [
            _thinking_chunk("Step 1..."),
            _thinking_chunk(" Step 2..."),
            _usage_chunk(),
        ]
        provider, coordinator = _make_provider()
        request = _make_request()

        with patch("amplifier_module_provider_litellm.provider.litellm") as m:
            _patch_litellm_errors(m)
            m.acompletion = _fake_acompletion(*chunks)
            result = await provider.complete(request)

        starts = _events(coordinator, "llm:stream_block_start")
        assert len(starts) == 1
        assert starts[0]["block_type"] == "thinking"
        assert starts[0]["block_index"] == 0

        # Contract: ONE block_delta event for all content; block_type carries the distinction.
        # No llm:stream_thinking_delta — only llm:stream_block_delta with block_type=="thinking".
        assert _events(coordinator, "llm:stream_thinking_delta") == []
        thinking_deltas = [
            d for d in _events(coordinator, "llm:stream_block_delta")
            if d["block_type"] == "thinking"
        ]
        assert len(thinking_deltas) == 2
        assert [d["sequence"] for d in thinking_deltas] == [0, 1]
        assert all(d["block_index"] == 0 for d in thinking_deltas)

        ends = _events(coordinator, "llm:stream_block_end")
        assert len(ends) == 1
        assert ends[0]["block_type"] == "thinking"

        from amplifier_core.message_models import ThinkingBlock

        assert len(result.content) == 1
        assert isinstance(result.content[0], ThinkingBlock)
        assert result.content[0].thinking == "Step 1... Step 2..."

    @pytest.mark.asyncio
    async def test_thinking_transitions_to_text(self):
        """Thinking->text: thinking block closed before text block opens."""
        chunks = [
            _thinking_chunk("reasoning"),
            _text_chunk("answer"),
            _usage_chunk(),
        ]
        provider, coordinator = _make_provider()
        request = _make_request()

        with patch("amplifier_module_provider_litellm.provider.litellm") as m:
            _patch_litellm_errors(m)
            m.acompletion = _fake_acompletion(*chunks)
            result = await provider.complete(request)

        starts = _events(coordinator, "llm:stream_block_start")
        ends = _events(coordinator, "llm:stream_block_end")
        assert len(starts) == 2
        assert len(ends) == 2

        assert starts[0]["block_type"] == "thinking"
        assert starts[0]["block_index"] == 0
        assert starts[1]["block_type"] == "text"
        assert starts[1]["block_index"] == 1

        # thinking block_end must precede text block_start in event order
        event_pairs = [(n, p.get("block_type")) for n, p in coordinator.hooks.events]
        thinking_end_pos = next(
            i for i, (n, bt) in enumerate(event_pairs)
            if n == "llm:stream_block_end" and bt == "thinking"
        )
        text_start_pos = next(
            i for i, (n, bt) in enumerate(event_pairs)
            if n == "llm:stream_block_start" and bt == "text"
        )
        assert thinking_end_pos < text_start_pos

        from amplifier_core.message_models import ThinkingBlock

        assert isinstance(result.content[0], ThinkingBlock)
        assert result.content[0].thinking == "reasoning"
        assert result.content[1].text == "answer"

    @pytest.mark.asyncio
    async def test_sequence_is_per_block_not_global(self):
        """sequence counter resets to 0 per block (not global)."""
        chunks = [
            _thinking_chunk("T1"),
            _thinking_chunk("T2"),
            _text_chunk("A"),
            _text_chunk("B"),
            _text_chunk("C"),
            _usage_chunk(),
        ]
        provider, coordinator = _make_provider()
        request = _make_request()

        with patch("amplifier_module_provider_litellm.provider.litellm") as m:
            _patch_litellm_errors(m)
            m.acompletion = _fake_acompletion(*chunks)
            await provider.complete(request)

        # Both thinking and text use llm:stream_block_delta; block_type distinguishes them.
        all_deltas = _events(coordinator, "llm:stream_block_delta")
        think_seqs = [d["sequence"] for d in all_deltas if d["block_type"] == "thinking"]
        text_seqs = [d["sequence"] for d in all_deltas if d["block_type"] == "text"]
        assert think_seqs == [0, 1], "thinking sequence must restart at 0"
        assert text_seqs == [0, 1, 2], "text sequence must restart at 0"

    @pytest.mark.asyncio
    async def test_block_index_is_shared_space(self):
        """block_index draws from one shared 0-based space across block types."""
        chunks = [_thinking_chunk("r"), _text_chunk("a"), _usage_chunk()]
        provider, coordinator = _make_provider()
        request = _make_request()

        with patch("amplifier_module_provider_litellm.provider.litellm") as m:
            _patch_litellm_errors(m)
            m.acompletion = _fake_acompletion(*chunks)
            await provider.complete(request)

        starts = _events(coordinator, "llm:stream_block_start")
        assert starts[0]["block_index"] == 0  # thinking
        assert starts[1]["block_index"] == 1  # text


class TestToolCallStream:
    """Tool-call blocks: block_start/block_end only, no arg deltas."""

    @pytest.mark.asyncio
    async def test_tool_call_emits_start_and_end_no_arg_deltas(self):
        """Tool call: block_start(tool_use, name) + block_end; no stream_block_delta."""
        chunks = [
            _tool_chunk(0, id="call_123", name="web_search", args_fragment=None),
            _tool_chunk(0, args_fragment='{"query":'),
            _tool_chunk(0, args_fragment='"litellm"}', finish_reason="tool_calls"),
            _usage_chunk(),
        ]
        provider, coordinator = _make_provider()
        request = _make_request()

        with patch("amplifier_module_provider_litellm.provider.litellm") as m:
            _patch_litellm_errors(m)
            m.acompletion = _fake_acompletion(*chunks)
            result = await provider.complete(request)

        starts = _events(coordinator, "llm:stream_block_start")
        assert len(starts) == 1
        assert starts[0]["block_type"] == "tool_use"
        assert starts[0]["name"] == "web_search"
        assert starts[0]["block_index"] == 0

        assert _events(coordinator, "llm:stream_block_delta") == []

        ends = _events(coordinator, "llm:stream_block_end")
        assert len(ends) == 1
        assert ends[0]["block_type"] == "tool_use"

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_123"
        assert result.tool_calls[0].name == "web_search"
        assert result.tool_calls[0].arguments == {"query": "litellm"}

    @pytest.mark.asyncio
    async def test_tool_block_start_includes_name(self):
        """block_start for tool_use includes 'name' when a name is present."""
        chunks = [
            _tool_chunk(0, id="call_1", name="my_tool"),
            _tool_chunk(0, args_fragment="{}", finish_reason="tool_calls"),
            _usage_chunk(),
        ]
        provider, coordinator = _make_provider()
        request = _make_request()

        with patch("amplifier_module_provider_litellm.provider.litellm") as m:
            _patch_litellm_errors(m)
            m.acompletion = _fake_acompletion(*chunks)
            await provider.complete(request)

        starts = _events(coordinator, "llm:stream_block_start")
        assert "name" in starts[0]
        assert starts[0]["name"] == "my_tool"


class TestNonStreamingFallback:
    """Non-streaming path emits no llm:stream_* events."""

    @pytest.mark.asyncio
    async def test_metadata_stream_false_disables_streaming(self):
        """metadata={"stream": False} -> non-streaming, zero llm:stream_* events."""
        provider, coordinator = _make_provider()
        request = _make_request(metadata={"stream": False})

        mock_resp = MagicMock()
        choice = MagicMock()
        choice.message.content = "hello"
        choice.message.tool_calls = None
        choice.finish_reason = "stop"
        mock_resp.choices = [choice]
        mock_resp.usage.prompt_tokens = 5
        mock_resp.usage.completion_tokens = 2
        mock_resp.usage.total_tokens = 7
        mock_resp.usage.prompt_tokens_details = None
        mock_resp.model = "openai/gpt-4o"

        with patch("amplifier_module_provider_litellm.provider.litellm") as m:
            _patch_litellm_errors(m)
            m.acompletion = AsyncMock(return_value=mock_resp)
            result = await provider.complete(request)

        stream_names = [n for n in _event_names(coordinator) if n.startswith("llm:stream_")]
        assert stream_names == [], f"Expected no stream events, got: {stream_names}"
        assert result.content[0].text == "hello"

    @pytest.mark.asyncio
    async def test_use_streaming_false_config_disables_streaming(self):
        """use_streaming=False in config -> non-streaming, zero llm:stream_* events."""
        provider, coordinator = _make_provider(config={"use_streaming": False})
        request = _make_request()

        mock_resp = MagicMock()
        choice = MagicMock()
        choice.message.content = "response"
        choice.message.tool_calls = None
        choice.finish_reason = "stop"
        mock_resp.choices = [choice]
        mock_resp.usage.prompt_tokens = 5
        mock_resp.usage.completion_tokens = 2
        mock_resp.usage.total_tokens = 7
        mock_resp.usage.prompt_tokens_details = None
        mock_resp.model = "openai/gpt-4o"

        with patch("amplifier_module_provider_litellm.provider.litellm") as m:
            _patch_litellm_errors(m)
            m.acompletion = AsyncMock(return_value=mock_resp)
            result = await provider.complete(request)

        stream_names = [n for n in _event_names(coordinator) if n.startswith("llm:stream_")]
        assert stream_names == []
        assert result.content[0].text == "response"

    @pytest.mark.asyncio
    async def test_metadata_stream_zero_does_not_disable_streaming(self):
        """metadata={"stream": 0} does NOT disable streaming (identity check, not ==)."""
        chunks = [_text_chunk("hi"), _usage_chunk()]
        provider, coordinator = _make_provider()
        request = _make_request(metadata={"stream": 0})  # 0 is not False by identity

        with patch("amplifier_module_provider_litellm.provider.litellm") as m:
            _patch_litellm_errors(m)
            m.acompletion = _fake_acompletion(*chunks)
            await provider.complete(request)

        # Streaming events were produced
        assert "llm:stream_block_start" in _event_names(coordinator)


class TestStreamAborted:
    """llm:stream_aborted fires only after a partial emit."""

    @pytest.mark.asyncio
    async def test_error_after_partial_emits_stream_aborted(self):
        """Exception after first delta -> llm:stream_aborted with error info."""
        error = RuntimeError("connection lost")
        provider, coordinator = _make_provider()
        request = _make_request()

        with patch("amplifier_module_provider_litellm.provider.litellm") as m:
            _patch_litellm_errors(m)
            m.acompletion = _fake_acompletion(_text_chunk("partial"), error)
            with pytest.raises(RuntimeError, match="connection lost"):
                await provider.complete(request)

        aborted = _events(coordinator, "llm:stream_aborted")
        assert len(aborted) == 1
        assert aborted[0]["error"]["type"] == "RuntimeError"
        assert aborted[0]["error"]["msg"] == "connection lost"
        assert "request_id" in aborted[0]

    @pytest.mark.asyncio
    async def test_error_before_any_delta_no_stream_aborted(self):
        """Exception before any delta -> NO llm:stream_aborted."""
        error = RuntimeError("fail immediately")
        provider, coordinator = _make_provider()
        request = _make_request()

        with patch("amplifier_module_provider_litellm.provider.litellm") as m:
            _patch_litellm_errors(m)
            m.acompletion = _fake_acompletion(error)
            with pytest.raises(RuntimeError, match="fail immediately"):
                await provider.complete(request)

        aborted = _events(coordinator, "llm:stream_aborted")
        assert len(aborted) == 0, "stream_aborted must not fire before any delta"

    @pytest.mark.asyncio
    async def test_stream_aborted_request_id_matches_block_events(self):
        """llm:stream_aborted.request_id matches all other stream event request_ids."""
        error = RuntimeError("dropped")
        provider, coordinator = _make_provider()
        request = _make_request()

        with patch("amplifier_module_provider_litellm.provider.litellm") as m:
            _patch_litellm_errors(m)
            m.acompletion = _fake_acompletion(_text_chunk("part"), error)
            with pytest.raises(RuntimeError):
                await provider.complete(request)

        stream_events = [(n, p) for n, p in coordinator.hooks.events if n.startswith("llm:stream_")]
        all_request_ids = {p["request_id"] for _, p in stream_events}
        assert len(all_request_ids) == 1


class TestUsageCapture:
    """Usage tokens from final chunk -> llm:response and ChatResponse."""

    @pytest.mark.asyncio
    async def test_usage_from_final_chunk(self):
        """Token counts from the usage-only chunk in llm:response and ChatResponse."""
        chunks = [
            _text_chunk("answer"),
            _usage_chunk(prompt_tokens=42, completion_tokens=7),
        ]
        provider, coordinator = _make_provider()
        request = _make_request()

        with patch("amplifier_module_provider_litellm.provider.litellm") as m:
            _patch_litellm_errors(m)
            m.acompletion = _fake_acompletion(*chunks)
            result = await provider.complete(request)

        resp_events = _events(coordinator, "llm:response")
        assert len(resp_events) == 1
        usage = resp_events[0]["usage"]
        assert usage["input"] == 42
        assert usage["output"] == 7

        assert result.usage.input_tokens == 42
        assert result.usage.output_tokens == 7
        assert result.usage.total_tokens == 49

    @pytest.mark.asyncio
    async def test_reasoning_tokens_from_completion_tokens_details(self):
        """completion_tokens_details.reasoning_tokens -> ChatResponse.usage.reasoning_tokens."""
        chunks = [
            _text_chunk("answer"),
            _usage_chunk(prompt_tokens=10, completion_tokens=20, reasoning_tokens=15),
        ]
        provider, coordinator = _make_provider()
        request = _make_request()

        with patch("amplifier_module_provider_litellm.provider.litellm") as m:
            _patch_litellm_errors(m)
            m.acompletion = _fake_acompletion(*chunks)
            result = await provider.complete(request)

        assert result.usage.reasoning_tokens == 15

    @pytest.mark.asyncio
    async def test_no_usage_chunk_gives_none_usage(self):
        """Stream with no usage chunk -> ChatResponse.usage is None."""
        chunks = [_text_chunk("hi")]
        provider, coordinator = _make_provider()
        request = _make_request()

        with patch("amplifier_module_provider_litellm.provider.litellm") as m:
            _patch_litellm_errors(m)
            m.acompletion = _fake_acompletion(*chunks)
            result = await provider.complete(request)

        assert result.usage is None
