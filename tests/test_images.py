"""Offline fixtures: Core boundary and real LiteLLM payload translation, not server acceptance."""

from __future__ import annotations

import base64
import copy
import json
import socket
import traceback
from importlib.metadata import version
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import litellm
import pytest
from amplifier_core.llm_errors import InvalidRequestError
from amplifier_core.message_models import (
    ChatRequest, ImageBlock, Message, TextBlock, ToolCallBlock, ToolResultBlock,
)

from amplifier_module_provider_litellm._images import INVALID, UNAVAILABLE, diagnostic_copy
from amplifier_module_provider_litellm.provider import LiteLLMProvider, _to_litellm_messages

# Complete 1x1 PNG; no decoder or image dependency is needed.
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aJ1kAAAAASUVORK5CYII="
# The local model metadata snapshots differ (the current SDK retires old IDs).
# Exercise each SDK's actual known vision model, never a made-up capability.
MODEL = (
    "anthropic/claude-3-5-sonnet-20241022"
    if version("litellm") == "1.53.2"
    else "anthropic/claude-sonnet-4-5"
)


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Image fixtures must not contact a real server")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    if hasattr(litellm, "telemetry"):
        monkeypatch.setattr(litellm, "telemetry", False)


def image(**source):
    return ImageBlock(source={"type": "base64", "media_type": "image/png", "data": PNG, **source})


def history(content, *, nested=False):
    assistant = Message(role="assistant", content=[
        ToolCallBlock(id="read-1", name="read_file", input={"path": "fixture.png"}),
    ])
    result = (
        Message(role="user", content=[ToolResultBlock(tool_call_id="read-1", output=content)])
        if nested else Message(role="tool", tool_call_id="read-1", content=content)
    )
    return ChatRequest(model=MODEL, messages=[assistant, result])


def response():
    return litellm.ModelResponse(
        model=MODEL, choices=[{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("persisted", [False, True])
def test_original_tool_slot_typed_and_nested(nested, persisted):
    blocks = [TextBlock(text="before"), image(), TextBlock(text="after"), image()]
    request = history(
        {"content": [b.model_dump() for b in blocks]} if nested else blocks,
        nested=nested,
    )
    if persisted:
        request = ChatRequest.model_validate(request.model_dump())
    snapshot = request.model_dump()
    messages = _to_litellm_messages(request)
    assert [m["role"] for m in messages] == ["assistant", "tool"]
    assert messages[0]["tool_calls"][0]["id"] == messages[1]["tool_call_id"] == "read-1"
    assert [b["type"] for b in messages[1]["content"]] == ["text", "image_url", "text", "image_url"]
    assert messages[1]["content"][1]["image_url"]["url"] == f"data:image/png;base64,{PNG}"
    assert request.model_dump() == snapshot
    messages[1]["content"][0]["text"] = "changed"
    assert request.model_dump() == snapshot


def test_direct_image_order_and_exact_bytes():
    request = ChatRequest(model="openai/gpt-4o", messages=[
        Message(role="user", content=[TextBlock(text="one"), image(), TextBlock(text="two"), image()]),
    ])
    parts = _to_litellm_messages(request)[0]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url", "text", "image_url"]
    assert base64.b64decode(parts[1]["image_url"]["url"].split(",", 1)[1]) == base64.b64decode(PNG)


@pytest.mark.parametrize("model", ["openai/gpt-4o", "openai/gpt-3.5-turbo", "unknown/private-alias"])
def test_unsupported_tool_slot_does_not_move_or_leak_pixels(model):
    request = history([TextBlock(text="before"), image(), TextBlock(text="after")])
    messages = _to_litellm_messages(request, model=model)
    assert [m["role"] for m in messages] == ["assistant", "tool"]
    assert messages[1]["content"] == f"before\n{UNAVAILABLE}\nafter"
    assert PNG not in json.dumps(messages)


def test_old_sdk_tool_images_truthfully_unavailable(monkeypatch):
    monkeypatch.setattr("amplifier_module_provider_litellm.provider.version", lambda _: "1.50.0")
    messages = _to_litellm_messages(history([image()]))
    assert messages[1]["content"] == UNAVAILABLE
    assert PNG not in json.dumps(messages)


@pytest.mark.parametrize("source", [
    {"data": "PRIVATE bad base64"}, {"media_type": "image/svg+xml"},
    {"type": "file", "path": "/private/image.png"},
    {"type": "url", "url": "file:///private/image.png"},
    {"type": "url", "url": "https://user:password@example.com/private.png"},
])
def test_malformed_image_retains_later_text_and_no_sensitive_repr(source):
    request = history([image(**source), TextBlock(text="still here")])
    messages = _to_litellm_messages(request)
    assert messages[1]["content"] == f"{INVALID}\nstill here"
    assert "PRIVATE" not in json.dumps(messages)


def test_remote_url_preserved_without_fetch(monkeypatch):
    block = ImageBlock(source={"type": "url", "url": "https://example.invalid/fixture.png"})
    messages = _to_litellm_messages(history([block]))
    assert messages[1]["content"] == [
        {"type": "image_url", "image_url": {"url": "https://example.invalid/fixture.png"}},
    ]


def test_json_output_never_stringifies_nested_image():
    request = history({"unrecognized": {"nested": image().model_dump()}}, nested=True)
    result = _to_litellm_messages(request)
    assert PNG not in json.dumps(result)
    assert "image omitted" in result[1]["content"]


def test_tool_role_typed_results_use_block_call_id():
    request = history([image()])
    request.messages[1] = Message(role="tool", content=[
        ToolResultBlock(tool_call_id="read-1", output=[image()]),
    ])
    assert LiteLLMProvider()._find_missing_tool_results(request.messages) == []
    messages = _to_litellm_messages(request)
    assert messages[1]["tool_call_id"] == "read-1"
    assert PNG in json.dumps(messages[1]["content"])


def test_tool_role_nested_result_with_adjacent_text_stays_one_slot():
    request = history([image()])
    request.messages[1] = Message(role="tool", content=[
        TextBlock(text="before"),
        ToolResultBlock(tool_call_id="read-1", output=[image()]),
        TextBlock(text="after"),
    ])
    messages = _to_litellm_messages(request)
    assert [m["role"] for m in messages] == ["assistant", "tool"]
    assert messages[1]["tool_call_id"] == "read-1"
    assert [b["type"] for b in messages[1]["content"]] == ["text", "image_url", "text"]


@pytest.mark.parametrize("output", [
    {"ok": False, "error": "truncated", "content": []},
    {"status": "failure", "content": [{"type": "text", "text": "partial"}]},
])
def test_envelope_failure_metadata_is_not_discarded(output):
    messages = _to_litellm_messages(history(output, nested=True))
    assert "false" in messages[1]["content"] or "failure" in messages[1]["content"]
    if output["content"]:
        assert "partial" in messages[1]["content"]


def test_ordinary_arguments_are_not_diagnostic_redacted_in_model_input():
    request = history({
        "arguments": "--dry-run", "ok": False,
        "content": [{"type": "text", "text": "partial"}],
    }, nested=True)
    text = _to_litellm_messages(request)[1]["content"]
    assert "--dry-run" in text and "false" in text and "partial" in text


@pytest.mark.asyncio
async def test_serialized_tool_output_becomes_typed_image_and_safe_diagnostics(monkeypatch):
    output = json.dumps({"content": [image().model_dump()]})
    request = history(output, nested=True)
    snapshot = request.model_dump()
    sdk = AsyncMock(return_value=response())
    monkeypatch.setattr(litellm, "acompletion", sdk)

    async def emit(name, payload):
        assert PNG not in json.dumps(payload, default=str)

    provider = LiteLLMProvider({"use_streaming": False, "raw": True}, SimpleNamespace(hooks=SimpleNamespace(emit=emit)))
    await provider.complete(request)
    assert sdk.call_args.kwargs["messages"][1]["content"] == [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}},
    ]
    assert request.model_dump() == snapshot


@pytest.mark.asyncio
async def test_serialized_tool_arguments_redacted_before_emit(monkeypatch):
    request = history("plain result")
    request.messages[0].content[0].input = {"image": image().model_dump()}
    sdk = AsyncMock(return_value=response())
    monkeypatch.setattr(litellm, "acompletion", sdk)

    async def emit(name, payload):
        assert PNG not in json.dumps(payload, default=str)

    provider = LiteLLMProvider({"use_streaming": False, "raw": True}, SimpleNamespace(hooks=SimpleNamespace(emit=emit)))
    await provider.complete(request)
    assert PNG in sdk.call_args.kwargs["messages"][0]["tool_calls"][0]["function"]["arguments"]


def test_real_anthropic_sdk_payload_builder():
    messages = _to_litellm_messages(history([TextBlock(text="before"), image(), TextBlock(text="after")]))
    try:
        from litellm.litellm_core_utils.prompt_templates.factory import convert_to_anthropic_tool_result
    except ImportError:  # LiteLLM 1.53.2's original module location
        from litellm.llms.prompt_templates.factory import convert_to_anthropic_tool_result
    result = convert_to_anthropic_tool_result(messages[1])
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "read-1"
    assert [b["type"] for b in result["content"]] == ["text", "image", "text"]
    assert result["content"][1]["source"]["data"] == PNG
    assert result["content"][1]["source"]["media_type"] == "image/png"


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True])
async def test_complete_has_no_false_repair_and_diagnostics_detached(monkeypatch, nested):
    request = history([TextBlock(text="before"), image()], nested=nested)
    snapshot = request.model_dump()
    emitted = []

    async def emit(name, payload):
        assert PNG not in json.dumps(payload, default=str)
        emitted.append((name, copy.deepcopy(payload)))
        if name == "llm:request":
            payload["raw"]["messages"].clear()  # A hook must not mutate the SDK request.

    sdk = AsyncMock(return_value=response())
    monkeypatch.setattr(litellm, "acompletion", sdk)
    provider = LiteLLMProvider({"use_streaming": False, "raw": True}, SimpleNamespace(hooks=SimpleNamespace(emit=emit)))
    await provider.complete(request)
    actual = sdk.call_args.kwargs["messages"]
    assert len(actual) == 2 and PNG in json.dumps(actual)
    assert not any(name == "provider:tool_sequence_repaired" for name, _ in emitted)
    assert request.model_dump() == snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_image_error_diagnostics_omit_echoed_payload(monkeypatch, caplog, stream):
    exc = litellm.BadRequestError(
        message=f"echo {PNG}", model=MODEL, llm_provider="anthropic",
    )
    exc.body = {"request": image().model_dump()}
    sdk = AsyncMock(side_effect=exc)
    monkeypatch.setattr(litellm, "acompletion", sdk)
    emitted = []

    async def emit(name, payload):
        emitted.append((name, payload))

    provider = LiteLLMProvider({"use_streaming": stream, "raw": True}, SimpleNamespace(hooks=SimpleNamespace(emit=emit)))
    with pytest.raises(InvalidRequestError) as caught:
        await provider.complete(history([image()]))
    assert PNG not in str(caught.value)
    assert PNG not in "".join(traceback.format_exception(caught.value))
    assert PNG not in json.dumps(emitted, default=str)
    assert PNG not in caplog.text
    assert sdk.await_count == 1  # Bad requests must not retry.


def test_plain_tool_text_after_malformed_request():
    assert _to_litellm_messages(history([image(data="bad")]))[1]["content"] == INVALID
    assert _to_litellm_messages(history("plain output"))[1]["content"] == "plain output"


@pytest.mark.asyncio
async def test_image_rate_limit_retry_preserves_payload_and_safe_event(monkeypatch):
    error = litellm.RateLimitError(message=f"echo {PNG}", model=MODEL, llm_provider="anthropic")
    sdk = AsyncMock(side_effect=[error, response()])
    monkeypatch.setattr(litellm, "acompletion", sdk)
    events = []

    async def emit(name, payload):
        assert PNG not in json.dumps(payload, default=str)
        events.append((name, payload))

    provider = LiteLLMProvider({
        "use_streaming": False, "raw": True, "max_retries": 1,
        "min_retry_delay": 0, "max_retry_delay": 0, "retry_jitter": False,
    }, SimpleNamespace(hooks=SimpleNamespace(emit=emit)))
    await provider.complete(history([image()]))
    assert sdk.await_count == 2
    assert sdk.call_args_list[0].kwargs["messages"] == sdk.call_args_list[1].kwargs["messages"]
    assert len([e for e in events if e[0] == "provider:retry"]) == 1


@pytest.mark.asyncio
async def test_image_midstream_failure_never_retries_and_omits_echo(monkeypatch):
    from amplifier_core.llm_errors import LLMError

    async def chunks():
        yield SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="partial", reasoning_content=None, tool_calls=None),
            finish_reason=None,
        )], usage=None)
        raise RuntimeError(f"echo {PNG}")

    sdk = AsyncMock(return_value=chunks())
    monkeypatch.setattr(litellm, "acompletion", sdk)
    events = []

    async def emit(name, payload):
        assert PNG not in json.dumps(payload, default=str)
        events.append((name, payload))

    provider = LiteLLMProvider({"raw": True}, SimpleNamespace(hooks=SimpleNamespace(emit=emit)))
    with pytest.raises(LLMError) as caught:
        await provider.complete(history([image()]))
    assert not caught.value.retryable
    assert PNG not in "".join(traceback.format_exception(caught.value))
    assert sdk.await_count == 1
    assert any(e[0] == "llm:stream_aborted" for e in events)


def test_diagnostic_copy_does_not_mutate_payload():
    payload = {"messages": [{"role": "tool", "content": [image().model_dump()]}]}
    snapshot = copy.deepcopy(payload)
    assert PNG not in json.dumps(diagnostic_copy(payload))
    assert payload == snapshot


@pytest.mark.asyncio
async def test_actual_litellm_http_transport_payload_no_real_server(monkeypatch):
    """Run real acompletion -> Anthropic adapter -> HTTPX mock transport."""
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    captured = []

    async def handle(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, request=request, json={
            "id": "msg_fixture", "type": "message", "role": "assistant",
            "model": MODEL.split("/", 1)[1], "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    # A fake key and intercepted transport only; no credential/environment read.
    client = AsyncHTTPHandler()
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    provider = LiteLLMProvider({
        "api_key": "fixture-not-a-secret", "api_base": "https://fixture.invalid",
        "use_streaming": False, "max_retries": 0,
        "extra_request_params": {
            "client": client, "max_tokens": 8,
            "tools": [{"type": "function", "function": {
                "name": "read_file", "description": "Read fixture",
                "parameters": {"type": "object", "properties": {}},
            }}],
        },
    })
    try:
        request = history([TextBlock(text="before"), image()])
        request.messages.insert(0, Message(role="user", content="Read the fixture image."))
        snapshot = request.model_dump()
        result = await provider.complete(request)
    finally:
        await client.client.aclose()
    assert result.content[0].text == "ok"
    assert len(captured) == 1
    assert request.model_dump() == snapshot
    native_result = captured[0]["messages"][2]["content"][0]
    assert native_result["type"] == "tool_result"
    assert native_result["tool_use_id"] == "read-1"
    assert native_result["content"][1]["source"]["data"] == PNG


@pytest.mark.asyncio
@pytest.mark.parametrize("override", [
    {"model": "openai/gpt-4o"},
    {"custom_llm_provider": "openai"},
])
async def test_effective_route_override_prevents_tool_pixels(monkeypatch, override):
    sdk = AsyncMock(return_value=response())
    monkeypatch.setattr(litellm, "acompletion", sdk)
    provider = LiteLLMProvider({"use_streaming": False, "extra_request_params": override})
    await provider.complete(history([image()]))
    messages = sdk.call_args.kwargs["messages"]
    assert messages[1]["content"] == UNAVAILABLE
    assert PNG not in json.dumps(messages)