"""Transport policy and capture options pass through generic provider dispatch."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ChatRequest, Message

from amplifier_module_provider_litellm import provider as module


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [None, 45])
@pytest.mark.parametrize("raw", [False, True])
async def test_wait_policy_and_full_redacted_capture(monkeypatch, timeout, raw):
    create = AsyncMock(
        return_value=SimpleNamespace(
            model="openai/gpt-4o",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="OK", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=1,
                total_tokens=11,
                prompt_tokens_details=None,
            ),
        )
    )
    monkeypatch.setattr(module.litellm, "acompletion", create)
    hooks = SimpleNamespace(emit=AsyncMock())
    p = module.LiteLLMProvider(
        {
            "default_model": "openai/gpt-4o",
            "timeout": timeout,
            "raw": raw,
            "api_key": "fake-api-key",
            "use_streaming": False,
        },
        coordinator=SimpleNamespace(hooks=hooks),
    )
    await p.complete(
        ChatRequest(
            messages=[Message(role="user", content="x" * 20000)],
            metadata={"stream": False},
        )
    )
    option = create.call_args.kwargs["timeout"]
    assert option == 45 if timeout else option.read is None and option.connect == 5
    event = next(
        c.args[1] for c in hooks.emit.call_args_list if c.args[0] == "llm:request"
    )
    assert ("raw" in event) is raw
    if raw:
        assert "api_key" not in event["raw"]
        assert event["raw"]["messages"][0]["content"] == "x" * 20000
