"""Small Core -> LiteLLM content boundary; never stringify image-bearing blocks."""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

UNAVAILABLE = "[Image unavailable: image support is not verified for this model/transport and message position.]"
INVALID = "[Image unavailable: invalid or unsupported image source.]"
REDACTED = "[image omitted]"
IMAGE_TYPES = frozenset({"image", "image_url", "input_image"})


def block_dict(value: Any) -> Any:
    return value.model_dump() if hasattr(value, "model_dump") else value


def contains_image(value: Any) -> bool:
    value = block_dict(value)
    if isinstance(value, Mapping):
        return value.get("type") in IMAGE_TYPES or any(
            contains_image(v) for v in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(contains_image(v) for v in value)
    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            return contains_image(json.loads(value))
        except (ValueError, RecursionError):
            pass
    return False


def diagnostic_copy(value: Any) -> Any:
    """Detached copy, omitting image sources *before* an event is emitted."""
    value = block_dict(value)
    if isinstance(value, Mapping):
        if value.get("type") in IMAGE_TYPES:
            return {"type": "text", "text": REDACTED}
        return {k: diagnostic_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [diagnostic_copy(v) for v in value]
    if isinstance(value, str):
        if value.lower().startswith("data:image/"):
            return REDACTED
        # Both tool arguments and persisted tool outputs may already be JSON.
        if value.lstrip().startswith(("{", "[")):
            try:
                parsed = json.loads(value)
                if contains_image(parsed):
                    return json.dumps(diagnostic_copy(parsed))
            except (ValueError, RecursionError):
                pass
    return value


def image_part(block: Mapping[str, Any], *, supported: bool) -> dict[str, Any]:
    """Translate inline bytes without resizing, or preserve a remote URL."""
    if not supported:
        return {"type": "text", "text": UNAVAILABLE}
    source = block.get("source")
    url: Any = None
    if block.get("type") in {"image_url", "input_image"}:
        image_url = block.get("image_url")
        url = image_url.get("url") if isinstance(image_url, Mapping) else image_url
    elif isinstance(source, Mapping):
        if source.get("type") == "base64":
            media_type, data = source.get("media_type"), source.get("data")
            if media_type in {"image/png", "image/jpeg", "image/gif", "image/webp"} and isinstance(data, str):
                try:
                    if data and base64.b64decode(data, validate=True):
                        url = f"data:{media_type};base64,{data}"
                except (ValueError, binascii.Error):
                    pass
        elif source.get("type") == "url":
            url = source.get("url")
    if isinstance(url, str):
        if url.startswith("data:image/"):
            try:
                header, data = url.split(";base64,", 1)
                if header[5:] in {"image/png", "image/jpeg", "image/gif", "image/webp"} and data and base64.b64decode(data, validate=True):
                    return {"type": "image_url", "image_url": {"url": url}}
            except (ValueError, binascii.Error):
                pass
        else:
            try:
                parsed = urlsplit(url)
                if parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password:
                    return {"type": "image_url", "image_url": {"url": url}}
            except ValueError:
                pass
    return {"type": "text", "text": INVALID}


def content_parts(content: Any, *, images: bool) -> list[dict[str, Any]]:
    """Known Core text/image/tool-result envelopes, including persisted dicts."""
    content = block_dict(content)
    if isinstance(content, str):
        if contains_image(content):
            # Only structured, image-bearing JSON is unwrapped. Ordinary tool
            # strings (including non-JSON "arguments" values) remain unchanged.
            return content_parts(json.loads(content), images=images)
        return [{"type": "text", "text": content}]
    if isinstance(content, (list, tuple)):
        return [part for item in content for part in content_parts(item, images=images)]
    if isinstance(content, Mapping):
        kind = content.get("type")
        if kind in IMAGE_TYPES:
            return [image_part(content, supported=images)]
        if kind == "text":
            text = content.get("text")
            return [{"type": "text", "text": text if isinstance(text, str) else ""}]
        if kind == "tool_result":
            return content_parts(content.get("output", content.get("content", "")), images=images)
        if "content" in content and isinstance(content["content"], (list, tuple)):
            siblings = {k: v for k, v in content.items() if k != "content"}
            metadata = (
                [{"type": "text", "text": json.dumps(diagnostic_copy(siblings), default=lambda _: "[unsupported value]")}]
                if siblings else []
            )
            return metadata + content_parts(content["content"], images=images)
        # Ordinary JSON tool output remains text; unrecognized nested images
        # must not enter it as pixels or a Python/Pydantic repr.
        return [{"type": "text", "text": json.dumps(diagnostic_copy(content), default=lambda _: "[unsupported value]")}]
    return [{"type": "text", "text": "" if content is None else str(content) if isinstance(content, (bool, int, float)) else "[unsupported content]"}]


def tool_content(content: Any, *, images: bool) -> str | list[dict[str, Any]]:
    if isinstance(content, str) and not contains_image(content):
        return content
    parts = content_parts(content, images=images)
    if any(part["type"] == "image_url" for part in parts):
        return parts
    return "\n".join(part["text"] for part in parts)