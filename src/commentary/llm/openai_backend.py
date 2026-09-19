"""OpenAI Responses API backend with strict Pydantic structured output.

The rest of the application deliberately speaks in Anthropic-shaped content
blocks.  This adapter keeps that contract local to the backend and translates
the blocks to the Responses API's ``input_text``/``input_image`` form.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from typing import Any, TypeVar

import openai
from pydantic import BaseModel, ValidationError

from commentary.llm.base import Block, LLMError, Parsed, Usage
from commentary.llm.pricing import cost_usd
from commentary.llm.schema import strict_schema

T = TypeVar("T", bound=BaseModel)


def responses_blocks(blocks: list[Block]) -> list[dict[str, Any]]:
    """Convert the application's Anthropic-shaped blocks to Responses input.

    Images are sent as data URLs so this adapter never needs to upload or
    expose a frame at a separate URL.
    """
    converted: list[dict[str, Any]] = []
    for block in blocks:
        kind = block.get("type")
        if kind == "text":
            converted.append({"type": "input_text", "text": str(block.get("text", ""))})
            continue
        if kind == "image":
            source = block.get("source")
            if not isinstance(source, dict) or source.get("type") != "base64":
                raise LLMError("OpenAI: image blocks must use a base64 source")
            media_type = source.get("media_type")
            data = source.get("data")
            if not isinstance(media_type, str) or not isinstance(data, str):
                raise LLMError("OpenAI: malformed base64 image block")
            converted.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{media_type};base64,{data}",
                }
            )
            continue
        raise LLMError(f"OpenAI: unsupported content block type {kind!r}")
    return converted


def _value(obj: Any, name: str, default: Any = None) -> Any:
    """Read both SDK objects and the small namespaces used by backend tests."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class OpenAIBackend:
    """Structured calls against OpenAI's Responses API."""

    def __init__(
        self,
        client: openai.AsyncOpenAI | None = None,
        *,
        timeout_s: float = 8.0,
        max_retries: int = 1,
        base_url: str | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        elif base_url:
            self._client = openai.AsyncOpenAI(
                timeout=timeout_s, max_retries=max_retries, base_url=base_url
            )
        else:
            self._client = openai.AsyncOpenAI(timeout=timeout_s, max_retries=max_retries)
        self._total = Usage()
        self._lock = asyncio.Lock()
        # Scoped callers such as recorded research may attach Responses-only
        # parameters (for example, web search) for one request.
        self.extra_params: dict[str, Any] = {}

    @property
    def total(self) -> Usage:
        return self._total

    async def parse(
        self,
        *,
        model: str,
        system: str,
        blocks: list[Block],
        output_format: type[T],
        max_tokens: int = 1024,
        effort: str | None = None,
        cache_system: bool = True,
        tag: str = "",
    ) -> Parsed[T]:
        del cache_system  # Responses prompt caching is automatic for matching prefixes.
        started = time.monotonic()
        params: dict[str, Any] = {
            "model": model,
            "instructions": system,
            "input": [{"role": "user", "content": responses_blocks(blocks)}],
            "max_output_tokens": max_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": output_format.__name__,
                    "strict": True,
                    "schema": strict_schema(output_format),
                }
            },
        }
        if effort is not None:
            params["reasoning"] = {"effort": effort}
        params.update(self.extra_params)

        try:
            response = await self._client.responses.create(**params)
        except (openai.APIStatusError, openai.APIConnectionError, openai.APITimeoutError) as exc:
            status = _value(exc, "status_code")
            detail = _value(exc, "message", str(exc))
            prefix = f"{status} " if status is not None else ""
            raise LLMError(f"{tag or model}: OpenAI API error {prefix}{detail}") from exc
        except openai.APIError as exc:
            raise LLMError(f"{tag or model}: OpenAI API error {exc}") from exc

        usage = self._usage(response, model, started)
        async with self._lock:
            self._total = self._total + usage

        status = _value(response, "status")
        if status == "incomplete":
            details = _value(response, "incomplete_details")
            reason = _value(details, "reason", "unknown")
            raise LLMError(
                f"{tag or model}: incomplete response ({reason}) after ${usage.cost_usd:.3f}"
            )
        if status == "failed":
            error = _value(response, "error")
            message = _value(error, "message", "unknown error")
            raise LLMError(
                f"{tag or model}: failed response ({message}) after ${usage.cost_usd:.3f}"
            )

        refusal = self._refusal(response)
        if refusal is not None:
            raise LLMError(f"{tag or model}: refused ({refusal}) after ${usage.cost_usd:.3f}")

        text = self._text(response)
        if not isinstance(text, str) or not text.strip():
            raise LLMError(f"{tag or model}: no text in response after ${usage.cost_usd:.3f}")
        try:
            value = output_format.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMError(
                f"{tag or model}: unparseable output after ${usage.cost_usd:.3f} "
                f"(status {status or 'unknown'}, {usage.output_tokens} of {max_tokens} "
                f"output tokens): {exc}"
            ) from exc
        return Parsed(value=value, usage=usage, model=model)

    @staticmethod
    def _refusal(response: Any) -> str | None:
        output = _value(response, "output", []) or []
        for item in output:
            for content in _value(item, "content", []) or []:
                if _value(content, "type") == "refusal":
                    return str(_value(content, "refusal", "model refused"))
        return None

    @staticmethod
    def _text(response: Any) -> str | None:
        """Read the SDK convenience field, with a fallback for raw responses."""
        text = _value(response, "output_text")
        if isinstance(text, str):
            return text
        chunks: list[str] = []
        for item in _value(response, "output", []) or []:
            for content in _value(item, "content", []) or []:
                if _value(content, "type") != "output_text":
                    continue
                chunk = _value(content, "text")
                if isinstance(chunk, str):
                    chunks.append(chunk)
        return "".join(chunks) or None

    @staticmethod
    def _usage(response: Any, model: str, started: float) -> Usage:
        raw = _value(response, "usage")
        input_tokens = int(_value(raw, "input_tokens", 0) or 0)
        output_tokens = int(_value(raw, "output_tokens", 0) or 0)
        details = _value(raw, "input_tokens_details")
        cached = int(_value(details, "cached_tokens", 0) or 0)
        # Responses reports input_tokens including cached input; Usage keeps
        # its input field as the uncached portion, matching Anthropic usage.
        usage = Usage(
            input_tokens=max(0, input_tokens - cached),
            output_tokens=output_tokens,
            cache_read_tokens=cached,
            latency_s=time.monotonic() - started,
        )
        return replace(usage, cost_usd=cost_usd(model, usage))
