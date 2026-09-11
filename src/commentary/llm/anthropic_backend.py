"""The real backend: Anthropic's Messages API with structured outputs.

One code path for every agent — ``messages.create`` with a JSON-schema output
format, parsed and validated on the way out. Thinking configuration differs
per model family and is derived here rather than passed around.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from commentary.llm.base import Block, LLMError, Parsed, Usage, text_block
from commentary.llm.pricing import cost_usd
from commentary.llm.schema import strict_schema

T = TypeVar("T", bound=BaseModel)

#: Models that take no ``thinking`` and reject ``effort``.
_NO_THINKING = {"claude-haiku-4-5"}


def thinking_params(model: str, effort: str | None) -> dict[str, Any]:
    """Per-model thinking and effort settings.

    Haiku 4.5 predates adaptive thinking and errors on ``effort``. Sonnet 5 and
    Opus 5 take adaptive thinking; both accept ``effort``. Nothing here asks
    for a thinking budget — ``budget_tokens`` is a 400 on the current models.
    """
    if model in _NO_THINKING:
        return {}
    params: dict[str, Any] = {"thinking": {"type": "adaptive"}}
    if effort is not None:
        params["output_config"] = {"effort": effort}
    return params


class AnthropicBackend:
    """Structured vision calls against the Anthropic API.

    Latency matters more than completeness here: a caller line that arrives
    after the moment has passed is worse than no line, so calls carry a short
    timeout and a single retry, and a timeout surfaces as :class:`LLMError`
    for the director to shrug off.
    """

    def __init__(
        self,
        client: anthropic.AsyncAnthropic | None = None,
        *,
        timeout_s: float = 8.0,
        max_retries: int = 1,
    ) -> None:
        self._client = client or anthropic.AsyncAnthropic(
            timeout=timeout_s, max_retries=max_retries
        )
        self._total = Usage()
        self._lock = asyncio.Lock()

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
        started = time.monotonic()
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": [text_block(system, cache=cache_system)],
            "messages": [{"role": "user", "content": blocks}],
            **thinking_params(model, effort),
        }
        output_config = dict(params.get("output_config", {}))
        output_config["format"] = {
            "type": "json_schema",
            "schema": strict_schema(output_format),
        }
        params["output_config"] = output_config

        try:
            response = await self._client.messages.create(**params)
        except anthropic.APIStatusError as exc:
            raise LLMError(f"{tag or model}: {exc.status_code} {exc.message}") from exc
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
            raise LLMError(f"{tag or model}: {type(exc).__name__}") from exc

        if response.stop_reason == "refusal":
            raise LLMError(f"{tag or model}: refused")

        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMError(f"{tag or model}: no text block in response")
        try:
            value = output_format.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMError(f"{tag or model}: unparseable output: {exc}") from exc

        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
            latency_s=time.monotonic() - started,
        )
        usage = replace(usage, cost_usd=cost_usd(model, usage))
        async with self._lock:
            self._total = self._total + usage
        return Parsed(value=value, usage=usage, model=model)
