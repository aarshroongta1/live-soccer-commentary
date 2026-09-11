"""Backends that need no key and no network.

:class:`ScriptedBackend` dispatches on the call-site tag, which is what tests
and the offline demo run on. Every call is recorded, so a test can assert on
what the caller was actually shown, not just on what came back.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel

from commentary.llm.base import Block, LLMError, Parsed, Usage

T = TypeVar("T", bound=BaseModel)

#: Given the blocks a call was made with, produce the object it should return.
Handler = Callable[[list[Block], type[BaseModel]], BaseModel]


@dataclass
class Call:
    """One recorded call, for assertions."""

    tag: str
    model: str
    system: str
    blocks: list[Block]
    output_format: type[BaseModel]

    @property
    def images(self) -> int:
        return sum(1 for b in self.blocks if b.get("type") == "image")

    @property
    def text(self) -> str:
        return "\n".join(b["text"] for b in self.blocks if b.get("type") == "text")


@dataclass
class ScriptedBackend:
    """Answers by tag. Unregistered tags raise rather than invent a reply."""

    handlers: dict[str, Handler] = field(default_factory=dict)
    latency_s: float = 0.0
    calls: list[Call] = field(default_factory=list)
    _total: Usage = field(default_factory=Usage)

    def register(self, tag: str, handler: Handler) -> None:
        self.handlers[tag] = handler

    def always(self, tag: str, value: BaseModel) -> None:
        """Return the same object for every call with this tag."""
        self.handlers[tag] = lambda _blocks, _fmt: value

    def queue(self, tag: str, values: Sequence[BaseModel]) -> None:
        """Return these objects in order, then repeat the last one forever."""
        remaining = list(values)
        if not remaining:
            raise ValueError("queue needs at least one value")

        def handler(_blocks: list[Block], _fmt: type[BaseModel]) -> BaseModel:
            return remaining.pop(0) if len(remaining) > 1 else remaining[0]

        self.handlers[tag] = handler

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
        self.calls.append(
            Call(tag=tag, model=model, system=system, blocks=blocks, output_format=output_format)
        )
        handler = self.handlers.get(tag)
        if handler is None:
            raise LLMError(f"no scripted handler for tag {tag!r}")
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        value = handler(blocks, output_format)
        if not isinstance(value, output_format):
            raise LLMError(f"handler for {tag!r} returned {type(value).__name__}")
        usage = Usage(input_tokens=0, output_tokens=0, latency_s=self.latency_s)
        self._total = self._total + usage
        return Parsed(value=value, usage=usage, model=model)

    def calls_tagged(self, tag: str) -> list[Call]:
        return [c for c in self.calls if c.tag == tag]
