"""The contract every agent uses to reach a model.

One method, ``parse``: content blocks in, a validated Pydantic object out.
Research, observations and scripts all use structured output. A backend that
cannot produce the requested type raises rather than guessing.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

import numpy as np
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

#: An Anthropic content block: ``{"type": "text", ...}`` or ``{"type": "image", ...}``.
Block = dict[str, Any]


class LLMError(RuntimeError):
    """A model call failed in a way the caller has to decide about."""


@dataclass(frozen=True)
class Usage:
    """What one call cost, in tokens and dollars."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            latency_s=max(self.latency_s, other.latency_s),
        )


@dataclass(frozen=True)
class Parsed[U: BaseModel]:
    """A validated model response plus what it cost to get it."""

    value: U
    usage: Usage = field(default_factory=Usage)
    model: str = ""


class LLMBackend(Protocol):
    """Anything that can turn content blocks into a validated object."""

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
        """Run one call and return ``output_format`` filled in.

        ``tag`` names the call site (research, observation or writing) for
        tracing, cost attribution, and scripted-backend dispatch.
        """
        ...

    @property
    def total(self) -> Usage:
        """Everything this backend has spent so far."""
        ...


def text_block(text: str, *, cache: bool = False) -> Block:
    block: Block = {"type": "text", "text": text}
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    return block


def image_block(jpeg: bytes) -> Block:
    """A JPEG as an Anthropic-compatible image block."""
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.standard_b64encode(jpeg).decode("ascii"),
        },
    }


def encode_frame(image: np.ndarray, *, quality: int = 70, max_width: int = 768) -> bytes:
    """BGR array to JPEG bytes, preserving aspect ratio within max_width."""
    import cv2

    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        image = cv2.resize(image, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise LLMError("JPEG encode failed")
    return bytes(buf)
