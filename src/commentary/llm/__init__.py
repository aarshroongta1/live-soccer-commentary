"""Shared model protocol, adapters, test backend, and structured-output helpers."""

from commentary.llm.base import (
    Block,
    LLMBackend,
    LLMError,
    Parsed,
    Usage,
    encode_frame,
    image_block,
    text_block,
)
from commentary.llm.fake import ScriptedBackend
from commentary.llm.pricing import cost_usd
from commentary.llm.schema import strict_schema

__all__ = [
    "Block",
    "LLMBackend",
    "LLMError",
    "Parsed",
    "ScriptedBackend",
    "Usage",
    "cost_usd",
    "encode_frame",
    "image_block",
    "openai_factory",
    "strict_schema",
    "text_block",
]


def openai_factory(
    timeout_s: float | None = None,
    *,
    max_retries: int = 1,
) -> LLMBackend:
    """Construct the OpenAI Responses adapter from explicit environment settings."""
    import os

    if not os.getenv("OPENAI_API_KEY"):
        raise LLMError("no OpenAI credentials: set OPENAI_API_KEY in .env")
    from commentary.llm.openai_backend import OpenAIBackend

    return OpenAIBackend(
        timeout_s=(float(os.getenv("OPENAI_TIMEOUT_S", "8.0")) if timeout_s is None else timeout_s),
        max_retries=max_retries,
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
