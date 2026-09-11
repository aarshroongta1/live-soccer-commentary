"""The model layer: one protocol, two backends, and a price list.

Every agent talks to :class:`LLMBackend`. In production that is Anthropic; in
tests and offline runs it is a scripted or oracle backend, so the whole
pipeline runs with no network and no key.
"""

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
    "strict_schema",
    "text_block",
]


def default_backend() -> LLMBackend:
    """The real backend if a key is around, otherwise a loud failure.

    Imported lazily so that tests, the simulator, and CI never need the
    ``anthropic`` client constructed or a key present.
    """
    import os

    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        raise LLMError(
            "no Anthropic credentials: set ANTHROPIC_API_KEY in .env, "
            "or run against the simulator with --backend oracle"
        )
    from commentary.llm.anthropic_backend import AnthropicBackend

    return AnthropicBackend()
