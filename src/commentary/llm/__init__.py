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
    "default_backend",
    "encode_frame",
    "grading_backend",
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


def grading_backend(
    timeout_s: float = 600.0,
    *,
    max_retries: int = 0,
    no_key_hint: str = "or run against the simulator with --backend oracle",
) -> LLMBackend:
    """A backend for grading, which is the opposite of the one for a match.

    :func:`default_backend` gives every agent the runtime's client: an
    eight-second timeout and one retry, because a caller line that arrives
    after the moment has passed is worse than no line. Both settings are
    wrong for grading. The ``register`` command's judge call was the first
    real caller of a long-running grading model and it timed out on
    :func:`default_backend`'s eight seconds before this existed; a judge call
    reads a whole passage — or, for the factuality and pairwise judges in
    :mod:`commentary.grading.judge`, one line with its surrounding feed — at
    high effort, and takes minutes, not seconds.

    ``max_retries=0`` matters just as much and for a different reason. A
    request that times out on the client has very likely been billed on the
    server, and a retry doubles that spend for an answer nobody ever sees.
    One invocation of a grading command is at most one billed call per line.

    ``no_key_hint`` lets a caller point at whatever escape hatch it offers
    instead of a key — ``register`` has ``--no-model``, ``grade`` has none —
    without duplicating the credential check.
    """
    import os

    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        raise LLMError(f"no Anthropic credentials: set ANTHROPIC_API_KEY in .env, {no_key_hint}")
    from commentary.llm.anthropic_backend import AnthropicBackend

    return AnthropicBackend(timeout_s=timeout_s, max_retries=max_retries)
