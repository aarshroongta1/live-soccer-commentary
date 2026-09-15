"""The two backends the model layer hands out: a live match, and a grading run.

``default_backend`` is sized for a caller line that must land before the
moment passes. ``grading_backend`` is the opposite of it — a judge call reads
a whole passage at high effort and takes minutes, not seconds — and the
``register`` command's first real call timed out on the wrong one before this
existed (docs/research/real-commentary-corpus.md, Gap 8's sibling fix). These
pin the two apart so the mistake cannot come back silently.
"""

from __future__ import annotations

import pytest

from commentary.llm import default_backend, grading_backend
from commentary.llm.anthropic_backend import AnthropicBackend
from commentary.llm.base import LLMError


def test_grading_backend_gets_minutes_not_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    backend = grading_backend()
    assert isinstance(backend, AnthropicBackend)
    assert backend._client.timeout == pytest.approx(600.0)
    assert backend._client.max_retries == 0


def test_grading_backend_takes_its_own_timeout_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    backend = grading_backend(120.0, max_retries=2)
    assert backend._client.timeout == pytest.approx(120.0)
    assert backend._client.max_retries == 2


def test_grading_backend_is_slower_than_the_live_match_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bug this guards: judge.py reaching for the eight-second client."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    live = default_backend()
    grading = grading_backend()
    assert grading._client.timeout > live._client.timeout


def test_grading_backend_refuses_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(LLMError):
        grading_backend()


def test_grading_backend_carries_the_callers_own_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(LLMError, match="or use --no-model"):
        grading_backend(no_key_hint="or use --no-model")
