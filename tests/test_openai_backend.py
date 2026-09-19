"""Offline contract tests for the OpenAI Responses adapter."""

from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from commentary.llm import openai_factory
from commentary.llm.base import LLMError, image_block, text_block
from commentary.llm.openai_backend import OpenAIBackend, responses_blocks
from commentary.recorded_research import ResearchDraft, research_brief


class BoardRead(BaseModel):
    bug_visible: bool
    home_score: int | None = None
    away_score: int | None = None
    clock: str | None = None
    confidence: float


class _Responses:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.params: dict[str, Any] | None = None

    async def create(self, **params: Any) -> Any:
        self.params = params
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


class _Client:
    def __init__(self, response: Any) -> None:
        self.responses = _Responses(response)


def _usage(input_tokens: int = 100, output_tokens: int = 20, cached: int = 0) -> Any:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_tokens_details=SimpleNamespace(cached_tokens=cached),
    )


def _response(text: str, **kwargs: Any) -> Any:
    fields = {"status": "completed", "output_text": text, "output": [], "usage": _usage()}
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def test_anthropic_blocks_become_responses_text_and_data_url_images() -> None:
    converted = responses_blocks([text_block("hello"), image_block(b"jpeg")])
    assert converted == [
        {"type": "input_text", "text": "hello"},
        {"type": "input_image", "image_url": "data:image/jpeg;base64,anBlZw=="},
    ]


def test_success_sends_strict_schema_and_records_usage() -> None:
    client = _Client(
        _response(
            '{"bug_visible": true, "home_score": 1, "away_score": 2, '
            '"clock": "3:00", "confidence": 0.9}'
        )
    )
    backend = OpenAIBackend(client)  # type: ignore[arg-type]

    parsed = asyncio.run(
        backend.parse(
            model="gpt-5.6-luna",
            system="read the board",
            blocks=[text_block("look")],
            output_format=BoardRead,
            max_tokens=99,
            effort="low",
            tag="board",
        )
    )

    assert parsed.value.home_score == 1
    assert parsed.usage.input_tokens == 100
    assert parsed.usage.cost_usd == pytest.approx(0.000044)
    assert backend.total == parsed.usage
    assert client.responses.params is not None
    assert client.responses.params["instructions"] == "read the board"
    assert client.responses.params["max_output_tokens"] == 99
    assert client.responses.params["reasoning"] == {"effort": "low"}
    assert client.responses.params["text"]["format"]["type"] == "json_schema"
    assert client.responses.params["text"]["format"]["strict"] is True


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            _response(
                "",
                output=[
                    SimpleNamespace(
                        content=[SimpleNamespace(type="refusal", refusal="not allowed")]
                    )
                ],
            ),
            "refused",
        ),
        (
            _response(
                "",
                status="incomplete",
                incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            ),
            "incomplete",
        ),
        (_response("not json"), "unparseable"),
    ],
)
def test_failures_still_count_the_response(response: Any, message: str) -> None:
    backend = OpenAIBackend(_Client(response))  # type: ignore[arg-type]
    with pytest.raises(LLMError, match=message):
        asyncio.run(
            backend.parse(model="gpt-5.6-luna", system="", blocks=[], output_format=BoardRead)
        )
    assert backend.total.output_tokens == 20


def test_cached_input_uses_the_existing_discount() -> None:
    response = _response("{}")
    response.output_text = (
        '{"bug_visible": false, "home_score": null, "away_score": null, '
        '"clock": null, "confidence": 0.2}'
    )
    response.usage = _usage(input_tokens=100, output_tokens=0, cached=40)
    backend = OpenAIBackend(_Client(response))  # type: ignore[arg-type]
    parsed = asyncio.run(
        backend.parse(model="gpt-5.6-luna", system="", blocks=[], output_format=BoardRead)
    )
    assert parsed.usage.input_tokens == 60
    assert parsed.usage.cache_read_tokens == 40
    assert parsed.usage.cost_usd == pytest.approx((60 * 0.2 + 40 * 0.2 * 0.1) / 1_000_000)


def test_research_scopes_required_openai_web_search_to_one_call() -> None:
    client = _Client(_response('{"facts": []}'))
    backend = OpenAIBackend(client)  # type: ignore[arg-type]
    backend.extra_params["metadata"] = {"shared": "value"}

    parsed = asyncio.run(
        research_brief(
            backend,
            fixture="Green v Orange",
            as_of=date(2026, 1, 1),
        )
    )

    assert isinstance(parsed.value, ResearchDraft)
    assert client.responses.params is not None
    assert client.responses.params["tools"] == [{"type": "web_search"}]
    assert client.responses.params["tool_choice"] == "required"
    assert client.responses.params["max_tool_calls"] == 3
    assert client.responses.params["max_output_tokens"] == 4096
    assert backend.extra_params == {"metadata": {"shared": "value"}}


def test_openai_factory_requires_its_own_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        openai_factory()


def test_openai_factory_reads_structural_validation_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_TIMEOUT_S", "15")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.example/v1")

    configured = openai_factory()
    explicit = openai_factory(6.0, max_retries=0)

    assert isinstance(configured, OpenAIBackend)
    assert configured._client.timeout == pytest.approx(15.0)
    assert explicit._client.timeout == pytest.approx(6.0)
    assert str(explicit._client.base_url).startswith("https://openai.example/v1")
    assert explicit._client.max_retries == 0
