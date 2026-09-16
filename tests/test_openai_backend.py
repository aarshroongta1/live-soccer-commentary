"""Offline contract tests for the OpenAI Responses adapter."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from commentary.__main__ import build_parser
from commentary.llm import default_backend, openai_factory
from commentary.llm.base import LLMError, image_block, text_block
from commentary.llm.openai_backend import OpenAIBackend, responses_blocks
from commentary.schemas import BoardRead


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
            backend.parse(
                model="gpt-5.6-luna", system="", blocks=[], output_format=BoardRead
            )
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


def test_openai_factory_requires_its_own_key_and_cli_names_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        default_backend("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert isinstance(openai_factory(), OpenAIBackend)
    args = build_parser().parse_args(["run", "--backend", "openai"])
    assert args.backend == "openai"
