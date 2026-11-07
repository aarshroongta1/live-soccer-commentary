from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from commentary.llm.base import LLMError, image_block, text_block
from commentary.llm.budget import BudgetExceeded, RunBudget
from commentary.llm.openai_backend import OpenAIBackend


class Reply(BaseModel):
    message: str


class Responses:
    def __init__(self, *, blocked: bool = False, text: str = '{"message":"Ready"}') -> None:
        self.input_tokens = SimpleNamespace(count=self.count)
        self.count_params: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.blocked = blocked
        self.text = text

    async def count(self, **params: Any) -> Any:
        self.count_params.append(params)
        return SimpleNamespace(input_tokens=1000)

    async def create(self, **params: Any) -> Any:
        self.requests.append(params)
        self.started.set()
        if self.blocked:
            await self.release.wait()
        return SimpleNamespace(
            status="completed",
            output_text=self.text,
            usage=SimpleNamespace(
                input_tokens=1000,
                output_tokens=20,
                input_tokens_details=SimpleNamespace(cached_tokens=0),
            ),
        )


def backend(responses: Responses, budget: RunBudget) -> OpenAIBackend:
    return OpenAIBackend(SimpleNamespace(responses=responses), budget=budget, default_effort="none")  # type: ignore[arg-type]


async def call(model: OpenAIBackend, *, name: str = "gpt-6-luna") -> Any:
    return await model.parse(
        model=name,
        system="Answer briefly.",
        blocks=[text_block("Look"), image_block(b"jpeg")],
        output_format=Reply,
        max_tokens=100,
    )


async def test_counts_images_and_schema_and_reports_actual_token_spend() -> None:
    responses = Responses()
    budget = RunBudget(max_usd=0.001, max_calls=2)
    parsed = await call(backend(responses, budget))
    assert parsed.value.message == "Ready"
    assert responses.count_params[0]["input"] == responses.requests[0]["input"]
    assert responses.count_params[0]["text"] == responses.requests[0]["text"]
    assert responses.requests[0]["reasoning"] == {"effort": "none"}
    assert responses.requests[0]["service_tier"] == "default"
    assert budget.report()["reported_usd"] == pytest.approx(0.00011)
    assert budget.report()["unsettled_requests"] == 0


async def test_concurrent_calls_cannot_double_spend_reserved_dollars() -> None:
    responses = Responses(blocked=True)
    budget = RunBudget(max_usd=0.0002, max_calls=5)
    model = backend(responses, budget)
    first = asyncio.create_task(call(model))
    await responses.started.wait()
    with pytest.raises(BudgetExceeded, match="remaining"):
        await call(model)
    assert len(responses.requests) == 1
    responses.release.set()
    await first


async def test_cancelled_generation_keeps_its_reservation() -> None:
    responses = Responses(blocked=True)
    budget = RunBudget(max_usd=0.0002, max_calls=2)
    model = backend(responses, budget)
    task = asyncio.create_task(call(model))
    await responses.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert budget.report()["unsettled_requests"] == 1
    with pytest.raises(BudgetExceeded):
        await call(model)
    assert len(responses.requests) == 1


async def test_call_limit_includes_failed_preflights() -> None:
    responses = Responses()
    budget = RunBudget(max_usd=1, max_calls=1)
    model = backend(responses, budget)
    with pytest.raises(LLMError, match="known token pricing"):
        await call(model, name="unpriced-model")
    with pytest.raises(BudgetExceeded, match="call limit"):
        await call(model)
    assert not responses.requests
    assert not responses.count_params


async def test_hosted_tools_cannot_spend_outside_token_budget() -> None:
    responses = Responses()
    model = backend(responses, RunBudget())
    model.extra_params["tools"] = [{"type": "web_search"}]
    with pytest.raises(LLMError, match="hosted tool fees"):
        await call(model)
    assert not responses.requests


async def test_invalid_output_still_settles_billed_usage() -> None:
    responses = Responses(text="not json")
    budget = RunBudget()
    with pytest.raises(LLMError, match="unparseable"):
        await call(backend(responses, budget))
    assert budget.report()["reported_usd"] == pytest.approx(0.00011)
    assert budget.report()["unsettled_requests"] == 0


async def test_model_override_cannot_hide_an_expensive_model_from_accounting() -> None:
    responses = Responses()
    model = backend(responses, RunBudget())
    model.extra_params["model"] = "gpt-6-astra"
    with pytest.raises(LLMError, match="override the model"):
        await call(model)
    assert not responses.requests
    assert not responses.count_params


async def test_long_context_is_rejected_before_generation() -> None:
    responses = Responses()

    async def large_count(**_params: Any) -> Any:
        return SimpleNamespace(input_tokens=272001)

    responses.input_tokens.count = large_count
    with pytest.raises(LLMError, match="short-context"):
        await call(backend(responses, RunBudget(max_usd=10)))
    assert not responses.requests


async def test_missing_usage_does_not_release_potentially_billed_reservation() -> None:
    responses = Responses()
    original_create = responses.create

    async def no_usage(**params: Any) -> Any:
        response = await original_create(**params)
        response.usage = None
        return response

    responses.create = no_usage  # type: ignore[method-assign]
    budget = RunBudget()
    await call(backend(responses, budget))
    assert budget.report()["unsettled_requests"] == 1
    assert budget.report()["reported_usd"] == 0
