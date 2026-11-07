from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pytest

from commentary.llm.base import LLMError
from commentary.llm.fake import ScriptedBackend
from commentary.recorded_demo import Frame, ObservedActor
from commentary.recorded_research import ResearchBrief, ResearchDraft
from commentary.replay.preparation import ReplayObserver, prepare_research, team_names
from commentary.streaming_demo import StreamingObservation, StreamingObserverResult


@dataclass
class SearchBackend(ScriptedBackend):
    extra_params: dict[str, Any] = field(default_factory=lambda: {"sentinel": True})


async def test_research_is_bounded_scoped_and_not_available_without_search() -> None:
    backend = SearchBackend()

    def answer(_blocks: list[Any], _schema: Any) -> ResearchDraft:
        assert backend.extra_params["tools"][0]["max_uses"] == 3
        return ResearchDraft(facts=[])

    backend.register("replay_researcher", answer)
    result = await prepare_research(
        backend, fixture="Green v Orange", as_of=date(2026, 1, 1), model="test"
    )
    assert result.brief.fixture == "Green v Orange"
    assert result.audit["model_called"]
    assert backend.extra_params == {"sentinel": True}
    with pytest.raises(LLMError, match="search-capable"):
        await prepare_research(
            ScriptedBackend(), fixture="Green v Orange", as_of=date(2026, 1, 1), model="test"
        )


async def test_search_permissions_are_restored_after_cancellation() -> None:
    backend = SearchBackend(latency_s=1)
    backend.always("replay_researcher", ResearchDraft(facts=[]))
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await prepare_research(
                backend, fixture="Green v Orange", as_of=date(2026, 1, 1), model="test"
            )
    assert backend.extra_params == {"sentinel": True}


async def test_saved_research_is_fixture_and_cutoff_bound() -> None:
    brief = ResearchBrief(fixture="Green v Orange", as_of=date(2026, 1, 2), facts=[])
    backend = ScriptedBackend()
    for fixture, cutoff in [
        ("Other fixture", date(2026, 1, 3)),
        ("Green v Orange", date(2026, 1, 1)),
    ]:
        with pytest.raises(ValueError, match="fixture and pre-match"):
            await prepare_research(
                backend, fixture=fixture, as_of=cutoff, model="test", saved=brief
            )
    result = await prepare_research(
        backend, fixture=brief.fixture, as_of=brief.as_of, model="test", saved=brief
    )
    assert result.brief == brief
    assert not backend.calls


async def test_observer_overwrites_model_guessed_identity_using_readable_evidence() -> None:
    backend = ScriptedBackend()
    backend.always(
        "replay_observer",
        StreamingObserverResult(
            observations=[
                StreamingObservation(
                    id="x",
                    at_s=1,
                    kind="play",
                    description="A pass is played forward.",
                    actor=ObservedActor(
                        side="home",
                        shirt_number=7,
                        identity_source="shirt_number",
                        action_role="passer",
                        resolved_name="Invented Name",
                    ),
                )
            ]
        ),
    )
    pack = {"home": {"name": "Green", "starters": [{"name": "Ada Green", "number": 7}]}}
    observer = ReplayObserver(backend, model="test", pack=pack)
    result = await observer([Frame(at_s=1, jpeg=b"test")], 1)
    assert result[0].actor is not None
    assert result[0].actor.resolved_name == "Ada Green"
    assert result[0].id == "v1-o1"
    assert team_names(pack) == {"home": "Green"}
    assert observer.audit[0]["frames_at_s"] == [1]


async def test_observer_rejects_future_output_and_never_sends_future_frames() -> None:
    backend = ScriptedBackend()
    observer = ReplayObserver(backend, model="test")
    with pytest.raises(ValueError, match="available"):
        await observer([Frame(at_s=2, jpeg=b"test")], 1)
    assert not backend.calls
    backend.always(
        "replay_observer",
        StreamingObserverResult(
            observations=[
                StreamingObservation(id="x", at_s=2, kind="goal", description="Ball is in the net.")
            ]
        ),
    )
    with pytest.raises(ValueError, match="inside its frame batch"):
        await observer([Frame(at_s=1, jpeg=b"test")], 1)
    assert not observer.audit
