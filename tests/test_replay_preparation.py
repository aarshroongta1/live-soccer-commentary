from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import cv2
import numpy as np
import pytest

from commentary.football import FootballAction
from commentary.llm.base import LLMError
from commentary.llm.fake import ScriptedBackend
from commentary.recorded_demo import Frame, ObservedActor
from commentary.recorded_research import ResearchBrief, ResearchDraft
from commentary.replay.preparation import (
    FrameInspection,
    ReplayObservation,
    ReplayObserver,
    ReplayObserverFinal,
    ReplayObserverResult,
    inspect_frame,
    prepare_research,
    team_names,
)


def observation(**data: Any) -> ReplayObservation:
    is_goal = data["kind"] == "goal"
    return ReplayObservation(
        **data,
        action=FootballAction(
            type="goal" if is_goal else "pass",
            certainty="confirmed",
            cues=["ball_in_net" if is_goal else "ball_motion"],
            support_at_s=[data["at_s"]],
        ),
    )


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
        ReplayObserverResult(
            observations=[
                observation(
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
        ReplayObserverResult(
            observations=[
                observation(id="x", at_s=2, kind="goal", description="Ball is in the net.")
            ]
        ),
    )
    with pytest.raises(ValueError, match="inside its frame batch"):
        await observer([Frame(at_s=1, jpeg=b"test")], 1)
    assert observer.audit[0]["status"] == "failed"
    assert observer.audit[0]["resolved"] == []


async def test_observer_can_select_one_existing_frame_for_bounded_inspection() -> None:
    backend = ScriptedBackend()
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:, 100:] = (0, 255, 0)
    ok, jpeg = cv2.imencode(".jpg", image)
    assert ok
    frames = [Frame(at_s=1, jpeg=jpeg.tobytes())]
    backend.always(
        "replay_observer",
        ReplayObserverResult(
            inspection=FrameInspection(
                frame_at_s=1,
                left=0.5,
                top=0,
                right=1,
                bottom=1,
            )
        ),
    )
    backend.always(
        "replay_observer_inspection",
        ReplayObserverFinal(
            observations=[
                observation(id="g", at_s=1, kind="goal", description="Ball is in the net.")
            ]
        ),
    )
    observer = ReplayObserver(backend, model="test")
    result = await observer(frames, 1)
    assert result[0].kind == "goal"
    assert [call.tag for call in backend.calls] == ["replay_observer", "replay_observer_inspection"]
    assert backend.calls[1].images == 2
    assert observer.audit[0]["inspection"]["frame_at_s"] == 1
    cropped = cv2.imdecode(
        np.frombuffer(
            inspect_frame(
                frames,
                FrameInspection(
                    frame_at_s=1,
                    left=0.5,
                    top=0,
                    right=1,
                    bottom=1,
                ),
            ),
            np.uint8,
        ),
        cv2.IMREAD_COLOR,
    )
    assert cropped.shape[0] <= 720 and cropped.shape[1] <= 1280
    assert float(cropped[:, :, 1].mean()) > 250


async def test_inspection_cannot_fetch_a_future_or_unsupplied_frame() -> None:
    backend = ScriptedBackend()
    backend.always(
        "replay_observer",
        ReplayObserverResult(
            inspection=FrameInspection(
                frame_at_s=2,
                left=0,
                top=0,
                right=1,
                bottom=1,
            )
        ),
    )
    with pytest.raises(ValueError, match="exact supplied"):
        await ReplayObserver(backend, model="test")([Frame(at_s=1, jpeg=b"test")], 1)
    assert [call.tag for call in backend.calls] == ["replay_observer"]


async def test_inspected_result_still_cannot_invent_future_evidence() -> None:
    backend = ScriptedBackend()
    ok, jpeg = cv2.imencode(".jpg", np.zeros((20, 20, 3), dtype=np.uint8))
    assert ok
    backend.always(
        "replay_observer",
        ReplayObserverResult(
            inspection=FrameInspection(
                frame_at_s=1,
                left=0,
                top=0,
                right=1,
                bottom=1,
            )
        ),
    )
    backend.always(
        "replay_observer_inspection",
        ReplayObserverFinal(
            observations=[
                observation(id="future", at_s=2, kind="goal", description="Not yet visible.")
            ]
        ),
    )
    with pytest.raises(ValueError, match="inside its frame batch"):
        await ReplayObserver(backend, model="test")([Frame(at_s=1, jpeg=jpeg.tobytes())], 1)
    assert len(backend.calls) == 2


async def test_failed_inspection_keeps_the_selected_tool_in_the_audit() -> None:
    backend = ScriptedBackend()
    ok, jpeg = cv2.imencode(".jpg", np.zeros((20, 20, 3), dtype=np.uint8))
    assert ok
    backend.always(
        "replay_observer",
        ReplayObserverResult(
            inspection=FrameInspection(
                frame_at_s=1,
                left=0,
                top=0,
                right=1,
                bottom=1,
            )
        ),
    )
    observer = ReplayObserver(backend, model="test")
    with pytest.raises(LLMError, match="no scripted handler"):
        await observer([Frame(at_s=1, jpeg=jpeg.tobytes())], 1)
    assert observer.audit[0]["status"] == "failed"
    assert observer.audit[0]["inspection_status"] == "executed"
    assert observer.audit[0]["inspection"]["frame_at_s"] == 1
    assert observer.audit[0]["resolved"] == []


async def test_unsupported_path_is_downgraded_and_audited_without_losing_goal() -> None:
    backend = ScriptedBackend()
    event = ReplayObservation(
        id="g",
        at_s=1,
        kind="goal",
        description="The ball is in the net.",
        action=FootballAction(
            type="goal",
            certainty="confirmed",
            ball_path="ground",
            cues=["ball_in_net"],
            support_at_s=[1],
        ),
    )
    backend.always("replay_observer", ReplayObserverResult(observations=[event]))
    observer = ReplayObserver(backend, model="offline")
    result = await observer([Frame(at_s=1, jpeg=b"test")], 1)
    assert result[0].kind == "goal"
    assert result[0].action is not None and result[0].action.ball_path == "unknown"
    assert observer.audit[0]["ball_path_downgrades"] == [
        {
            "id": "g",
            "reported": "ground",
            "accepted": "unknown",
            "reason": "missing_corresponding_visual_cue",
        }
    ]
