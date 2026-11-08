from __future__ import annotations

import asyncio
import json
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
from commentary.replay.identity import IdentityAction, IdentityLookup, IdentityProposal
from commentary.replay.preparation import (
    FrameInspection,
    ReplayObservation,
    ReplayObserver,
    ReplayObserverFinal,
    ReplayObserverResult,
    inspect_frame,
    inspect_frames,
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


async def test_observer_requires_agent_identity_proposal_after_visual_reading() -> None:
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
    backend.queue(
        "replay_observer_identity",
        [
            IdentityLookup(kind="research_roster"),
            IdentityAction(
                kind="identify",
                identities=[
                    IdentityProposal(
                        observation_id="v1-o1",
                        squad_id="supplied-home",
                        player_name="Ada Green",
                    )
                ],
            ),
        ],
    )
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


async def test_motion_inspection_compares_only_selected_current_frames() -> None:
    frames = []
    for at_s, color in [(1, (255, 0, 0)), (2, (0, 255, 0)), (3, (0, 0, 255))]:
        raw = np.full((50, 100, 3), color, dtype=np.uint8)
        ok, jpeg = cv2.imencode(".jpg", raw)
        assert ok
        frames.append(Frame(at_s, jpeg.tobytes()))
    request = FrameInspection(
        frame_at_s=2,
        context_at_s=[1, 3],
        left=0.5,
        top=0,
        right=1,
        bottom=1,
    )
    crops = inspect_frames(frames, request)
    assert [frame.at_s for frame in crops] == [1, 2, 3]
    for index, crop in enumerate(crops):
        decoded = cv2.imdecode(np.frombuffer(crop.jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert float(decoded[:, :, index].mean()) > 250
    backend = ScriptedBackend()
    backend.always("replay_observer", ReplayObserverResult(inspection=request))
    backend.always(
        "replay_observer_inspection",
        ReplayObserverFinal(
            observations=[
                ReplayObservation(
                    id="save",
                    at_s=2,
                    kind="play",
                    description="The keeper deflects the shot.",
                    action=FootballAction(
                        type="save",
                        certainty="confirmed",
                        cues=["keeper_contact"],
                        support_at_s=[1, 2, 3],
                    ),
                ),
            ]
        ),
    )
    observer = ReplayObserver(backend, model="offline")
    result = await observer(frames, 3)
    assert result[0].action is not None and result[0].action.type == "save"
    assert backend.calls[1].images == 6  # three originals and three selected crops
    assert observer.audit[0]["inspection_frames_at_s"] == [1, 2, 3]
    with pytest.raises(ValueError, match="supplied"):
        inspect_frames(frames, request.model_copy(update={"context_at_s": [1, 4]}))


async def test_inspection_first_plans_a_bounded_region_before_classification() -> None:
    backend = ScriptedBackend()
    ok, jpeg = cv2.imencode(".jpg", np.zeros((60, 80, 3), dtype=np.uint8))
    assert ok
    frames = [Frame(1, jpeg.tobytes()), Frame(2, jpeg.tobytes())]
    backend.always(
        "replay_observer_inspection_plan",
        FrameInspection(
            frame_at_s=2,
            context_at_s=[1],
            left=0,
            top=0,
            right=1,
            bottom=1,
        ),
    )
    backend.always(
        "replay_observer_inspection",
        ReplayObserverFinal(
            observations=[
                ReplayObservation(
                    id="card",
                    at_s=2,
                    kind="play",
                    description="A red card is raised.",
                    action=FootballAction(
                        type="red_card",
                        certainty="confirmed",
                        cues=["red_card_visible"],
                        support_at_s=[2],
                    ),
                ),
            ]
        ),
    )
    observer = ReplayObserver(backend, model="offline", inspection_first=True)
    result = await observer(frames, 2)
    assert result[0].action is not None and result[0].action.type == "red_card"
    assert [call.tag for call in backend.calls] == [
        "replay_observer_inspection_plan",
        "replay_observer_inspection",
    ]
    assert observer.audit[0]["inspection_mode"] == "first"
    assert observer.audit[0]["inspection_frames_at_s"] == [1, 2]
    assert len(observer.audit[0]["prompt_sha256"]) == 64


async def test_inspection_first_rejects_unsupplied_planner_frames_before_classification() -> None:
    backend = ScriptedBackend()
    backend.always(
        "replay_observer_inspection_plan",
        FrameInspection(
            frame_at_s=2,
            context_at_s=[3],
            left=0,
            top=0,
            right=1,
            bottom=1,
        ),
    )
    observer = ReplayObserver(backend, model="offline", inspection_first=True)
    with pytest.raises(ValueError, match="supplied"):
        await observer([Frame(2, b"unused")], 2)
    assert [call.tag for call in backend.calls] == ["replay_observer_inspection_plan"]
    assert observer.audit[0]["status"] == "failed"


async def test_fractional_frame_labels_round_trip_without_weakening_membership() -> None:
    backend = ScriptedBackend()
    timestamp = 1 / 3

    def answer(blocks: Any, _format: Any) -> ReplayObserverResult:
        payload = json.loads(blocks[0]["text"])
        assert payload["available_frame_timestamps"] == [timestamp]
        label = blocks[1]["text"].split("at_s=")[1]
        assert float(label) == timestamp
        return ReplayObserverResult(observations=[])

    backend.register("replay_observer", answer)
    await ReplayObserver(backend, model="offline")([Frame(timestamp, b"unused")], timestamp)


@pytest.mark.parametrize(
    "roster,expected",
    [
        ([{"name": "Ada Green", "number": 7}], "Ada Green"),
        ([{"name": "Ada Green", "number": 8}], None),
        ([{"name": "Ada Green", "number": 7}, {"name": "Bea Green", "number": 7}], None),
    ],
)
async def test_identity_inspection_reads_before_python_resolves_roster(roster, expected) -> None:
    backend = ScriptedBackend()
    ok, jpeg = cv2.imencode(".jpg", np.zeros((60, 80, 3), dtype=np.uint8))
    assert ok
    frames = [Frame(1, jpeg.tobytes())]
    pack = {"home": {"name": "Green", "kit": "green", "starters": roster}}

    def plan(blocks: list[Any], _schema: Any) -> FrameInspection:
        payload = json.loads(blocks[0]["text"])
        assert payload["pack"] == {
            "teams": {
                "home": {
                    "name": "Green",
                    "kit_colours": "green",
                }
            }
        }
        assert "Ada Green" not in json.dumps(blocks)
        return FrameInspection(frame_at_s=1, left=0, top=0, right=1, bottom=1)

    def read(blocks: list[Any], _schema: Any) -> ReplayObserverFinal:
        assert "Ada Green" not in json.dumps(blocks)
        return ReplayObserverFinal(
            observations=[
                observation(
                    id="pass",
                    at_s=1,
                    kind="play",
                    description="Number 7 passes forward.",
                    actor=ObservedActor(
                        side="home",
                        shirt_number=7,
                        identity_source="shirt_number",
                        action_role="passer",
                    ),
                )
            ]
        )

    backend.register("replay_observer_inspection_plan", plan)
    backend.register("replay_observer_inspection", read)
    backend.queue(
        "replay_observer_identity",
        [
            IdentityLookup(kind="research_roster"),
            IdentityAction(
                kind="identify",
                identities=[
                    IdentityProposal(
                        observation_id="v1-o1",
                        squad_id="supplied-home",
                        player_name="Ada Green",
                    )
                ],
            ),
        ],
    )
    observer = ReplayObserver(backend, model="offline", pack=pack, inspection_focus="identity")
    result = await observer(frames, 1)
    assert (result[0].actor.resolved_name if result[0].actor else None) == expected
    assert observer.audit[0]["inspection_focus"] == "identity"
    assert len(backend.calls) == 4


def test_unknown_identity_inspection_focus_is_rejected() -> None:
    with pytest.raises(ValueError, match="inspection focus"):
        ReplayObserver(ScriptedBackend(), model="offline", inspection_focus="other")


@pytest.mark.parametrize("context,expected", [([2, 1], [1, 2]), ([3, 3], []), ([3, 1], [1])])
def test_crop_context_is_canonical_without_adding_frames(context, expected) -> None:
    request = FrameInspection(frame_at_s=3, context_at_s=context, left=0, top=0, right=1, bottom=1)
    assert request.context_at_s == expected


@pytest.mark.parametrize("context", [[float("nan")], [float("inf")], [-1]])
def test_invalid_crop_times_are_rejected(context) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        FrameInspection(frame_at_s=3, context_at_s=context, left=0, top=0, right=1, bottom=1)


async def test_squad_research_focus_is_bound_to_fixture_and_cutoff() -> None:
    backend = SearchBackend()
    backend.always("replay_researcher", ResearchDraft(facts=[], squads=[]))
    result = await prepare_research(
        backend, fixture="Green v Orange", as_of=date(2022, 12, 9), model="offline", focus="squads"
    )
    payload = json.loads(backend.calls[0].blocks[0]["text"])
    assert payload == {
        "fixture": "Green v Orange",
        "as_of": "2022-12-09",
        "focus": "squads",
        "requested_sides": ["home", "away"],
    }
    assert result.audit["focus"] == "squads"
    assert len(result.audit["prompt_sha256"]) == 64
    assert result.brief.squads == []
    assert backend.extra_params == {"sentinel": True}


async def test_invalid_research_focus_does_not_call_model() -> None:
    backend = SearchBackend()
    with pytest.raises(ValueError, match="research focus"):
        await prepare_research(
            backend,
            fixture="Green v Orange",
            as_of=date(2022, 12, 9),
            model="offline",
            focus="other",
        )
    assert not backend.calls


async def test_research_rejects_an_unrequested_team() -> None:
    from commentary.recorded_research import ResearchSquad, SquadPlayer

    backend = SearchBackend()
    backend.always(
        "replay_researcher",
        ResearchDraft(
            facts=[],
            squads=[
                ResearchSquad(
                    id="home-squad",
                    side="home",
                    team="Green",
                    players=[SquadPlayer(name="Player", number=9)],
                    source_url="https://example.org/squad",
                    source_date=date(2022, 11, 14),
                )
            ],
        ),
    )
    with pytest.raises(ValueError, match="unrequested squad side"):
        await prepare_research(
            backend,
            fixture="Green v Orange",
            as_of=date(2022, 12, 9),
            model="offline",
            focus="squads",
            squad_sides=("away",),
        )
    assert backend.extra_params == {"sentinel": True}


@pytest.mark.parametrize("sides", [(), ("home", "home"), ("other",)])
async def test_invalid_requested_sides_do_not_call_model(sides) -> None:
    backend = SearchBackend()
    with pytest.raises(ValueError, match="research squad sides"):
        await prepare_research(
            backend,
            fixture="Green v Orange",
            as_of=date(2022, 12, 9),
            model="offline",
            focus="squads",
            squad_sides=sides,
        )
    assert not backend.calls


async def test_observer_history_is_past_context_without_actor_identity() -> None:
    backend = ScriptedBackend()

    def answer(blocks, schema):
        payload = json.loads(blocks[0]["text"])
        at = payload["available_frame_timestamps"][-1]
        return ReplayObserverResult(
            observations=[
                observation(
                    id="event",
                    at_s=at,
                    kind="play",
                    description="The ball moves forward.",
                )
            ]
        )

    backend.register("replay_observer", answer)
    observer = ReplayObserver(backend, model="offline")
    await observer([Frame(1, b"frame")], 1)
    await observer([Frame(3, b"frame")], 3)
    context = json.loads(backend.calls[1].blocks[0]["text"])["earlier_observations"]
    assert [item["at_s"] for item in context] == [1]
    assert "actor" not in context[0]
    await observer([Frame(0.5, b"frame")], 0.5)
    assert json.loads(backend.calls[2].blocks[0]["text"])["earlier_observations"] == []
