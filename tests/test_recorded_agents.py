"""Recorded preparation must use source evidence, not generation-clock timing."""

import pytest

from commentary.football import FootballAction
from commentary.llm.budget import BudgetExceeded
from commentary.recorded_demo import Observation
from commentary.replay.contracts import SpeechLine, TurnResult
from commentary.replay.recorded import prepare_dialogue, ready_at


def event(identifier, at_s, support):
    return Observation(
        id=identifier,
        at_s=at_s,
        kind="play",
        description="The ball moves forward.",
        action=FootballAction(
            type="pass", certainty="confirmed", cues=["ball_motion"], support_at_s=support
        ),
    )


async def test_recorded_turns_never_receive_later_support_or_evidence() -> None:
    observations = [event("early", 0, [0]), event("later", 2, [2, 5])]
    snapshots = []

    async def run(role, view):
        snapshots.append(view)
        assert all(ready_at(item) <= view.cutoff_s for item in view.observations)
        return TurnResult(
            role=role,
            snapshot_version=view.version,
            cutoff_s=view.cutoff_s,
            line=SpeechLine(
                voice=role, text="The ball moves forward.", evidence_ids=[view.observations[-1].id]
            ),
        )

    result = await prepare_dialogue(run, observations, duration_s=12, max_turns=8)
    assert any("later" in line["evidence_ids"] for line in result["lines"])
    assert all(line["at_s"] >= 5 for line in result["lines"] if "later" in line["evidence_ids"])
    assert result["source_duration_s"] == 12
    assert result["turns"][0]["source_at_s"] == 0
    assert result["turns"][0]["generation_elapsed_s"] >= 0


@pytest.mark.parametrize("mode", ["future", "origin", "voice", "word_budget"])
async def test_recorded_custom_roles_cannot_bypass_contracts(mode) -> None:
    async def run(role, view):
        return TurnResult(
            role=role,
            snapshot_version=view.version + (mode == "origin"),
            cutoff_s=view.cutoff_s,
            line=SpeechLine(
                voice=("analyst" if role == "caller" else "caller") if mode == "voice" else role,
                text="word " * (view.speech_budget_words + 1)
                if mode == "word_budget"
                else "The ball moves forward.",
                evidence_ids=["future"] if mode == "future" else ["early"],
            ),
        )

    result = await prepare_dialogue(
        run, [event("early", 0, [0]), event("future", 10, [10])], duration_s=12, max_turns=2
    )
    assert result["lines"] == []
    assert all(turn.get("rejected") for turn in result["turns"])


async def test_recorded_budget_stop_preserves_partial_audit() -> None:
    calls = 0

    async def run(role, view):
        nonlocal calls
        calls += 1
        raise BudgetExceeded("cap reached")

    result = await prepare_dialogue(run, [event("early", 0, [0])], duration_s=12)
    assert calls == 1
    assert result["lines"] == []
    assert result["turns"][0]["rejected"] == "model budget exhausted"


async def test_abstaining_twice_does_not_loop_on_the_same_snapshot() -> None:
    seen = []

    async def run(role, view):
        seen.append((role, view.cutoff_s))
        return TurnResult(
            role=role,
            snapshot_version=view.version,
            cutoff_s=view.cutoff_s,
            reason="Nothing new to add.",
        )

    result = await prepare_dialogue(run, [event("early", 0, [0])], duration_s=12)
    assert len(seen) == 2
    assert result["lines"] == []


async def test_analyst_gets_space_after_a_major_call_and_room_for_a_response() -> None:
    from commentary.replay.recorded import ready_at

    goal = Observation(
        id="goal",
        at_s=0,
        kind="goal",
        description="The ball crosses the line.",
        action=FootballAction(
            type="goal", certainty="confirmed", cues=["ball_in_net"], support_at_s=[0]
        ),
    )
    snapshots = []

    async def run(role, view):
        snapshots.append(view)
        assert ready_at(goal) <= view.cutoff_s
        return TurnResult(
            role=role,
            snapshot_version=view.version,
            cutoff_s=view.cutoff_s,
            line=SpeechLine(voice=role, text="He's scored!", evidence_ids=["goal"])
            if role == "caller"
            else None,
        )

    await prepare_dialogue(run, [goal], duration_s=20, max_turns=2)
    assert snapshots[1].cutoff_s >= 2 / 2.6 + 1.2
    assert snapshots[1].speech_budget_words == 28


async def test_short_slot_can_go_to_caller_when_analyst_cannot_fit() -> None:
    seen = []
    goal = Observation(id="goal", at_s=10, kind="goal", description="A goal.")

    async def run(role, view):
        seen.append((role, view.cutoff_s))
        line = None
        if role == "caller":
            line = SpeechLine(
                voice=role, text="Forward...", evidence_ids=[view.observations[-1].id]
            )
        return TurnResult(
            role=role, snapshot_version=view.version, cutoff_s=view.cutoff_s, line=line
        )

    await prepare_dialogue(
        run, [event("early", 0, [0]), event("approach", 9, [9]), goal], duration_s=20, max_turns=6
    )
    assert ("caller", 9) in seen


def test_unknown_celebration_tag_is_weakened_without_promoting_a_goal() -> None:
    from commentary.replay.recorded import RecordedObservation

    scene = RecordedObservation.model_validate(
        {
            "id": "celebration",
            "at_s": 1,
            "kind": "celebration",
            "description": "Players celebrate.",
            "action": {
                "type": "unknown",
                "certainty": "confirmed",
                "cues": [],
                "support_at_s": [1],
            },
        }
    )
    assert scene.action.certainty == "uncertain"
    assert scene._scene_certainty_downgraded
    with pytest.raises(ValueError):
        RecordedObservation.model_validate(
            {
                "id": "goal",
                "at_s": 1,
                "kind": "goal",
                "description": "Players celebrate.",
                "action": {
                    "type": "unknown",
                    "certainty": "confirmed",
                    "cues": [],
                    "support_at_s": [1],
                },
            }
        )


async def test_full_recorded_observer_cannot_invent_frame_timestamps() -> None:
    from commentary.llm.fake import ScriptedBackend
    from commentary.recorded_demo import Frame
    from commentary.replay.recorded import RecordedObservations, observe_recorded

    backend = ScriptedBackend()
    backend.always(
        "recorded_observer",
        RecordedObservations.model_validate(
            {
                "observations": [
                    {
                        "id": "pass",
                        "at_s": 0.25,
                        "kind": "play",
                        "description": "A pass.",
                        "action": {
                            "type": "pass",
                            "certainty": "confirmed",
                            "cues": ["ball_motion"],
                            "support_at_s": [0.25],
                        },
                    }
                ]
            }
        ),
    )
    with pytest.raises(ValueError, match="supplied frames"):
        await observe_recorded(
            backend,
            [Frame(0, b"test"), Frame(1, b"test")],
            model="offline",
            identity_model="offline",
            research=None,
            pack=None,
        )
