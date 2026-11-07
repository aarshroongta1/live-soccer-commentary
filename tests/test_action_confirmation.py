"""Perception regressions found in the first paid action checks."""

import pytest

from commentary.football import FootballAction
from commentary.recorded_demo import Observation
from commentary.replay.cache import SavedVision
from commentary.replay.contracts import RuntimeSnapshot


def test_shot_can_have_an_observed_aerial_path() -> None:
    result = FootballAction(
        type="shot",
        certainty="confirmed",
        ball_path="aerial",
        cues=["ball_flight"],
        support_at_s=[1],
    )
    assert result.ball_path == "aerial"


def test_later_available_frame_can_confirm_an_earlier_action() -> None:
    observation = Observation(
        id="shot",
        at_s=0.5,
        kind="play",
        description="A shot is struck.",
        action=FootballAction(
            type="shot", certainty="confirmed", cues=["ball_motion"], support_at_s=[0.5, 1]
        ),
    )
    assert RuntimeSnapshot(version=1, cutoff_s=1, observations=[observation]).observations
    with pytest.raises(ValueError, match="support.*visible"):
        RuntimeSnapshot(version=1, cutoff_s=0.5, observations=[observation])


def test_snapshot_rejects_support_after_its_cutoff_even_for_custom_observers() -> None:
    observation = Observation.model_construct(
        id="shot",
        at_s=0.5,
        kind="play",
        description="A shot.",
        action=FootballAction(
            type="shot", certainty="confirmed", cues=["ball_motion"], support_at_s=[1]
        ),
    )
    with pytest.raises(ValueError, match="support.*visible"):
        RuntimeSnapshot(version=1, cutoff_s=0.5, observations=[observation])


def test_saved_vision_rejects_support_from_unsupplied_frames() -> None:
    observation = Observation(
        id="pass",
        at_s=1,
        kind="play",
        description="A pass.",
        action=FootballAction(
            type="pass", certainty="confirmed", cues=["ball_motion"], support_at_s=[0.25]
        ),
    )
    with pytest.raises(ValueError, match="support.*supplied"):
        SavedVision(
            [
                {
                    "frames_at_s": [0, 1],
                    "cutoff_s": 1,
                    "usage": {"latency_s": 1},
                    "resolved": [observation.model_dump()],
                }
            ]
        )


def test_path_without_a_visible_cue_abstains_without_losing_a_valid_goal() -> None:
    action = FootballAction(
        type="goal",
        certainty="confirmed",
        ball_path="ground",
        cues=["ball_in_net"],
        support_at_s=[1],
    )
    assert action.type == "goal" and action.certainty == "confirmed"
    assert action.ball_path == "unknown"
    assert action._unsupported_ball_path == "ground"


def test_action_can_cite_all_eight_supplied_frames() -> None:
    action = FootballAction(
        type="red_card",
        certainty="confirmed",
        cues=["red_card_visible"],
        support_at_s=[i / 2 for i in range(8)],
    )
    assert len(action.support_at_s) == 8
