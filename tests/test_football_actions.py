"""Action contracts and classification diagnostics, with no model calls."""

from typing import Any

import pytest
from pydantic import ValidationError

from commentary.football import FootballAction
from commentary.llm.fake import ScriptedBackend
from commentary.llm.schema import strict_schema
from commentary.recorded_demo import Frame, Observation
from commentary.replay.agents import RoleAgents
from commentary.replay.contracts import AgentAction, ReplayResult, RuntimeSnapshot, SpeechLine
from commentary.replay.evaluation import EvaluationCase, evaluate_replay
from commentary.replay.preparation import ReplayObservation, ReplayObserver, ReplayObserverResult


def action(kind: str = "pass", **updates: Any) -> FootballAction:
    data = {
        "type": kind,
        "certainty": "confirmed",
        "cues": ["ball_motion"],
        "support_at_s": [1],
    }
    data.update(updates)
    return FootballAction.model_validate(data)


@pytest.mark.parametrize(
    "kind,cue",
    [
        ("goal", "ball_in_net"),
        ("foul", "referee_signal"),
        ("offside", "offside_signal"),
        ("penalty_awarded", "referee_signal"),
        ("yellow_card", "yellow_card_visible"),
        ("red_card", "red_card_visible"),
        ("substitution", "substitution_board"),
    ],
)
def test_decisive_events_need_specific_visible_support(kind: str, cue: str) -> None:
    with pytest.raises(ValidationError, match="event-specific"):
        action(kind)
    assert action(kind, cues=[cue]).type == kind


@pytest.mark.parametrize(
    "updates",
    [
        {"support_at_s": []},
        {"support_at_s": [float("nan")]},
        {"support_at_s": [2, 1]},
        {"support_at_s": [1, 1]},
        {"cues": []},
        {"type": "unknown"},
        {"ball_path": "aerial"},
    ],
)
def test_unsupported_action_claims_are_rejected(updates: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        action(**updates)


def test_new_perception_requires_labels_but_legacy_results_remain_readable() -> None:
    data = dict(id="e1", at_s=1, kind="play", description="A pass.")
    assert Observation.model_validate(data).action is None
    with pytest.raises(ValidationError, match="action"):
        ReplayObservation.model_validate(data)
    schema = strict_schema(ReplayObserverResult)
    assert "action" in schema["$defs"]["ReplayObservation"]["required"]


def test_replay_goal_is_historical_and_uncertain_goal_cannot_trigger_live_goal() -> None:
    goal = action("goal", cues=["ball_in_net"])
    data = dict(id="e1", at_s=1, description="The finish is shown again.", action=goal)
    assert ReplayObservation(kind="replay", **data).kind == "replay"
    with pytest.raises(ValidationError, match="live goal"):
        ReplayObservation(
            id="e2",
            at_s=1,
            kind="goal",
            description="Maybe in.",
            action=action("goal", certainty="uncertain"),
        )
    with pytest.raises(ValidationError, match="after"):
        ReplayObservation(
            id="e3",
            at_s=0,
            kind="play",
            description="A pass.",
            action=action(),
        )


async def test_support_cannot_cite_unsupplied_intermediate_frame() -> None:
    backend = ScriptedBackend()
    backend.always(
        "replay_observer",
        ReplayObserverResult(
            observations=[
                ReplayObservation(
                    id="e1", at_s=2, kind="play", description="A pass.", action=action()
                )
            ]
        ),
    )
    observer = ReplayObserver(backend, model="offline")
    with pytest.raises(ValueError, match="exact supplied"):
        await observer([Frame(0, b"test"), Frame(2, b"test")], 2)
    assert observer.audit[0]["status"] == "failed"


@pytest.mark.parametrize("kind,certainty", [("play", "uncertain"), ("replay", "confirmed")])
async def test_caller_cannot_turn_uncertainty_or_replay_into_live_call(
    kind: str,
    certainty: str,
) -> None:
    event = ReplayObservation.model_validate(
        dict(
            id="e1",
            at_s=1,
            kind=kind,
            description="A possible pass.",
            action=action(certainty=certainty),
        )
    )
    snapshot = RuntimeSnapshot(version=1, cutoff_s=1, observations=[event])
    backend = ScriptedBackend()
    backend.always(
        "replay_caller",
        AgentAction(
            kind="speak",
            line=SpeechLine(
                voice="caller",
                text="The pass goes forward.",
                evidence_ids=["e1"],
            ),
        ),
    )
    agents = RoleAgents(backend, "offline", "offline")
    with pytest.raises(ValueError, match="uncertain actions or replay"):
        await agents.run("caller", snapshot)


def reference_case(**updates: Any) -> EvaluationCase:
    data = dict(
        id="actions",
        match_id="fictional",
        split="development",
        duration_s=4,
        source=dict(
            kind="synthetic",
            locator="synthetic://actions",
            provenance="Fictional test labels.",
            usage_terms="Project-owned fixture.",
            annotation_status="synthetic",
        ),
        events=[dict(id="r1", at_s=1, kind="play", action_type="pass", ball_path="ground")],
    )
    data.update(updates)
    return EvaluationCase.model_validate(data)


def report(prediction: FootballAction | None, **case_updates: Any) -> dict[str, Any]:
    result = ReplayResult(
        observations=[
            Observation(
                id="e1",
                at_s=1,
                kind="play",
                description="Visible action.",
                action=prediction,
            )
        ]
    )
    return evaluate_replay(result, reference_case(**case_updates))["actions"]


def test_action_confusion_is_not_hidden_by_successful_coarse_event_match() -> None:
    scores = report(action("shot"), action_labels_complete=True)
    assert scores["labeled_recall"] == 0
    assert scores["precision"] == 0
    assert scores["confusions"] == [{"reference": "pass", "prediction": "shot", "count": 1}]
    assert scores["per_action"]["shot"]["precision"] == 0


def test_ground_pass_misread_as_aerial_is_measured_separately() -> None:
    scores = report(action(ball_path="aerial", cues=["ball_flight"]))
    assert scores["labeled_recall"] == 1
    assert scores["precision"] is None  # sparse references cannot establish false positives
    assert scores["ball_path"]["correct"] == 0
    assert scores["ball_path"]["confusions"] == [
        {"reference": "ground", "prediction": "aerial", "count": 1},
    ]


@pytest.mark.parametrize("prediction", [None, action(certainty="uncertain")])
def test_missing_and_uncertain_labels_are_abstentions(prediction: FootballAction | None) -> None:
    scores = report(prediction)
    assert scores["labeled_recall"] == 0
    assert scores["confirmed_predictions"] == 0
    assert scores["confusions"][0]["prediction"] == "abstain"


def test_unmatched_reference_and_extra_detection_affect_recall_and_precision() -> None:
    result = ReplayResult(
        observations=[
            Observation(
                id="e1",
                at_s=4,
                kind="play",
                description="A pass later.",
                action=action(support_at_s=[4]),
            )
        ]
    )
    scores = evaluate_replay(result, reference_case(action_labels_complete=True), 0.5)["actions"]
    assert scores["labeled_recall"] == 0
    assert scores["precision"] == 0
    assert scores["confusions"][0]["prediction"] == "abstain"
