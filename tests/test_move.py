from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from commentary.move import MoveBuffer
from commentary.schemas import (
    Action,
    ActionBeat,
    CallerLine,
    Event,
    IdentitySource,
    PlayerIdentity,
    Scene,
    Side,
)


def identity(
    name: str | None = None,
    *,
    number: int | None = None,
    confidence: float = 0.9,
) -> PlayerIdentity:
    return PlayerIdentity(
        name=name,
        number=number,
        side=Side.AWAY,
        confidence=confidence,
        source=IdentitySource.SHIRT_NUMBER if number is not None else IdentitySource.UNKNOWN,
    )


def action(
    kind: Action,
    ts: float,
    *,
    actor: PlayerIdentity | None = None,
    target: PlayerIdentity | None = None,
    confidence: float = 0.9,
    origin: str | None = None,
    destination: str | None = None,
    direction: str | None = None,
    delivery: str | None = None,
    outcome: str | None = None,
    evidence: str | None = None,
) -> ActionBeat:
    return ActionBeat(
        video_ts=ts,
        action=kind,
        actor=actor,
        target=target,
        origin_zone=origin,
        destination_zone=destination,
        direction=direction,
        delivery=delivery,
        outcome=outcome,
        confidence=confidence,
        evidence=evidence,
    )


def form(
    *actions: ActionBeat,
    scene: Scene = Scene.LIVE_PLAY,
    event: Event = Event.BUILD_UP,
    side: Side = Side.AWAY,
    team: str | None = "Barcelona",
) -> CallerLine:
    return CallerLine(
        scene=scene,
        event=event,
        side=side,
        team=team,
        actions=list(actions),
        confidence=0.9,
        speak=True,
        line="A caller line.",
    )


def test_retains_only_the_short_live_window_in_chronological_order() -> None:
    buffer = MoveBuffer(window_s=12.0)

    buffer.observe(1.0, form(action(Action.CARRY, 1.0)))
    buffer.observe(8.0, form(action(Action.PASS, 8.0)))
    buffer.observe(
        14.0,
        form(
            action(Action.CROSS, 13.5),
            action(Action.RECEIVE, 12.0),
        ),
    )

    current = buffer.current
    assert current is not None
    assert [(beat.action, beat.video_ts) for beat in current.beats] == [
        (Action.PASS, 8.0),
        (Action.RECEIVE, 12.0),
        (Action.CROSS, 13.5),
    ]


def test_adjacent_duplicate_observations_merge_and_keep_richer_fact() -> None:
    buffer = MoveBuffer()
    buffer.observe(
        10.0,
        form(
            action(
                Action.PASS,
                9.8,
                origin="right channel",
                destination="edge of box",
                evidence="A pass goes inside.",
                confidence=0.7,
            )
        ),
    )
    buffer.observe(
        10.5,
        form(
            action(
                Action.PASS,
                10.0,
                actor=identity("Jules Koundé", number=23),
                origin="right channel",
                destination="edge of box",
                evidence="Koundé plays a first-touch pass inside.",
            )
        ),
    )

    current = buffer.current
    assert current is not None
    assert len(current.beats) == 1
    merged = current.beats[0]
    assert merged.video_ts == 9.8
    assert merged.actor is not None
    assert merged.actor.name == "Jules Koundé"
    assert merged.actor.number == 23
    assert merged.confidence == 0.9
    assert merged.evidence == "Koundé plays a first-touch pass inside."


def test_action_transitions_never_merge_even_at_the_same_anchor() -> None:
    buffer = MoveBuffer()
    kinds = [Action.PASS, Action.RECEIVE, Action.CROSS, Action.SHOT, Action.FINISH]

    buffer.observe(
        20.0,
        form(*(action(kind, 20.0 + index * 0.1) for index, kind in enumerate(kinds))),
    )

    current = buffer.current
    assert current is not None
    assert [beat.action for beat in current.beats] == kinds


def test_an_overlapping_multi_action_window_merges_each_repeated_touch() -> None:
    buffer = MoveBuffer()
    kounde = identity("Jules Koundé", number=23)
    yamal = identity("Lamine Yamal", number=27)
    buffer.observe(
        20.0,
        form(
            action(Action.PASS, 19.0, actor=kounde, target=yamal),
            action(Action.RECEIVE, 19.4, actor=yamal),
        ),
    )

    buffer.observe(
        20.5,
        form(
            action(
                Action.PASS,
                19.1,
                actor=kounde,
                target=yamal,
                evidence="Koundé finds Yamal inside.",
            ),
            action(
                Action.RECEIVE,
                19.5,
                actor=yamal,
                evidence="Yamal takes the ball first time.",
            ),
            action(Action.CROSS, 20.3, actor=yamal),
        ),
    )

    current = buffer.current
    assert current is not None
    assert [beat.action for beat in current.beats] == [
        Action.PASS,
        Action.RECEIVE,
        Action.CROSS,
    ]
    assert current.beats[0].evidence == "Koundé finds Yamal inside."
    assert current.beats[1].evidence == "Yamal takes the ball first time."


def test_two_contradictory_passes_remain_distinct() -> None:
    buffer = MoveBuffer()
    buffer.observe(
        10.0,
        form(action(Action.PASS, 10.0, actor=identity("Jules Koundé", number=23))),
    )
    buffer.observe(
        10.2,
        form(action(Action.PASS, 10.2, actor=identity("Lamine Yamal", number=27))),
    )

    current = buffer.current
    assert current is not None
    assert len(current.beats) == 2


def test_live_goal_freezes_an_immutable_sequence_including_the_finish() -> None:
    buffer = MoveBuffer()
    pass_beat = action(Action.PASS, 25.0, actor=identity("Jules Koundé", number=23))
    buffer.observe(25.0, form(pass_beat))

    frozen = buffer.observe(
        27.0,
        form(
            action(
                Action.FINISH,
                26.8,
                actor=identity("Ferran Torres", number=7),
                outcome="goal at the near post",
            ),
            event=Event.GOAL,
        ),
    )

    assert frozen is buffer.frozen
    assert frozen is not None
    assert buffer.current is None
    assert frozen.goal_ts == 27.0
    assert [beat.action for beat in frozen.beats] == [Action.PASS, Action.FINISH]
    assert frozen.beats[-1].outcome == "goal at the near post"
    with pytest.raises(FrozenInstanceError):
        frozen.team = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        frozen.beats[-1].outcome = "changed"  # type: ignore[misc]

    # The snapshot copied the schema objects and does not change if a caller
    # form is later amended or a new live move begins.
    pass_beat.outcome = "mutated source"
    buffer.observe(31.0, form(action(Action.CARRY, 31.0)))
    assert frozen.beats[0].outcome is None
    assert [beat.action for beat in frozen.beats] == [Action.PASS, Action.FINISH]


def test_replay_and_other_non_live_forms_cannot_mutate_or_freeze_the_move() -> None:
    buffer = MoveBuffer()
    buffer.observe(10.0, form(action(Action.CROSS, 10.0)))
    before = buffer.current

    replay_result = buffer.observe(
        100.0,
        form(
            action(Action.FINISH, 100.0),
            scene=Scene.REPLAY,
            event=Event.GOAL,
            side=Side.HOME,
            team="Betis",
        ),
    )
    buffer.observe(
        101.0,
        form(action(Action.SHOT, 101.0), scene=Scene.CLOSE_UP, event=Event.GOAL),
    )

    assert replay_result is None
    assert buffer.current == before
    assert buffer.frozen is None


def test_known_possession_change_starts_a_new_move() -> None:
    buffer = MoveBuffer()
    buffer.observe(5.0, form(action(Action.PASS, 5.0)))

    buffer.observe(
        6.0,
        form(
            action(Action.CARRY, 6.0),
            side=Side.HOME,
            team="Betis",
        ),
    )

    current = buffer.current
    assert current is not None
    assert current.side is Side.HOME
    assert current.team == "Betis"
    assert [beat.action for beat in current.beats] == [Action.CARRY]


def test_weak_actions_do_not_enter_the_useful_fact_buffer() -> None:
    buffer = MoveBuffer(min_confidence=0.5)

    buffer.observe(5.0, form(action(Action.PASS, 5.0, confidence=0.49)))

    assert buffer.current is None


@pytest.mark.parametrize("window", [9.9, 15.1])
def test_window_is_constrained_to_the_planned_ten_to_fifteen_seconds(window: float) -> None:
    with pytest.raises(ValueError, match="between 10 and 15"):
        MoveBuffer(window_s=window)
