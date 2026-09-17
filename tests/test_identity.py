"""Deterministic coverage for structured player resolution and continuity."""

from commentary.identity import IdentityContinuity, rank_players
from commentary.schemas import (
    Action,
    ActionBeat,
    CallerLine,
    Event,
    IdentitySource,
    KnowledgePack,
    Player,
    PlayerIdentity,
    Scene,
    Side,
    Sighting,
    TeamSheet,
    Trigger,
)


def pack() -> KnowledgePack:
    return KnowledgePack(
        home=TeamSheet(
            name="Home",
            starters=[
                Player(name="Home Fullback", number=23, position="Right Back"),
                Player(name="Home Midfielder", number=8, position="Center Midfield"),
            ],
        ),
        away=TeamSheet(
            name="Away",
            starters=[
                Player(name="Jules Koundé", number=23, position="Right Back"),
                Player(name="Ferran Torres", number=19, position="Center Forward"),
                Player(name="Away Midfielder", number=8, position="Center Midfield"),
            ],
        ),
    )


def identity(
    *,
    name: str | None = None,
    number: int | None = None,
    side: Side = Side.AWAY,
    role: str | None = None,
    confidence: float = 0.95,
    source: IdentitySource = IdentitySource.SHIRT_NUMBER,
) -> PlayerIdentity:
    return PlayerIdentity(
        name=name,
        number=number,
        side=side,
        role=role,
        confidence=confidence,
        source=source,
    )


def form(
    *actions: ActionBeat,
    side: Side = Side.AWAY,
    event: Event = Event.BUILD_UP,
) -> CallerLine:
    return CallerLine(
        scene=Scene.LIVE_PLAY,
        event=event,
        side=side,
        confidence=0.9,
        speak=False,
        actions=list(actions),
    )


def test_number_side_and_role_rank_the_correct_pack_player() -> None:
    ranked = rank_players(pack(), identity(number=23, role="right-back"))

    assert [(item.side, item.player.name) for item in ranked] == [(Side.AWAY, "Jules Koundé")]


def test_number_read_resolves_to_the_pack_name_without_changing_the_action() -> None:
    continuity = IdentityContinuity(pack())
    line = continuity.resolve(
        form(ActionBeat(action=Action.CROSS, actor=identity(number=23), confidence=0.96)),
        10.0,
    )

    actor = line.actions[0].actor
    assert actor is not None
    assert actor.name == "Jules Koundé"
    assert actor.number == 23
    assert line.actions[0].action is Action.CROSS
    assert line.actions[0].confidence == 0.96


def test_verified_receiver_carries_only_through_a_short_continuous_possession() -> None:
    continuity = IdentityContinuity(pack())
    continuity.resolve(
        form(
            ActionBeat(
                action=Action.PASS,
                actor=identity(number=23),
                target=identity(number=19),
                confidence=0.9,
            )
        ),
        10.0,
    )

    carried = continuity.resolve(
        form(
            ActionBeat(
                action=Action.CARRY,
                actor=identity(
                    role="forward",
                    source=IdentitySource.VISIBLE_ROLE,
                    confidence=0.4,
                ),
                confidence=0.94,
            )
        ),
        14.0,
    )

    actor = carried.actions[0].actor
    assert actor is not None
    assert actor.name == "Ferran Torres"
    assert actor.source is IdentitySource.POSSESSION_CONTINUITY
    assert actor.role == "forward"

    expired = continuity.resolve(
        form(ActionBeat(action=Action.SHOT, actor=None, confidence=0.9)),
        18.1,
    )
    assert expired.actions[0].actor is None


def test_camera_cut_does_not_inherit_a_previous_player() -> None:
    continuity = IdentityContinuity(pack())
    continuity.resolve(
        form(ActionBeat(action=Action.CARRY, actor=identity(number=23), confidence=0.9)), 10.0
    )

    after_cut = continuity.resolve(
        form(
            ActionBeat(
                action=Action.CROSS,
                actor=identity(
                    role="right-back",
                    source=IdentitySource.VISIBLE_ROLE,
                    confidence=0.4,
                ),
                confidence=0.96,
            )
        ),
        11.0,
        [Trigger.CAMERA_CUT],
    )

    actor = after_cut.actions[0].actor
    assert actor is not None
    assert actor.name is None
    assert actor.role == "right-back"
    assert after_cut.actions[0].action is Action.CROSS


def test_turnover_and_contradictory_shirt_read_clear_the_carry() -> None:
    continuity = IdentityContinuity(pack())
    continuity.resolve(
        form(ActionBeat(action=Action.CARRY, actor=identity(number=23), confidence=0.9)), 10.0
    )

    turnover = continuity.resolve(
        form(
            ActionBeat(action=Action.CARRY, actor=None, confidence=0.9),
            side=Side.HOME,
            event=Event.INTERCEPTION,
        ),
        12.0,
    )
    assert turnover.actions[0].actor is None

    continuity.resolve(
        form(ActionBeat(action=Action.CARRY, actor=identity(number=23), confidence=0.9)), 14.0
    )
    contradiction = continuity.resolve(
        form(
            ActionBeat(
                action=Action.CARRY,
                actor=identity(name="Jules Koundé", number=8),
                confidence=0.9,
            )
        ),
        15.0,
    )
    actor = contradiction.actions[0].actor
    assert actor is not None
    assert actor.name is None
    assert actor.number == 8

    after_contradiction = continuity.resolve(
        form(ActionBeat(action=Action.CROSS, actor=None, confidence=0.95)), 16.0
    )
    assert after_contradiction.actions[0].actor is None


def test_role_only_identity_is_not_upgraded_to_a_likely_roster_name() -> None:
    continuity = IdentityContinuity(pack())
    line = continuity.resolve(
        form(
            ActionBeat(
                action=Action.CROSS,
                actor=identity(
                    role="right-back",
                    source=IdentitySource.VISIBLE_ROLE,
                    confidence=0.95,
                ),
                confidence=0.98,
            )
        ),
        10.0,
    )

    actor = line.actions[0].actor
    assert actor is not None
    assert actor.name is None
    assert actor.role == "right-back"
    assert line.actions[0].action is Action.CROSS


def test_cross_and_layoff_transfer_the_verified_receiver_to_the_next_action() -> None:
    continuity = IdentityContinuity(pack())
    crossed = continuity.resolve(
        form(
            ActionBeat(
                action=Action.CROSS,
                actor=identity(number=23),
                target=identity(number=19),
                confidence=0.96,
            )
        ),
        10.0,
    )
    assert crossed.actions[0].target is not None
    assert crossed.actions[0].target.name == "Ferran Torres"

    finish = continuity.resolve(
        form(ActionBeat(action=Action.FINISH, actor=None, confidence=0.97)), 11.0
    )
    assert finish.actions[0].actor is not None
    assert finish.actions[0].actor.name == "Ferran Torres"

    continuity.resolve(
        form(
            ActionBeat(
                action=Action.LAYOFF,
                actor=identity(number=19),
                target=identity(number=23),
                confidence=0.9,
            )
        ),
        12.0,
    )
    after_layoff = continuity.resolve(
        form(ActionBeat(action=Action.CROSS, actor=None, confidence=0.94)), 13.0
    )
    assert after_layoff.actions[0].actor is not None
    assert after_layoff.actions[0].actor.name == "Jules Koundé"


def test_a_pass_target_does_not_erase_the_carried_passer_before_the_pass() -> None:
    continuity = IdentityContinuity(pack())
    continuity.resolve(
        form(ActionBeat(action=Action.CARRY, actor=identity(number=23), confidence=0.9)), 10.0
    )

    passed = continuity.resolve(
        form(
            ActionBeat(
                action=Action.PASS,
                actor=None,
                target=identity(number=19),
                confidence=0.94,
            )
        ),
        11.0,
    )
    assert passed.actions[0].actor is not None
    assert passed.actions[0].actor.name == "Jules Koundé"
    assert passed.actions[0].target is not None
    assert passed.actions[0].target.name == "Ferran Torres"


def test_ambiguous_or_unsupported_identity_never_becomes_a_roster_name() -> None:
    continuity = IdentityContinuity(pack())
    ambiguous = continuity.resolve(
        form(
            ActionBeat(
                action=Action.CARRY,
                actor=identity(number=8, side=Side.UNKNOWN),
                confidence=0.9,
            ),
            side=Side.UNKNOWN,
        ),
        10.0,
    )
    assert ambiguous.actions[0].actor is not None
    assert ambiguous.actions[0].actor.name is None

    unsupported = continuity.resolve(
        form(
            ActionBeat(
                action=Action.CARRY,
                actor=identity(
                    name="Jules Koundé",
                    role="right-back",
                    source=IdentitySource.VISIBLE_ROLE,
                ),
                confidence=0.9,
            )
        ),
        11.0,
    )
    assert unsupported.actions[0].actor is not None
    assert unsupported.actions[0].actor.name is None


def test_low_confidence_and_no_pack_do_not_create_or_destroy_a_name() -> None:
    low_confidence = IdentityContinuity(pack()).resolve(
        form(
            ActionBeat(
                action=Action.CARRY,
                actor=identity(
                    name="Jules Koundé",
                    source=IdentitySource.SHIRT_NAME,
                    confidence=0.4,
                ),
                confidence=0.9,
            )
        ),
        10.0,
    )
    assert low_confidence.actions[0].actor is not None
    assert low_confidence.actions[0].actor.name is None

    without_pack = IdentityContinuity(None).resolve(
        form(ActionBeat(action=Action.CARRY, actor=identity(number=23), confidence=0.9)), 10.0
    )
    assert without_pack.actions[0].actor is not None
    assert without_pack.actions[0].actor.number == 23
    assert without_pack.actions[0].actor.name is None


def test_matching_sighting_supplies_the_missing_side_and_cuts_use_video_time() -> None:
    continuity = IdentityContinuity(pack())
    from_sighting = form(
        ActionBeat(
            action=Action.CARRY,
            actor=identity(number=23, side=Side.UNKNOWN, source=IdentitySource.UNKNOWN),
            confidence=0.9,
        ),
        side=Side.UNKNOWN,
    ).model_copy(
        update={"sightings": [Sighting(number=23, name="Jules Koundé", side=Side.AWAY)]}
    )
    resolved = continuity.resolve(from_sighting, 10.0)
    assert resolved.actions[0].actor is not None
    assert resolved.actions[0].actor.name == "Jules Koundé"

    continuity.resolve(
        form(
            ActionBeat(
                action=Action.CARRY,
                actor=identity(number=23),
                video_ts=10.0,
                confidence=0.9,
            )
        ),
        10.0,
    )
    after_cut = continuity.resolve(
        form(ActionBeat(action=Action.CROSS, actor=None, video_ts=12.0, confidence=0.95)),
        12.0,
        cut_timestamps=[11.0],
    )
    assert after_cut.actions[0].actor is None

    continuity.resolve(
        form(
            ActionBeat(
                action=Action.CARRY,
                actor=identity(number=23),
                video_ts=20.0,
                confidence=0.9,
            )
        ),
        20.0,
    )
    expired = continuity.resolve(
        form(ActionBeat(action=Action.SHOT, actor=None, video_ts=28.1, confidence=0.9)),
        21.0,
    )
    assert expired.actions[0].actor is None
