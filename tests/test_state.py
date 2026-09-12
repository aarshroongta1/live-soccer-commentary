from dataclasses import dataclass

import pytest

from commentary.schemas import (
    CallerLine,
    Event,
    Incident,
    KnowledgePack,
    Player,
    Scene,
    Side,
    Sighting,
    TeamSheet,
    WireEvent,
)
from commentary.state import (
    EntityRegistry,
    MatchStateTracker,
)


@dataclass
class StubBoard:
    """Stands in for a confirmed BoardTracker, without importing perception."""

    home_score: int | None = None
    away_score: int | None = None
    clock: str | None = None
    in_replay: bool = False
    bug_missing: bool = False


def pack() -> KnowledgePack:
    return KnowledgePack(
        home=TeamSheet(
            name="Arsenal",
            starters=[Player(name="Bukayo Saka", number=7), Player(name="Kai Havertz", number=29)],
            bench=[Player(name="Ethan Nwaneri", number=53)],
        ),
        away=TeamSheet(
            name="Chelsea",
            starters=[Player(name="Cole Palmer", number=10)],
        ),
    )


def caller(**kwargs: object) -> CallerLine:
    fields: dict[str, object] = {
        "scene": Scene.LIVE_PLAY,
        "event": Event.NONE,
        "confidence": 0.8,
        "speak": True,
        "line": "",
    }
    fields.update(kwargs)
    return CallerLine.model_validate(fields)


def test_registry_confidence_decays_with_time_since_the_sighting():
    registry = EntityRegistry(half_life_s=600.0)
    registry.believe(9, "Havertz", 100.0)

    assert registry.confidence(9, 100.0) == pytest.approx(1.0)
    assert registry.confidence(9, 700.0) == pytest.approx(0.5)
    assert registry.confidence(9, 2500.0) < 0.1
    assert registry.confidence(11, 100.0) == 0.0


def test_a_substitution_overwrites_the_number():
    registry = EntityRegistry(half_life_s=600.0)
    registry.believe(7, "Saka", 0.0)
    assert registry.name_for(7) == "Saka"

    registry.believe(7, "Nwaneri", 4200.0)
    assert registry.name_for(7) == "Nwaneri"
    assert registry.confidence(7, 4200.0) == pytest.approx(1.0)


def test_a_roster_is_believed_less_than_a_sighting():
    registry = EntityRegistry(half_life_s=600.0)
    registry.seed(pack())

    assert registry.confidence(29, 0.0, side=Side.HOME) < 1.0
    registry.believe(29, "Kai Havertz", 0.0, side=Side.HOME)
    assert registry.confidence(29, 0.0, side=Side.HOME) == pytest.approx(1.0)


def test_a_number_both_teams_wear_needs_a_side():
    registry = EntityRegistry()
    registry.believe(10, "Cole Palmer", 0.0, side=Side.AWAY)
    registry.believe(10, "Martin Odegaard", 0.0, side=Side.HOME)

    assert registry.name_for(10) is None
    assert registry.name_for(10, side=Side.HOME) == "Martin Odegaard"
    assert registry.name_for(10, side=Side.AWAY) == "Cole Palmer"


def test_names_the_caller_reads_off_a_graphic_reach_the_registry():
    tracker = MatchStateTracker.from_pack(pack())
    assert tracker.state.home == "Arsenal"

    tracker.apply_caller(
        caller(
            scene=Scene.GRAPHIC,
            side=Side.HOME,
            sightings=[Sighting(number=53, name="Ethan Nwaneri"), Sighting(name="crowd")],
        ),
        ts=4000.0,
    )

    assert tracker.registry.name_for(53, side=Side.HOME) == "Ethan Nwaneri"
    assert tracker.registry.confidence(53, 4000.0, side=Side.HOME) == pytest.approx(1.0)
    assert tracker.state.on_pitch["53"] == "Ethan Nwaneri"


def test_summary_says_when_the_bug_is_gone_rather_than_calling_it_a_replay():
    tracker = MatchStateTracker("Arsenal", "Madrid")
    tracker.apply_board(StubBoard(clock="10:00", in_replay=False, bug_missing=True))
    summary = tracker.summary()
    assert "no score bug visible" in summary
    assert "replay" not in summary


# -- the statistician --------------------------------------------------------


def wire_pack() -> KnowledgePack:
    return KnowledgePack(
        home=TeamSheet(
            name="Argentina",
            short="ARG",
            starters=[Player(name="Alexis Mac Allister", number=20)],
            bench=[Player(name="Paulo Dybala", number=21)],
        ),
        away=TeamSheet(
            name="France", short="FRA", starters=[Player(name="Adrien Rabiot", number=14)]
        ),
    )


def wire_event(**fields: object) -> WireEvent:
    return WireEvent.model_validate({"clock_s": 0.0, "period": 1, **fields})


def test_the_wire_may_move_the_score():
    """The one source other than the board that may, and it may because a feed
    is not a model guessing at a picture — it is reporting the match."""
    tracker = MatchStateTracker.from_pack(wire_pack())
    what = tracker.apply_wire(
        wire_event(
            event=Event.GOAL, side=Side.HOME, player="Alexis Mac Allister", home_score=1
        ),
        100.0,
    )
    assert (tracker.state.home_score, tracker.state.away_score) == (1, 0)
    assert what == "score 0-0 -> 1-0"
    assert [i.source for i in tracker.state.incidents] == ["wire"]


def test_a_wire_goal_names_a_board_goal_instead_of_duplicating_it():
    tracker = MatchStateTracker.from_pack(wire_pack())
    tracker.state.incidents.append(
        Incident(event=Event.GOAL, side=Side.HOME, player=None, video_ts=100.0, source="board")
    )
    what = tracker.apply_wire(
        wire_event(
            event=Event.GOAL, side=Side.HOME, player="Alexis Mac Allister", home_score=1
        ),
        106.0,
    )
    assert len(tracker.state.incidents) == 1, "the same goal was counted twice"
    assert tracker.state.incidents[0].player == "Alexis Mac Allister"
    assert tracker.state.incidents[0].source == "board"
    assert what == "goal home is Alexis Mac Allister"


def test_a_goal_long_after_a_board_goal_is_a_second_goal():
    tracker = MatchStateTracker.from_pack(wire_pack())
    tracker.state.incidents.append(
        Incident(event=Event.GOAL, side=Side.HOME, player=None, video_ts=100.0, source="board")
    )
    tracker.apply_wire(
        wire_event(event=Event.GOAL, side=Side.HOME, player="Paulo Dybala", home_score=2),
        400.0,
    )
    assert len(tracker.state.incidents) == 2


def test_possession_reaches_the_recipient_when_the_ball_does():
    tracker = MatchStateTracker.from_pack(wire_pack())
    tracker.apply_wire(
        wire_event(
            event=Event.PASS,
            side=Side.HOME,
            player="Nicolás Otamendi",
            recipient="Alexis Mac Allister",
            duration_s=1.2,
        ),
        100.0,
    )
    assert "on the ball: Nicolás Otamendi (ARG)" in tracker.summary(100.5)
    later = tracker.summary(101.5)
    assert "on the ball: Alexis Mac Allister (ARG), from Nicolás Otamendi" in later


def test_the_ball_line_goes_away_once_it_is_stale():
    """A caller told who has the ball will say so, and a name six seconds old
    belongs to a passage of play that has already ended."""
    tracker = MatchStateTracker.from_pack(wire_pack())
    tracker.apply_wire(
        wire_event(event=Event.CARRY, side=Side.HOME, player="Alexis Mac Allister"), 100.0
    )
    assert "on the ball" in tracker.summary(103.0)
    assert "on the ball" not in tracker.summary(110.0)


def test_named_events_are_said_plainly_and_then_expire():
    tracker = MatchStateTracker.from_pack(wire_pack())
    tracker.apply_wire(
        wire_event(
            event=Event.FOUL, side=Side.AWAY, player="Adrien Rabiot", recipient="Lionel Messi"
        ),
        100.0,
    )
    tracker.apply_wire(
        wire_event(event=Event.SAVE, side=Side.HOME, player="Emiliano Martínez"), 104.0
    )

    summary = tracker.summary(105.0)
    assert "just now:" in summary
    assert "foul by Adrien Rabiot (FRA) on Lionel Messi" in summary
    assert "save Emiliano Martínez (ARG)" in summary
    assert "just now:" not in tracker.summary(200.0)


def test_a_substitution_teaches_the_registry_the_shirt():
    tracker = MatchStateTracker.from_pack(wire_pack())
    tracker.apply_wire(
        wire_event(
            event=Event.SUBSTITUTION,
            side=Side.HOME,
            player="Paulo Dybala",
            recipient="Alexis Mac Allister",
        ),
        100.0,
    )
    assert tracker.registry.identified(100.0)[Side.HOME] == [(21, "Paulo Dybala")]
