from dataclasses import dataclass

import pytest

from commentary.schemas import (
    CallerLine,
    Event,
    Incident,
    KnowledgePack,
    Note,
    Player,
    Possession,
    Scene,
    Side,
    Sighting,
    TeamSheet,
    WireEvent,
)
from commentary.state import (
    NOTES_PER_CALL,
    EntityRegistry,
    MatchStateTracker,
    notes_for,
    parse_clock,
    period_for_clock,
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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("37:12", 2232.0),
        ("00:00", 0.0),
        ("90:00", 5400.0),
        ("45", 2700.0),
        ("45+2", 2820.0),
        ("45+2:13", 2833.0),
        ("90+3", 5580.0),
        ("67'", 4020.0),
        (" 12:05 ", 725.0),
    ],
)
def test_parse_clock_reads_the_forms_a_broadcast_actually_prints(text, expected):
    assert parse_clock(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    ["", "   ", "HT", "half time", "1-0", "12:99", "999:00", "45+99", "abc", "12:3a", None],
)
def test_parse_clock_returns_none_for_rubbish(text):
    assert parse_clock(text) is None


@pytest.mark.parametrize(
    ("text", "period"),
    [
        ("00:12", 1),
        ("44:59", 1),
        ("45+2", 1),
        ("45:00", 2),
        ("89:59", 2),
        ("90+3", 2),
        ("90:00", 3),
        ("105+1", 3),
        ("106:00", 4),
        ("nonsense", None),
    ],
)
def test_period_is_read_from_the_clock_including_the_plus(text, period):
    assert period_for_clock(text) == period


def test_the_board_moves_the_score():
    tracker = MatchStateTracker("Arsenal", "Chelsea")
    tracker.apply_board(StubBoard(home_score=1, away_score=0, clock="23:40"))

    assert tracker.state.scoreline == "Arsenal 1-0 Chelsea"
    assert tracker.state.clock_s == pytest.approx(1420.0)
    assert tracker.state.period == 1


def test_the_caller_cannot_move_the_score():
    tracker = MatchStateTracker("Arsenal", "Chelsea")
    tracker.apply_board(StubBoard(home_score=1, away_score=0, clock="23:40"))

    tracker.apply_caller(caller(event=Event.GOAL, side=Side.AWAY, line="It's in!"))

    assert (tracker.state.home_score, tracker.state.away_score) == (1, 0)
    assert tracker.state.last_events == [Event.GOAL]
    assert tracker.state.possession is Side.AWAY


def test_the_replay_flag_comes_from_the_board():
    tracker = MatchStateTracker("Arsenal", "Chelsea")
    tracker.apply_board(StubBoard(home_score=0, away_score=0, clock="10:00", in_replay=True))
    assert tracker.state.in_replay

    tracker.apply_caller(caller(scene=Scene.LIVE_PLAY))
    assert tracker.state.in_replay

    tracker.apply_board(StubBoard(home_score=0, away_score=0, clock="10:04"))
    assert not tracker.state.in_replay


def test_replays_do_not_add_events_or_possession():
    tracker = MatchStateTracker("Arsenal", "Chelsea")
    tracker.apply_caller(caller(scene=Scene.REPLAY, event=Event.GOAL, side=Side.HOME))
    assert tracker.state.last_events == []
    assert tracker.state.possession is Side.UNKNOWN


def test_recent_events_stay_bounded():
    tracker = MatchStateTracker("Arsenal", "Chelsea", max_events=3)
    for event in (Event.CORNER, Event.SHOT, Event.SAVE, Event.FOUL):
        tracker.apply_caller(caller(event=event))
    assert tracker.state.last_events == [Event.SHOT, Event.SAVE, Event.FOUL]


def test_summary_is_short_and_carries_the_state():
    tracker = MatchStateTracker("Arsenal", "Chelsea")
    tracker.apply_board(StubBoard(home_score=2, away_score=1, clock="67:14"))
    tracker.apply_caller(caller(event=Event.SAVE, side=Side.HOME))

    summary = tracker.summary()
    assert "Arsenal 2-1 Chelsea" in summary
    assert "67:14" in summary
    assert "second half" in summary
    assert "save" in summary
    assert len(summary.splitlines()) <= 5


def test_summary_says_when_the_screen_is_a_replay():
    tracker = MatchStateTracker("Arsenal", "Chelsea")
    tracker.apply_board(StubBoard(home_score=0, away_score=0, clock="5:00", in_replay=True))
    assert "replay" in tracker.summary()


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


# -- notes out of the pack ---------------------------------------------------
#
# The pack is frozen and full; a prompt is small and one moment wide. This is
# the narrowing: given the names on the screen, the two or three clauses worth
# putting in front of the voice, and nothing else.


def noted_pack() -> KnowledgePack:
    return pack().model_copy(
        update={
            "notes": [
                Note(about="Bukayo Saka", text="four goals in this competition", kind="stat"),
                Note(about="Bukayo Saka", text="scored in each of the last three", kind="stat"),
                Note(about="Cole Palmer", text="takes every Chelsea penalty", kind="habit"),
                Note(about="Arsenal", text="have not lost at home since April", kind="storyline"),
                Note(about="Chelsea", text="unbeaten in eleven away", kind="storyline"),
            ]
        }
    )


def test_notes_are_found_by_the_name_a_line_actually_says():
    """Surnames, because a commentator never says a first name."""
    picked = notes_for(noted_pack(), ["Saka"])
    assert [note.text for note in picked] == [
        "four goals in this competition",
        "scored in each of the last three",
    ]


def test_a_name_with_nothing_researched_about_it_yields_nothing():
    assert notes_for(noted_pack(), ["Havertz"]) == []
    assert notes_for(None, ["Saka"]) == []
    assert notes_for(pack(), ["Saka"]) == []


def test_the_same_player_named_twice_does_not_spend_two_places():
    picked = notes_for(noted_pack(), ["Saka", "Bukayo Saka"])
    assert len(picked) == 2


def test_the_tracker_puts_the_man_on_the_ball_first_and_the_teams_last():
    """Four places, and a wide shot names four players. The ball wins them."""
    tracker = MatchStateTracker.from_pack(noted_pack())
    tracker.state.ball = Possession(player="Bukayo Saka", side=Side.HOME, since_ts=0.0)
    picked = tracker.pack_notes_for(["Cole Palmer"])
    assert [note.about for note in picked] == [
        "Bukayo Saka",
        "Bukayo Saka",
        "Cole Palmer",
        "Arsenal",
    ]
    assert len(picked) == NOTES_PER_CALL


def test_both_teams_are_always_offered_when_there_is_room():
    tracker = MatchStateTracker.from_pack(noted_pack())
    picked = tracker.pack_notes_for([])
    assert [note.about for note in picked] == ["Arsenal", "Chelsea"]


def test_the_cap_holds_however_many_names_arrive():
    tracker = MatchStateTracker.from_pack(noted_pack())
    picked = tracker.pack_notes_for(["Saka", "Palmer"], limit=3)
    assert len(picked) == 3
