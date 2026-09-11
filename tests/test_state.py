from dataclasses import dataclass

import pytest

from commentary.schemas import (
    BoardRead,
    CallerLine,
    Event,
    KnowledgePack,
    Player,
    Scene,
    Side,
    TeamSheet,
)
from commentary.state import (
    EntityRegistry,
    MatchStateTracker,
    parse_clock,
    parse_sighting,
    period_for_clock,
)


@dataclass
class StubBoard:
    """Stands in for a confirmed BoardTracker, without importing perception."""

    home_score: int | None = None
    away_score: int | None = None
    clock: str | None = None
    in_replay: bool = False


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


def test_a_raw_board_read_without_a_bug_is_a_replay():
    tracker = MatchStateTracker("Arsenal", "Chelsea")
    tracker.apply_board(BoardRead(bug_visible=False, confidence=0.9))
    assert tracker.state.in_replay
    assert tracker.state.clock is None


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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("7 Saka", (7, "Saka")),
        ("#7 Saka", (7, "Saka")),
        ("7 Bukayo Saka", (7, "Bukayo Saka")),
        ("Saka (7)", (7, "Saka")),
        ("Saka #7", (7, "Saka")),
        ("Saka", None),
        ("7", None),
        ("", None),
    ],
)
def test_a_sighting_needs_both_a_number_and_a_name(text, expected):
    assert parse_sighting(text) == expected


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
        caller(scene=Scene.GRAPHIC, side=Side.HOME, names_read=["53 Ethan Nwaneri", "crowd"]),
        ts=4000.0,
    )

    assert tracker.registry.name_for(53, side=Side.HOME) == "Ethan Nwaneri"
    assert tracker.registry.confidence(53, 4000.0, side=Side.HOME) == pytest.approx(1.0)
    assert tracker.state.on_pitch["53"] == "Ethan Nwaneri"
