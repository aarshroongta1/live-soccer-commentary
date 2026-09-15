"""StatsBomb's event data, read for the wire and reduced for the grader."""

from __future__ import annotations

import json
from pathlib import Path

from commentary import statsbomb as reader
from commentary.grading import feed, statsbomb
from commentary.schemas import Event, Side, WireEvent

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "statsbomb-events.json"
WIRE_FIXTURE = FIXTURES / "statsbomb-wire-events.json"
LINEUPS = FIXTURES / "statsbomb-lineups.json"
HOME, AWAY = "Argentina", "France"


def converted() -> dict:
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return statsbomb.convert(rows, HOME, AWAY)


def test_only_the_events_a_commentator_is_graded_on_survive():
    events = converted()["events"]
    assert [e["type"] for e in events] == [
        "goal",
        "yellow card",
        "own goal",
        "substitution",
        "red card",
    ]


def test_a_goal_carries_its_clock_and_its_scorer():
    goal = converted()["events"][0]
    assert goal["clock"] == "35:22"
    assert goal["team"] == "home"
    assert goal["player"] == "Ángel Di María"


def test_an_own_goal_counts_for_the_other_side_and_names_nobody():
    own = converted()["events"][2]
    # Upamecano's, so the row is France's and the goal is Argentina's.
    assert own["team"] == "home"
    assert (own["home_score"], own["away_score"]) == (2, 0)
    # And no scorer. The only name available is an opposition player, and
    # writing it into a row credited to the other side would have the grader
    # reading a France defender as Argentina's scorer.
    assert own["player"] is None


def test_the_score_runs_forward():
    events = converted()["events"]
    assert [(e["home_score"], e["away_score"]) for e in events] == [
        (1, 0),
        (1, 0),
        (2, 0),
        (2, 0),
        (2, 0),
    ]


def test_a_substitution_names_the_player_coming_on():
    sub = converted()["events"][3]
    assert sub["player"] == "Marcus Thuram"
    assert sub["team"] == "away"


def test_a_second_yellow_is_written_as_the_sending_off_it_is():
    assert converted()["events"][4]["type"] == "red card"


def test_a_row_naming_neither_team_is_dropped():
    assert all(e["team"] in ("home", "away") for e in converted()["events"])


def test_the_result_is_a_feed_the_grader_reads(tmp_path: Path):
    path = tmp_path / "feed.json"
    path.write_text(json.dumps(converted()), encoding="utf-8")
    loaded = feed.load_feed(path)
    assert loaded.skipped == {}
    assert [e.event for e in loaded.events] == [
        Event.GOAL,
        Event.CARD,
        Event.GOAL,
        Event.SUBSTITUTION,
        Event.CARD,
    ]
    assert loaded.events[0].side is Side.HOME
    assert loaded.events[0].clock.seconds == 35 * 60 + 22


# -- the wire's reading of the same files ------------------------------


def wired() -> list[WireEvent]:
    return reader.read(WIRE_FIXTURE, LINEUPS, HOME, AWAY)


def only(event: Event) -> list[WireEvent]:
    return [e for e in wired() if e.event is event]


def test_the_wire_carries_every_row_that_names_a_player():
    assert [(e.event, e.clock_s) for e in wired()] == [
        (Event.PASS, 31.0),
        (Event.CARRY, 34.0),
        (Event.OFFSIDE, 70.0),
        (Event.FOUL, 722.0),
        (Event.CARD, 722.0),
        (Event.SHOT, 1215.0),
        (Event.SAVE, 1216.0),
        (Event.GOAL, 2122.0),
        (Event.INTERCEPTION, 2465.0),
        (Event.CLEARANCE, 2690.0),
        (Event.THROW_IN, 3600.0),
        (Event.FOUL, 3750.0),
        (Event.PENALTY, 3750.0),
        (Event.SUBSTITUTION, 4260.0),
        (Event.TACKLE, 5100.0),
        (Event.GOAL, 5292.0),
    ]


def test_a_player_is_called_what_the_lineups_call_him():
    passed = only(Event.PASS)[0]
    assert (passed.player, passed.recipient) == ("Lionel Messi", "Rodrigo De Paul")
    # The shot row names no id, so the nickname is found by the registered name.
    assert only(Event.GOAL)[0].player == "Ángel Di María"


def test_an_own_goal_is_credited_to_the_other_side():
    own = only(Event.GOAL)[1]
    # Upamecano's, so the row is France's and the goal is Argentina's, and
    # there is nobody on Argentina's side to name for it.
    assert (own.side, own.player, own.detail) == (Side.HOME, None, "own goal")
    assert (own.home_score, own.away_score) == (2, 0)


def test_a_pass_that_reaches_nobody_is_dropped():
    assert [e.clock_s for e in only(Event.PASS)] == [31.0]


def test_a_foul_names_the_fouled_player_and_emits_the_card_beside_it():
    foul, card = only(Event.FOUL)[0], only(Event.CARD)[0]
    assert (foul.player, foul.recipient, foul.detail) == ("Adrien Rabiot", "Lionel Messi", "yellow")
    assert (card.player, card.detail, card.clock_s) == ("Adrien Rabiot", "yellow", foul.clock_s)


def test_an_offside_pass_names_the_player_who_was_offside():
    offside = only(Event.OFFSIDE)[0]
    assert (offside.player, offside.recipient) == ("Antoine Griezmann", "Olivier Giroud")


def test_a_substitution_carries_the_player_on_and_the_player_off():
    sub = only(Event.SUBSTITUTION)[0]
    assert (sub.player, sub.recipient, sub.period) == ("Marcus Thuram", "Olivier Giroud", 2)


def test_a_pass_carries_how_long_it_took():
    assert only(Event.PASS)[0].duration_s == 1.2


def test_a_restart_is_the_restart_and_not_the_pass_it_is_made_of():
    """A throw-in is what a commentator calls, and what the grader counts.

    StatsBomb writes every restart as a Pass with a type, so without this the
    truth for a three-minute clip has no throw-in, no corner and no free kick
    in it — and the brief grades the system on saying exactly those.
    """
    throw = only(Event.THROW_IN)[0]
    assert (throw.player, throw.recipient) == ("Rodrigo De Paul", "Lionel Messi")
    assert throw.clock_s not in [e.clock_s for e in only(Event.PASS)]


def test_a_penalty_award_is_its_own_event_on_the_side_that_won_it():
    """The award and the kick are a minute apart and are both commentated.

    The kick arrives as a Shot row — a goal or a save — so without the award
    there is nothing in the truth at the moment the referee points to the
    spot, which is the moment the commentary is about.
    """
    penalty = only(Event.PENALTY)[0]
    assert (penalty.side, penalty.player, penalty.recipient) == (
        Side.HOME,
        "Ángel Di María",
        "Adrien Rabiot",
    )
    foul = [e for e in only(Event.FOUL) if e.clock_s == penalty.clock_s][0]
    assert (foul.side, foul.detail) == (Side.AWAY, "penalty")


def test_a_flagged_cross_or_switch_is_the_event_and_not_a_plain_pass():
    """StatsBomb marks these on the pass row itself: ``pass.cross``/``pass.switch``.

    Gap 8 item 1: the corpus study measured both straight off this flag, and
    without reading it a cross or a switch of play is indistinguishable from
    any other completed pass.
    """
    rows = [
        {
            "id": "x-cross",
            "minute": 10,
            "second": 0,
            "period": 1,
            "team": {"name": HOME},
            "player": {"id": 1, "name": "Lionel Messi"},
            "type": {"name": "Pass"},
            "pass": {"recipient": {"id": 2, "name": "Ángel Di María"}, "cross": True},
        },
        {
            "id": "x-switch",
            "minute": 20,
            "second": 0,
            "period": 1,
            "team": {"name": HOME},
            "player": {"id": 3, "name": "Rodrigo De Paul"},
            "type": {"name": "Pass"},
            "pass": {"recipient": {"id": 1, "name": "Lionel Messi"}, "switch": True},
        },
    ]
    wire = reader.events(rows, HOME, AWAY, {})
    cross = [e for e in wire if e.event is Event.CROSS][0]
    switch = [e for e in wire if e.event is Event.SWITCH][0]
    assert (cross.player, cross.recipient) == ("Lionel Messi", "Ángel Di María")
    assert (switch.player, switch.recipient) == ("Rodrigo De Paul", "Lionel Messi")
    assert not only_of(wire, Event.PASS)


def only_of(wire: list[WireEvent], event: Event) -> list[WireEvent]:
    return [e for e in wire if e.event is event]
