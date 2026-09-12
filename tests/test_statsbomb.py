"""StatsBomb's event data reduced to the feed the grader already reads."""

from __future__ import annotations

import json
from pathlib import Path

from commentary.grading import feed, statsbomb
from commentary.schemas import Event, Side

FIXTURE = Path(__file__).parent / "fixtures" / "statsbomb-events.json"
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


def test_an_own_goal_counts_for_the_other_side():
    own = converted()["events"][2]
    # Upamecano's, so the row is France's and the goal is Argentina's.
    assert own["team"] == "home"
    assert (own["home_score"], own["away_score"]) == (2, 0)


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
