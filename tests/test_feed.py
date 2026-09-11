"""A feed that loads is worth nothing if it lands on the wrong second."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from commentary.grading import feed as feedmod
from commentary.schemas import Event, Side

OURS: dict[str, Any] = {
    "home_team": "Arsenal",
    "away_team": "Real Madrid",
    "events": [
        {"clock": "12:30", "type": "corner", "team": "away"},
        {
            "clock": "37:12",
            "type": "goal",
            "team": "home",
            "player": "Bukayo Saka",
            "home_score": 1,
            "away_score": 0,
            "text": "Saka slots it in",
        },
        {"clock": "45+2", "type": "yellow card", "team": "away", "player": "Jude Bellingham"},
        {"clock": "61:00", "type": "Substitution", "team": "home"},
        {"clock": "70:00", "type": "goal kick", "team": "home"},
        {"clock": "77:00", "type": "VAR Review", "team": "home"},
    ],
}


def write(tmp_path: Path, document: Any) -> Path:
    path = tmp_path / "feed.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def reads(*pairs: tuple[float, str], visible: bool = True) -> list[dict[str, Any]]:
    """Board rows as the trace writes them: a video ``ts`` and a match clock."""
    return [{"ts": ts, "clock": clock, "bug_visible": visible} for ts, clock in pairs]


# -- reading the file --------------------------------------------------


def test_our_json_shape_loads(tmp_path: Path) -> None:
    loaded = feedmod.load_feed(write(tmp_path, OURS))
    assert [e.event for e in loaded.events] == [
        Event.CORNER,
        Event.GOAL,
        Event.CARD,
        Event.SUBSTITUTION,
    ]
    goal = loaded.events[1]
    assert goal.side is Side.HOME
    assert goal.player == "Bukayo Saka"
    assert (goal.home_score, goal.away_score) == (1, 0)
    assert goal.clock.seconds == pytest.approx(2232.0)


def test_a_goal_kick_is_not_a_goal_and_unknown_rows_are_counted(tmp_path: Path) -> None:
    loaded = feedmod.load_feed(write(tmp_path, OURS))
    assert all(e.event is not Event.GOAL or e.clock.seconds == 2232.0 for e in loaded.events)
    assert loaded.skipped == {"goal kick": 1, "var review": 1}


def test_first_half_stoppage_stays_in_the_first_half() -> None:
    stoppage = feedmod.parse_clock("45+2")
    second = feedmod.parse_clock("46:00")
    assert stoppage is not None and second is not None
    assert stoppage.seconds == 2820.0 and stoppage.period == 1
    assert second.period == 2


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("37", 2220.0),
        ("37'", 2220.0),
        ("37:12", 2232.0),
        ("90 + 4:30", 5670.0),
        ("45+2:13", 2833.0),
    ],
)
def test_the_clock_shapes_feeds_actually_print(text: str, seconds: float) -> None:
    clock = feedmod.parse_clock(text)
    assert clock is not None and clock.seconds == pytest.approx(seconds)


def test_a_row_may_name_the_team_instead_of_the_side(tmp_path: Path) -> None:
    loaded = feedmod.load_feed(
        write(
            tmp_path,
            {
                "home_team": "Arsenal",
                "away_team": "Real Madrid",
                "events": [{"clock": "37:12", "type": "goal", "team": "Arsenal"}],
            },
        )
    )
    assert loaded.events[0].side is Side.HOME


def test_scores_run_forward_when_the_file_does_not_carry_them(tmp_path: Path) -> None:
    document = {
        "events": [
            {"clock": "10:00", "type": "goal", "team": "home"},
            {"clock": "20:00", "type": "goal", "team": "away"},
            {"clock": "30:00", "type": "goal", "team": "home"},
        ]
    }
    loaded = feedmod.load_feed(write(tmp_path, document))
    assert [(e.home_score, e.away_score) for e in loaded.events] == [(1, 0), (1, 1), (2, 1)]


def test_a_file_that_is_not_a_feed_says_so(tmp_path: Path) -> None:
    with pytest.raises(feedmod.FeedError):
        feedmod.load_feed(write(tmp_path, {"nothing": "here"}))


# -- landing it on the video clock -------------------------------------


def test_alignment_recovers_a_known_offset() -> None:
    offset = 120.0
    board = reads(
        (600.0 + offset, "10:00"),
        (720.0 + offset, "12:00"),
        (900.0 + offset, "15:00"),
        (1200.0 + offset, "20:00"),
        (1500.0 + offset, "25:00"),
    )
    events = [
        feedmod.FeedEvent(clock=feedmod.Clock(2232.0, 1, "37:12"), event=Event.GOAL),
    ]
    aligned = feedmod.align(events, board)
    assert aligned.offset_s == pytest.approx(offset)
    assert aligned.events[0].video_ts == pytest.approx(2232.0 + offset)
    assert aligned.residual_s == pytest.approx(0.0)
    assert aligned.ok


def test_one_badly_misread_board_does_not_move_the_alignment() -> None:
    offset = 120.0
    board = reads(
        (600.0 + offset, "10:00"),
        (720.0 + offset, "12:00"),
        (900.0 + offset, "15:00"),
        (1200.0 + offset, "20:00"),
        (1500.0 + offset, "25:00"),
        (600.0 + offset, "18:00"),  # an 8 read as an 18
    )
    aligned = feedmod.align([], board)
    assert aligned.offset_s == pytest.approx(offset)
    assert aligned.residual_s == pytest.approx(0.0)
    assert aligned.worst_s == pytest.approx(480.0)
    assert "worst 480.00s" in aligned.summary()


def test_reads_that_disagree_with_each_other_produce_a_bad_residual() -> None:
    board = reads(
        (600.0, "10:00"),
        (630.0, "10:00"),
        (660.0, "10:00"),
        (690.0, "10:00"),
        (720.0, "10:00"),
        (750.0, "10:00"),
    )
    aligned = feedmod.align([], board)
    assert aligned.residual_s == pytest.approx(45.0)
    assert not aligned.ok
    assert "SUSPECT" in aligned.summary()


def test_each_half_gets_its_own_offset_across_the_interval() -> None:
    board = reads(
        (600.0, "10:00"),
        (900.0, "15:00"),
        (1200.0, "20:00"),
        (3600.0, "46:00"),
        (3900.0, "51:00"),
        (4200.0, "56:00"),
    )
    events = [
        feedmod.FeedEvent(clock=feedmod.Clock(600.0, 1, "10:00"), event=Event.SHOT),
        feedmod.FeedEvent(clock=feedmod.Clock(3600.0, 2, "60:00"), event=Event.GOAL),
    ]
    aligned = feedmod.align(events, board)
    assert aligned.offsets == {1: pytest.approx(0.0), 2: pytest.approx(840.0)}
    assert aligned.events[0].video_ts == pytest.approx(600.0)
    assert aligned.events[1].video_ts == pytest.approx(4440.0)
    assert aligned.ok


def test_replays_hide_the_bug_and_those_reads_are_counted_not_used() -> None:
    board = [
        *reads((600.0, "10:00"), (900.0, "15:00"), (1200.0, "20:00")),
        {"ts": 1000.0, "clock": None, "bug_visible": False},
        {"ts": 1100.0, "bug_visible": False},
    ]
    aligned = feedmod.align([], board)
    assert aligned.n == 3 and aligned.unusable == 2
    assert aligned.ok


def test_with_no_usable_reads_the_events_stay_on_match_time_and_say_so() -> None:
    events = [feedmod.FeedEvent(clock=feedmod.Clock(600.0, 1, "10:00"), event=Event.SHOT)]
    aligned = feedmod.align(events, [{"ts": 5.0, "bug_visible": False}])
    assert aligned.events[0].video_ts == pytest.approx(600.0)
    assert not aligned.ok
    assert "FAILED" in aligned.summary()


def test_shift_restamps_by_hand() -> None:
    events = [feedmod.FeedEvent(clock=feedmod.Clock(600.0, 1, "10:00"), event=Event.SHOT)]
    assert feedmod.shift(events, 42.0)[0].video_ts == pytest.approx(642.0)
