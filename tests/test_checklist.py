"""The definition of done, decided by machine on a trace nobody read by hand.

Every fixture here is a file on disk and nothing reaches the network. The
trace is written the way the runtime writes one — board reads with a clock on
them, caller rows, gate rows, sightings, tracker passes — because the whole
point of the command is that it reads a real trace, and a test that feeds it
a convenient shape would not notice when the trace changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from commentary import statsbomb
from commentary.bus import Message, Topic
from commentary.grading import checklist, feed, metrics
from commentary.schemas import Event, KnowledgePack, Player, TeamSheet
from commentary.trace import RunTrace, read_trace, rows_of

FIXTURES = Path(__file__).parent / "fixtures"
#: The clip's own clock: video 0 is 60:00 played, so the offset is -3600.
OFFSET = -3600.0


@pytest.fixture
def pack() -> KnowledgePack:
    return KnowledgePack(
        home=TeamSheet(
            name="Argentina",
            short="ARG",
            demonym="Argentine",
            starters=[
                Player(name="Lionel Messi", number=10),
                Player(name="Rodrigo De Paul", number=7),
                Player(name="Ángel Di María", number=11),
            ],
        ),
        away=TeamSheet(
            name="France",
            short="FRA",
            demonym="French",
            starters=[
                Player(name="Adrien Rabiot", number=14),
                Player(name="Dayot Upamecano", number=18),
            ],
        ),
        competition="World Cup final",
    )


@pytest.fixture
def wire():
    return statsbomb.read(
        FIXTURES / "statsbomb-wire-events.json",
        FIXTURES / "statsbomb-lineups.json",
        "Argentina",
        "France",
    )


def board(ts: float) -> tuple[Topic, float, dict]:
    """A board read whose clock puts video ``ts`` at 60:00 played plus ts."""
    played = int(ts - OFFSET)
    return (
        Topic.BOARD,
        ts,
        {
            "bug_visible": True,
            "home_score": 1,
            "away_score": 0,
            "clock": f"{played // 60}:{played % 60:02d}",
            "confidence": 0.95,
        },
    )


def spoken(
    ts: float, text: str, *, voice: str = "caller", event: str = "none", seconds: float = 4.0
):
    return (
        Topic.SPOKEN,
        ts,
        {
            "voice": voice,
            "text": text,
            "spoken": text,
            "live_ts": ts + 8.0,
            "event": event,
            "seconds": seconds,
        },
    )


def caller(ts: float, scene: str = "live_play"):
    return (Topic.CALLER, ts, {"scene": scene, "event": "none", "speak": True})


def gate(ts: float, passed: bool = True, reasons: list[str] | None = None):
    return (Topic.GATE, ts, {"passed": passed, "reasons": reasons or [], "line": ""})


def tracks(ts: float, ids: list[int]):
    payload = {"tracks": len(ids), "with_side": 0, "named": 0, "ms": 100.0, "ids": ids}
    return (Topic.TRACKS, ts, payload)


def sighting(
    ts: float,
    mark: str | None,
    number: int | None,
    name: str | None,
    bound: bool,
    side: str = "",
):
    return (
        Topic.SIGHTING,
        ts,
        {
            "sightings": [
                {
                    "mark": mark,
                    "number": number,
                    "name": name,
                    "bound": bound,
                    "live": bool(mark),
                    "side": side,
                }
            ]
        },
    )


def write_trace(tmp_path: Path, rows) -> Path:
    path = tmp_path / "clip.jsonl"
    with RunTrace(path=path) as trace:
        for topic, ts, payload in rows:
            trace.write(Message(topic=topic, ts=ts, payload=payload))
    return path


def a_good_run() -> list[tuple[Topic, float, dict]]:
    """A trace that should pass every item the fixture feed can decide.

    The fixture feed has a throw-in at 60:00 (video 0), a penalty foul at
    62:30 (video 150) and nothing else in the window. Lines land every
    fifteen seconds, which is what item 9 is about.
    """
    filler = [
        "Argentina keep it, patient now, France dropping into a low block.",
        "Messi drops in between the lines and the Argentine end lifts.",
        "De Paul feeds it wide and Argentina work the angle again.",
        "Upamecano steps out and heads the cross away from the near post.",
    ]
    rows: list[tuple[Topic, float, dict]] = [board(ts) for ts in (2.0, 20.0, 60.0, 120.0, 170.0)]
    rows += [tracks(float(t) / 10.0, [1, 2, 3]) for t in range(0, 1800)]
    said = {
        2.0: "De Paul takes the throw quickly and Messi has it on the turn.",
        30.0: "Di María drives at Upamecano down the left, looking for the overlap.",
        150.0: "Rabiot brings down Di María and the referee points to the spot!",
    }
    for i, ts in enumerate(float(t) for t in range(2, 172, 15)):
        text = said.get(ts, filler[i % len(filler)])
        rows += [
            caller(ts),
            gate(ts),
            spoken(ts, text, event="foul" if ts == 150.0 else "none"),
        ]
    rows += [
        sighting(30.0, "B", 11, "Di María", True),
        sighting(90.0, "C", 10, "Messi", True),
        (Topic.COST, 170.0, {"total_usd": 0.88}),
    ]
    return rows


def graded(path: Path, pack: KnowledgePack, wire_events, *, watched_s: float = 180.0):
    rows = read_trace(path)
    run = metrics.load_run(path)
    alignment = feed.align(feed.from_wire(wire_events), rows_of(rows, "board"))
    stamped = feed.stamp(wire_events, alignment)
    truth = [
        e
        for e in alignment.events
        if e.event not in (Event.PASS, Event.CARRY) and 0.0 <= e.video_ts <= watched_s
    ]
    state = checklist.watched(rows, run, truth, stamped, pack, watched_s=watched_s)
    return alignment, {item.number: item for item in checklist.check(state)}


# -- alignment -----------------------------------------------------------


def test_the_clock_on_the_board_puts_the_feed_on_video_time(pack, wire, tmp_path: Path) -> None:
    path = write_trace(tmp_path, a_good_run())
    alignment, _ = graded(path, pack, wire)
    assert alignment.ok
    assert alignment.offset_for(2) == pytest.approx(OFFSET, abs=1.0)


def test_a_board_that_never_read_a_clock_refuses_to_grade(pack, wire, tmp_path: Path) -> None:
    """No readings, no bridge between the two clocks, and no honest number."""
    rows = [
        (Topic.BOARD, ts, {"bug_visible": False, "confidence": 0.9}) for ts in (2.0, 20.0, 60.0)
    ]
    path = write_trace(tmp_path, rows + a_good_run()[5:])
    alignment, _ = graded(path, pack, wire)
    assert not alignment.ok
    assert "FAILED" in alignment.summary()


# -- the twelve ----------------------------------------------------------


def test_a_run_that_does_the_job_passes_the_items_the_feed_can_decide(
    pack, wire, tmp_path: Path
) -> None:
    path = write_trace(tmp_path, a_good_run())
    _, items = graded(path, pack, wire)
    # No goal in this window, so item 1 is vacuous and says so.
    assert items[1].ok and "no goal" in items[1].evidence
    assert items[3].ok, items[3].evidence
    assert items[4].ok
    assert items[5].ok, items[5].evidence
    assert items[8].ok, items[8].evidence
    assert items[9].ok, items[9].evidence
    assert items[10].ok
    assert items[11].ok
    assert items[12].ok, items[12].evidence


def test_the_throw_in_and_the_penalty_are_what_recall_is_measured_against(
    pack, wire, tmp_path: Path
) -> None:
    path = write_trace(tmp_path, a_good_run())
    _, items = graded(path, pack, wire)
    assert "throw_in 1/1" in items[2].evidence
    assert "penalty 1/1" in items[2].evidence


def test_a_goal_called_and_named_on_time_passes_the_first_item(
    pack, wire, tmp_path: Path
) -> None:
    # Di María's goal is at 35:22 in the other fixture; here the feed's own
    # penalty foul stands in for a moment and the goal is the own goal at
    # 88:12, well outside. So the window is moved instead: watch from 88:00.
    rows = [
        (
            Topic.BOARD,
            ts,
            {
                "bug_visible": True,
                "home_score": 2,
                "away_score": 0,
                "clock": f"{int(ts + 5280) // 60}:{int(ts + 5280) % 60:02d}",
                "confidence": 0.95,
            },
        )
        for ts in (2.0, 40.0, 90.0)
    ]
    rows += [
        caller(14.0),
        gate(14.0),
        spoken(14.0, "Upamecano turns it into his own net and Argentina are two up!", event="goal"),
        (Topic.COST, 90.0, {"total_usd": 0.5}),
    ]
    path = write_trace(tmp_path, rows)
    _, items = graded(path, pack, wire, watched_s=100.0)
    # An own goal names nobody, so what item 1 needs is the call, on time.
    assert "no goal line" not in items[1].evidence
    assert items[3].ok, items[3].evidence


def test_a_goal_called_where_there_was_none_is_a_phantom(pack, wire, tmp_path: Path) -> None:
    rows = a_good_run()
    rows += [caller(100.0), gate(100.0), spoken(100.0, "Messi has scored!", event="goal")]
    path = write_trace(tmp_path, rows)
    _, items = graded(path, pack, wire)
    assert not items[3].ok
    assert "Messi has scored" in items[3].evidence


def test_an_unconfirmed_goal_rejection_is_reported_with_the_line_it_killed(
    pack, wire, tmp_path: Path
) -> None:
    """A line about a goal that really happened, refused. The gate writes no
    text on a rejection, so the caller row beside it is the only evidence."""
    # Watched from 88:00, so the fixture's own goal at 88:12 is in the window
    # and the rejected line twenty seconds later is about it.
    rows = [
        (
            Topic.BOARD,
            ts,
            {
                "bug_visible": True,
                "home_score": 2,
                "away_score": 0,
                "clock": f"{int(ts + 5280) // 60}:{int(ts + 5280) % 60:02d}",
                "confidence": 0.95,
            },
        )
        for ts in (2.0, 40.0, 90.0)
    ]
    rows += [
        (Topic.CALLER, 32.0, {"scene": "replay", "event": "goal", "line": "Messi scores!"}),
        gate(32.0, passed=False, reasons=["unconfirmed_goal: no board change"]),
        (Topic.COST, 90.0, {"total_usd": 0.5}),
    ]
    path = write_trace(tmp_path, rows)
    _, items = graded(path, pack, wire, watched_s=100.0)
    assert not items[4].ok
    assert "Messi scores!" in items[4].evidence


def test_a_name_nobody_in_the_match_is_called_fails_the_precision_item(
    pack, wire, tmp_path: Path
) -> None:
    rows = a_good_run()
    rows += [caller(100.0), gate(100.0), spoken(100.0, "Kowalczyk turns and shoots")]
    path = write_trace(tmp_path, rows)
    _, items = graded(path, pack, wire)
    assert not items[5].ok
    assert "Kowalczyk" in items[5].evidence


def test_a_sighting_binding_a_number_the_named_player_does_not_wear_is_caught(
    pack, wire, tmp_path: Path
) -> None:
    rows = a_good_run()
    rows.append(sighting(100.0, "D", 18, "Messi", True))
    path = write_trace(tmp_path, rows)
    _, items = graded(path, pack, wire)
    assert not items[8].ok
    assert "18 Messi" in items[8].evidence


def test_a_number_both_squads_wear_is_not_a_contradiction(pack, wire, tmp_path: Path) -> None:
    """Two elevens on a pitch, and which one it was is settled at bind time."""
    rows = a_good_run()
    rows.append(sighting(100.0, "D", 11, "Di María", True))
    path = write_trace(tmp_path, rows)
    _, items = graded(path, pack, wire)
    assert items[8].ok, items[8].evidence


def test_a_name_carried_from_a_mark_into_a_later_line_is_what_item_seven_wants(
    pack, wire, tmp_path: Path
) -> None:
    path = write_trace(tmp_path, a_good_run())
    _, items = graded(path, pack, wire)
    assert "carried on a mark" in items[7].evidence
    assert "mark B" in items[7].evidence or "mark C" in items[7].evidence


def test_half_a_minute_of_live_play_with_nobody_talking_fails(
    pack, wire, tmp_path: Path
) -> None:
    rows = [row for row in a_good_run() if not (row[0] is Topic.SPOKEN and 30.0 <= row[1] <= 120.0)]
    path = write_trace(tmp_path, rows)
    _, items = graded(path, pack, wire)
    assert not items[9].ok
    assert "live-play gaps" in items[9].evidence


def test_the_scoreline_in_words_is_still_the_scoreline(pack, wire, tmp_path: Path) -> None:
    rows = a_good_run()
    rows += [caller(100.0), gate(100.0), spoken(100.0, "Two nil, and Argentina are cruising")]
    path = write_trace(tmp_path, rows)
    _, items = graded(path, pack, wire)
    assert not items[11].ok


def test_a_run_that_cost_too_much_or_errored_fails_the_health_item(
    pack, wire, tmp_path: Path
) -> None:
    rows = a_good_run()
    rows.append((Topic.ERROR, 90.0, {"where": "caller", "detail": "APITimeoutError"}))
    rows.append((Topic.COST, 175.0, {"total_usd": 1.90}))
    path = write_trace(tmp_path, rows)
    _, items = graded(path, pack, wire)
    assert not items[12].ok
    assert "1 errors" in items[12].evidence and "$1.90" in items[12].evidence


def test_the_grader_and_the_gate_agree_on_how_long_a_goal_is_talked_about() -> None:
    """One number, two places, and a test rather than a comment holding them.

    A grader stricter than the gate scores phantom goals the gate was right to
    let through, which is the A19 bug measured instead of run.
    """
    from commentary import runtime

    assert metrics.GOAL_TALK_S == runtime.GOAL_TALK_CAP_S


# -- the command ---------------------------------------------------------


def test_the_command_refuses_to_grade_an_alignment_it_cannot_trust(
    pack, wire, tmp_path: Path, capsys
) -> None:
    from commentary.__main__ import main

    pack_path = tmp_path / "pack.json"
    pack_path.write_text(pack.model_dump_json(), encoding="utf-8")
    rows = [(Topic.BOARD, ts, {"bug_visible": False, "confidence": 0.9}) for ts in (2.0, 20.0)]
    path = write_trace(tmp_path, rows + a_good_run()[5:])
    code = main(
        [
            "grade",
            str(path),
            "--pack",
            str(pack_path),
            "--statsbomb",
            str(FIXTURES / "statsbomb-wire-events.json"),
            "--lineups",
            str(FIXTURES / "statsbomb-lineups.json"),
        ]
    )
    assert code == 1
    assert "refusing to grade" in capsys.readouterr().out


def test_the_command_prints_the_table_the_detail_and_the_twelve(
    pack, wire, tmp_path: Path, capsys
) -> None:
    from commentary.__main__ import main

    pack_path = tmp_path / "pack.json"
    pack_path.write_text(pack.model_dump_json(), encoding="utf-8")
    path = write_trace(tmp_path, a_good_run())
    code = main(
        [
            "grade",
            str(path),
            "--pack",
            str(pack_path),
            "--statsbomb",
            str(FIXTURES / "statsbomb-wire-events.json"),
            "--lineups",
            str(FIXTURES / "statsbomb-lineups.json"),
            "--watched",
            "180",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "alignment ok" in out
    assert "| run | lines |" in out
    assert "Definition of done:" in out
    assert out.count("\nPASS") + out.count("\nFAIL") == 12
    assert "== spoken" in out


def test_without_statsbomb_the_command_still_prints_what_a_trace_alone_gives(
    tmp_path: Path, capsys
) -> None:
    from commentary.__main__ import main

    path = write_trace(tmp_path, a_good_run())
    assert main(["grade", str(path)]) == 0
    assert "lag p50/p95" in capsys.readouterr().out


def test_a_feed_and_a_trace_together_are_still_two_files_and_no_network(tmp_path: Path) -> None:
    """The wall: everything the grader reads is a path somebody saved."""
    source = (Path(__file__).parent.parent / "src/commentary/grading/checklist.py").read_text()
    assert "http" not in source
    assert json.loads((FIXTURES / "statsbomb-lineups.json").read_text())


def test_the_penalty_build_up_is_not_a_phantom_penalty(pack, wire, tmp_path: Path) -> None:
    """Ninety seconds pass between the award and the kick, and they are covered.

    On the clip this came off, the referee pointed to the spot at 20:53 and
    Messi struck it at 22:24. Every line in between — the wall clearing, the
    keeper alone on his line, the ball sitting on the spot — was scored as a
    penalty that never happened, because the feed stamps the award as an
    instant and the broadcast does not.
    """
    rows = a_good_run()
    # The fixture's penalty award is at video 150; the kick would be later.
    rows += [
        caller(170.0),
        gate(170.0),
        spoken(170.0, "The area empties, the keeper alone on his line.", event="penalty"),
    ]
    path = write_trace(tmp_path, rows)
    _, items = graded(path, pack, wire, watched_s=200.0)
    assert items[3].ok, items[3].evidence


def test_a_penalty_claimed_before_anybody_won_one_is_still_a_phantom(
    pack, wire, tmp_path: Path
) -> None:
    rows = a_good_run()
    rows += [
        caller(20.0),
        gate(20.0),
        spoken(20.0, "The referee points to the spot!", event="penalty"),
    ]
    path = write_trace(tmp_path, rows)
    _, items = graded(path, pack, wire)
    assert not items[3].ok
    assert "penalty" in items[3].evidence


def test_a_clip_with_no_clock_on_screen_can_be_graded_from_a_measured_offset(
    pack, tmp_path: Path, capsys
) -> None:
    """A shootout shows a tally, not a clock, so there is nothing to fit.

    Without this the grader refuses and the one clip most worth grading — the
    one where every assumption about the score bug is wrong — cannot be
    graded at all.
    """
    from commentary.__main__ import main

    pack_path = tmp_path / "pack.json"
    pack_path.write_text(pack.model_dump_json(), encoding="utf-8")
    rows = [(Topic.BOARD, ts, {"bug_visible": False, "confidence": 0.9}) for ts in (2.0, 20.0)]
    path = write_trace(tmp_path, rows + a_good_run()[5:])
    code = main(
        [
            "grade",
            str(path),
            "--pack",
            str(pack_path),
            "--statsbomb",
            str(FIXTURES / "statsbomb-wire-events.json"),
            "--lineups",
            str(FIXTURES / "statsbomb-lineups.json"),
            "--offset",
            "-3600",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "alignment BY HAND: -3600.0s, nothing fitted, nothing checked" in out
    assert "Definition of done:" in out


def test_a_number_bound_to_the_kit_the_caller_did_not_read_it_off_is_caught(
    pack, wire, tmp_path: Path
) -> None:
    """Once the caller says which kit, the check narrows to that squad.

    France's 18 is Upamecano and Argentina's is nobody's; a sighting that
    says it read 18 off the striped kit is a read that cannot be right.
    """
    rows = a_good_run()
    rows.append(sighting(100.0, "D", 18, None, True, side="home"))
    path = write_trace(tmp_path, rows)
    _, items = graded(path, pack, wire)
    assert not items[8].ok
    assert "18" in items[8].evidence


def test_the_same_number_on_the_kit_that_wears_it_is_fine(pack, wire, tmp_path: Path) -> None:
    rows = a_good_run()
    rows.append(sighting(100.0, "D", 18, None, True, side="away"))
    path = write_trace(tmp_path, rows)
    _, items = graded(path, pack, wire)
    assert items[8].ok, items[8].evidence


def test_half_a_compound_surname_is_still_that_player(pack, wire, tmp_path: Path) -> None:
    """A caller reading KOLO MUANI off a shirt may write either half.

    Matching the last word alone scored a correct read of the 12 as a
    sighting bound to the wrong man.
    """
    rows = a_good_run()
    rows.append(sighting(100.0, "D", 11, "María", True, side="home"))
    path = write_trace(tmp_path, rows)
    _, items = graded(path, pack, wire)
    assert items[8].ok, items[8].evidence


def test_rejecting_a_goal_nobody_scored_is_the_gate_working(pack, wire, tmp_path: Path) -> None:
    """The offside clip invented a goal at 31:15 and the gate killed it.

    Item 4 is about lines the gate wrongly refused — a line about the real
    goal or its celebration. A rejection of a goal that never happened is the
    single thing the gate exists for, and counting it as a failure scores the
    system for succeeding.
    """
    rows = a_good_run()
    rows += [
        (Topic.CALLER, 100.0, {"scene": "live_play", "event": "goal", "line": "It is in!"}),
        gate(100.0, passed=False, reasons=["unconfirmed_goal: no board change"]),
    ]
    path = write_trace(tmp_path, rows)
    _, items = graded(path, pack, wire)
    assert items[4].ok
    assert "never happened" in items[4].evidence
