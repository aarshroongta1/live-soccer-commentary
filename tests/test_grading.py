"""The numbers have to be wrong in the right direction when the run is bad."""

from __future__ import annotations

from pathlib import Path

import pytest

from commentary.bus import Message, Topic
from commentary.grading import metrics, report
from commentary.schemas import (
    Event,
    GroundTruthEvent,
    KnowledgePack,
    Player,
    Side,
    TeamSheet,
    WireEvent,
)
from commentary.trace import RunTrace, read_trace


@pytest.fixture
def pack() -> KnowledgePack:
    return KnowledgePack(
        home=TeamSheet(
            name="Arsenal",
            short="ARS",
            starters=[
                Player(name="Bukayo Saka", number=7),
                Player(name="Martín Ødegaard", number=8),
            ],
        ),
        away=TeamSheet(
            name="Real Madrid",
            short="RMA",
            starters=[Player(name="Jude Bellingham", number=5)],
        ),
    )


@pytest.fixture
def truth() -> list[GroundTruthEvent]:
    return [
        GroundTruthEvent(video_ts=30.0, event=Event.SHOT, side=Side.HOME),
        GroundTruthEvent(video_ts=60.0, event=Event.GOAL, side=Side.HOME, home_score=1),
        GroundTruthEvent(video_ts=120.0, event=Event.CORNER, side=Side.AWAY, home_score=1),
    ]


def write_trace(tmp_path: Path, rows: list[tuple[Topic, float, dict[str, object]]]) -> Path:
    path = tmp_path / "run.jsonl"
    with RunTrace(path=path) as trace:
        for topic, ts, payload in rows:
            trace.write(Message(topic=topic, ts=ts, payload=payload))
    return path


def spoken(
    ts: float, text: str, live: float | None = None, event: str = "none"
) -> tuple[Topic, float, dict]:
    """One spoken line. ``live`` is where the live edge had reached by then."""
    return (
        Topic.SPOKEN,
        ts,
        {
            "voice": "caller",
            "text": text,
            "spoken": text,
            "created_ts": ts,
            "live_ts": live if live is not None else ts + 8.4,
            "event": event,
        },
    )


def test_a_run_that_says_nothing_recalls_nothing(tmp_path: Path, truth, pack) -> None:
    path = write_trace(tmp_path, [])
    card = report.score("silent", path, truth, pack, duration_s=180.0)
    assert card.lines == 0
    assert card.event_recall == 0.0
    assert card.silence_ratio == 1.0


def test_lines_near_events_count_as_recall(tmp_path: Path, truth, pack) -> None:
    path = write_trace(
        tmp_path,
        [
            spoken(31.0, "Saka drives at the defence"),
            spoken(62.0, "Arsenal are in front"),
            spoken(121.0, "corner to Real Madrid"),
        ],
    )
    card = report.score("covered", path, truth, pack, duration_s=180.0)
    assert card.event_recall == pytest.approx(1.0)


def test_an_invented_name_is_counted_as_a_factual_error(tmp_path: Path, truth, pack) -> None:
    path = write_trace(tmp_path, [spoken(31.0, "Kowalczyk turns and shoots")])
    run = metrics.load_run(path)
    errors = metrics.factual_errors(run, truth, pack)
    assert [e.kind for e in errors] == ["name_off_roster"]
    assert errors[0].detail == "Kowalczyk"


def test_a_roster_name_and_a_team_name_are_not_errors(tmp_path: Path, truth, pack) -> None:
    path = write_trace(
        tmp_path,
        [spoken(31.0, "Saka feeds Ødegaard as Arsenal push on")],
    )
    run = metrics.load_run(path)
    assert metrics.factual_errors(run, truth, pack) == []


def test_an_ordinary_sentence_opener_is_not_mistaken_for_a_name(
    tmp_path: Path, truth, pack
) -> None:
    path = write_trace(tmp_path, [spoken(31.0, "Brilliant save by the keeper")])
    run = metrics.load_run(path)
    assert metrics.factual_errors(run, truth, pack) == []


def test_a_goal_claimed_where_none_happened_is_caught(tmp_path: Path, truth, pack) -> None:
    # A caller calling a goal marks the form as one, which is what the trace
    # carries; the bare word "goal" in a line is not itself a claim. Well past
    # the goal-talk window, so this is a claim about nothing rather than a
    # late line about the goal on 60.
    path = write_trace(tmp_path, [spoken(400.0, "and that is a goal for Arsenal", event="goal")])
    run = metrics.load_run(path)
    kinds = {e.kind for e in metrics.factual_errors(run, truth, pack)}
    assert "phantom_goal" in kinds


@pytest.mark.parametrize(
    "line",
    [
        "It is in! Saka has scored",
        "Saka finds the net",
        "Saka makes it two",
    ],
)
def test_a_phantom_goal_is_caught_however_it_is_phrased(
    tmp_path: Path, truth, pack, line: str
) -> None:
    """A checker that knows three phrasings is measuring its own vocabulary.

    The original list was "goal", "scores", "it's in", which missed "It is
    in! ... has scored" — the exact phrasing in use — so phantom goals were
    counted nowhere and every error rate was understated.
    """
    path = write_trace(tmp_path, [spoken(400.0, line)])
    run = metrics.load_run(path)
    kinds = {e.kind for e in metrics.factual_errors(run, truth, pack)}
    assert "phantom_goal" in kinds, f"not detected as a goal claim: {line!r}"


def test_the_celebration_is_not_a_phantom_goal(tmp_path: Path, truth, pack) -> None:
    """A line about the goal ninety seconds later is the broadcast, not a lie.

    The gate learned this in A19: a goal is talked about until play restarts,
    which is a minute or more of celebration, replays and the scorer's face.
    A grader with a twelve-second window scored every one of those lines as a
    phantom goal, which is the gate's old bug measured rather than run.
    """
    path = write_trace(
        tmp_path,
        [spoken(150.0, "Saka wheels away, and Arsenal have scored the goal that decides it")],
    )
    run = metrics.load_run(path)
    assert [e.kind for e in metrics.factual_errors(run, truth, pack)] == []


def test_a_goal_claim_before_the_goal_is_still_a_phantom(tmp_path: Path, truth, pack) -> None:
    """The window is asymmetric: ahead of the event is where inventing lives."""
    path = write_trace(tmp_path, [spoken(20.0, "Saka has scored", event="goal")])
    run = metrics.load_run(path)
    assert "phantom_goal" in {e.kind for e in metrics.factual_errors(run, truth, pack)}


def test_a_wrong_scoreline_is_caught(tmp_path: Path, truth, pack) -> None:
    path = write_trace(tmp_path, [spoken(90.0, "Arsenal lead 3-0 here")])
    run = metrics.load_run(path)
    errors = [e for e in metrics.factual_errors(run, truth, pack) if e.kind == "wrong_score"]
    assert errors and "was 1-0" in errors[0].detail


def test_lag_is_measured_between_the_pitch_and_the_line(tmp_path: Path, truth, pack) -> None:
    path = write_trace(
        tmp_path,
        [spoken(30.0, "one", live=38.0), spoken(60.0, "two", live=69.0)],
    )
    run = metrics.load_run(path)
    assert metrics.lag(run).p50 == pytest.approx(8.5)


def test_a_trace_with_no_live_edge_reports_no_lag_rather_than_zero(tmp_path: Path) -> None:
    path = write_trace(
        tmp_path,
        [(Topic.SPOKEN, 30.0, {"voice": "caller", "text": "old trace", "spoken": "old trace"})],
    )
    measured = metrics.lag(metrics.load_run(path))
    assert measured.n == 0 and measured.p50 == 0.0


def test_near_duplicate_lines_count_as_repetition(tmp_path: Path) -> None:
    path = write_trace(
        tmp_path,
        [
            spoken(10.0, "Arsenal push forward down the left"),
            spoken(20.0, "Arsenal pushing forward on the left"),
        ],
    )
    run = metrics.load_run(path)
    assert metrics.repetition_rate(run) > 0.0


def test_gate_rejections_are_tallied_by_reason_tag(tmp_path: Path) -> None:
    path = write_trace(
        tmp_path,
        [
            (Topic.GATE, 10.0, {"passed": False, "reasons": ["name_not_on_roster: Kowalczyk"]}),
            (Topic.GATE, 20.0, {"passed": False, "reasons": ["unconfirmed_goal: no board change"]}),
            (Topic.GATE, 30.0, {"passed": False, "reasons": ["name_not_on_roster: Smyth"]}),
            (Topic.GATE, 40.0, {"passed": True, "reasons": []}),
        ],
    )
    run = metrics.load_run(path)
    assert metrics.gate_table(run) == {"name_not_on_roster": 2, "unconfirmed_goal": 1}
    assert metrics.gate_rejection_rate(run) == pytest.approx(0.75)


def test_a_truncated_final_line_does_not_break_reading(tmp_path: Path) -> None:
    path = tmp_path / "killed.jsonl"
    path.write_text('{"topic": "spoken", "ts": 1.0, "text": "fine"}\n{"topic": "spok', "utf-8")
    assert len(read_trace(path)) == 1


def test_the_table_renders_every_card(tmp_path: Path, truth, pack) -> None:
    path = write_trace(tmp_path, [spoken(31.0, "Saka drives forward")])
    cards = [report.score(name, path, truth, pack, duration_s=180.0) for name in ("a", "b")]
    rendered = report.table(cards)
    assert rendered.count("\n") == 3
    assert "| a |" in rendered and "| b |" in rendered


def test_the_grader_does_not_count_half_a_compound_surname_as_invented():
    """"Di María" arrives as two capitalised words, and "Di" is on no sheet.

    Before the name parts went into the roster, every correct use of a
    compound surname cost the run a fabricated ``name_off_roster``.
    """
    from commentary.grading.metrics import Run, SpokenLine, factual_errors, roster_names

    pack = KnowledgePack(
        home=TeamSheet(name="Argentina", starters=[Player(name="Ángel Di María", number=11)]),
        away=TeamSheet(name="France"),
    )
    assert {"di", "maria", "angel"} <= roster_names(pack)

    line = SpokenLine(video_ts=10.0, voice="caller", text="Di María cuts in")
    run = Run(run_id="x", lines=[line])
    assert factual_errors(run, [], pack) == []


def test_the_grader_reads_the_event_field_rather_than_the_bare_word_goal():
    from commentary.grading.metrics import Run, SpokenLine, factual_errors

    pack = KnowledgePack(home=TeamSheet(name="Argentina"), away=TeamSheet(name="France"))
    lines = [
        SpokenLine(
            video_ts=50.0,
            voice="caller",
            text="France scrambling back towards their own goal",
            event="build_up",
        ),
        SpokenLine(video_ts=200.0, voice="caller", text="And it is worked wide", event="goal"),
    ]
    errors = factual_errors(Run(run_id="x", lines=lines), [], pack)
    assert [e.kind for e in errors] == ["phantom_goal"]
    assert errors[0].video_ts == 200.0


# -- does it name players, and is it right -----------------------------------


def wire_event(ts: float, **fields) -> WireEvent:
    return WireEvent.model_validate({"clock_s": ts, "period": 1, "video_ts": ts, **fields})


def test_names_scores_each_name_against_what_the_feed_says_happened():
    from commentary.grading.metrics import Run, SpokenLine, names

    feed = [
        wire_event(
            50.0,
            event=Event.PASS,
            side=Side.HOME,
            player="Rodrigo De Paul",
            recipient="Ángel Di María",
        ),
        wire_event(90.0, event=Event.SHOT, side=Side.HOME, player="Lionel Messi"),
    ]
    lines = [
        # Right: Di María is the recipient of the pass a second earlier.
        SpokenLine(video_ts=51.0, voice="caller", text="Di María takes it down the left"),
        # Wrong: Messi is on the pitch but is doing nothing here.
        SpokenLine(video_ts=52.0, voice="caller", text="Messi drops in to collect"),
        # No name at all.
        SpokenLine(video_ts=53.0, voice="caller", text="Worked patiently across the back"),
    ]
    scored = names(Run(run_id="x", lines=lines), feed)

    assert (scored.total, scored.lines_with_name, scored.names) == (3, 2, 2)
    assert scored.correct == 1
    assert scored.rate == pytest.approx(2 / 3)
    assert scored.precision == pytest.approx(0.5)


def test_a_name_far_from_what_the_player_did_is_not_credited():
    """A player who touched the ball twenty seconds ago is not this moment."""
    from commentary.grading.metrics import Run, SpokenLine, names

    feed = [wire_event(50.0, event=Event.SHOT, side=Side.HOME, player="Lionel Messi")]
    lines = [SpokenLine(video_ts=75.0, voice="caller", text="Messi picks it up again")]
    scored = names(Run(run_id="x", lines=lines), feed)
    assert (scored.names, scored.correct) == (1, 0)


def test_without_a_feed_the_name_columns_are_zero_rather_than_guessed():
    from commentary.grading.metrics import Run, SpokenLine, names

    lines = [SpokenLine(video_ts=10.0, voice="caller", text="Messi turns")]
    scored = names(Run(run_id="x", lines=lines), [])
    assert (scored.names, scored.lines_with_name, scored.rate) == (0, 0, 0.0)


def test_where_the_match_is_being_played_is_not_an_invented_name():
    """A commentator says where they are, and the gate has always allowed it.

    The grader used to report "Park" in "away at Ashcombe Park" as a name the
    system invented, so the results table counted errors the gate had
    deliberately let through.
    """
    from commentary.grading.metrics import Run, SpokenLine, factual_errors

    pack = KnowledgePack(
        home=TeamSheet(name="Ashcombe Rangers", short="ASH", demonym="Ashcombe"),
        away=TeamSheet(name="Verity Athletic", short="VER"),
        competition="Coastal Cup",
        venue="Ashcombe Park",
    )
    line = SpokenLine(video_ts=1.0, voice="caller", text="We are away at Ashcombe Park")
    assert factual_errors(Run(run_id="x", lines=[line]), [], pack) == []


@pytest.mark.parametrize("text", ["Red shirts swarm the ball", "Whoever wins tops the group"])
def test_an_opener_the_caller_was_told_to_use_is_not_an_invented_name(text: str):
    from commentary.grading.metrics import Run, SpokenLine, factual_errors

    pack = KnowledgePack(home=TeamSheet(name="Argentina"), away=TeamSheet(name="France"))
    line = SpokenLine(video_ts=1.0, voice="caller", text=text)
    assert factual_errors(Run(run_id="x", lines=[line]), [], pack) == []
