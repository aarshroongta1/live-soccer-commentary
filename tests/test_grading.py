"""The numbers have to be wrong in the right direction when the run is bad."""

from __future__ import annotations

from pathlib import Path

import pytest

from commentary.bus import Message, Topic
from commentary.grading import metrics, report
from commentary.schemas import Event, GroundTruthEvent, KnowledgePack, Player, Side, TeamSheet
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


def spoken(ts: float, text: str, live: float | None = None) -> tuple[Topic, float, dict]:
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
    path = write_trace(tmp_path, [spoken(150.0, "and that is a goal for Arsenal")])
    run = metrics.load_run(path)
    kinds = {e.kind for e in metrics.factual_errors(run, truth, pack)}
    assert "phantom_goal" in kinds


@pytest.mark.parametrize(
    "line",
    [
        "and that is a goal for Arsenal",
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
    path = write_trace(tmp_path, [spoken(150.0, line)])
    run = metrics.load_run(path)
    kinds = {e.kind for e in metrics.factual_errors(run, truth, pack)}
    assert "phantom_goal" in kinds, f"not detected as a goal claim: {line!r}"


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
