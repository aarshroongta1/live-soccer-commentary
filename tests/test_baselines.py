"""The ablations, checked for the two things that make them worth running.

They have to be comparable: every variant writes a trace of the same shape and
grades to a scorecard with the same fields, or the results table is comparing
different things. And they have to be different: taking a piece out of the
system has to show up as a number, or the piece was not doing anything.

The second of those is the point of the whole repository. If bypassing the
fact gate does not let more of the oracle's lies reach a microphone, then the
central claim of the project is false and this file is where that would be
found out.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import pytest

from commentary.config import (
    CallerConfig,
    CaptureConfig,
    DirectorConfig,
    PredictorConfig,
    Settings,
)
from commentary.gate import FactGate
from commentary.grading import metrics, report
from commentary.grading.baselines import (
    DELAY_DEPTHS,
    WORLDCUPVOICE_CADENCE_S,
    CallerOnlyDirector,
    OpenGate,
    delay_name,
    delay_of,
    delay_sweep,
    error_count,
    full,
    no_gate,
    render,
    run_suite,
    run_variant,
    standard_variants,
    sweep_table,
    trace_file,
    wire,
    worldcupvoice,
)
from commentary.runtime import Runtime
from commentary.schemas import (
    Beat,
    CallerLine,
    Event,
    MatchState,
    Scene,
    Side,
    Sighting,
    Voice,
)
from commentary.sim import MatchSim, SimOracle, SimSource
from commentary.trace import read_trace, rows_of
from commentary.voice import LogSpeaker

#: How far video time can jump between two ticks. The tick loop sleeps against
#: the wall while the match runs at ``speed`` times real time, and a caller
#: call blocks it for as long as the call takes, so no cadence lands on the
#: exact second.
TICK_SLOP_S = 1.0


def fast(delay_s: float = 4.0) -> Settings:
    """A match a test can afford.

    640 pixels is the floor on width, not a taste: below it the simulator's
    embedded timestamp does not fit in the picture, and the oracle refuses to
    answer about a frame whose moment it cannot read.
    """
    return Settings(
        capture=CaptureConfig(
            width=640,
            height=360,
            fps=8,
            delay_s=delay_s,
            # A shallow buffer keeps these fast, and the presentation
            # offset cannot reach back further than the buffer holds.
            # Nothing here watches the picture, so it only has to be legal.
            history_s=3.0,
            present_offset_s=3.0,
        ),
        caller=CallerConfig(min_gap_s=2.0),
        predictor=PredictorConfig(tick_s=0.02),
        director=DirectorConfig(max_beat_age_s=30.0),
    )


def caller_call_times(path: Path) -> list[float]:
    """Video times at which the caller was asked for a line."""
    return [float(row["ts"]) for row in rows_of(read_trace(path), "caller")]


def gaps(times: list[float]) -> list[float]:
    return [b - a for a, b in zip(times, times[1:], strict=False)]


# -- the switches, in isolation -----------------------------------------


def a_terrible_line() -> CallerLine:
    """A line a real gate rejects three times over: invented name, invented
    goal, invented scoreline."""
    return CallerLine(
        scene=Scene.LIVE_PLAY,
        event=Event.GOAL,
        side=Side.HOME,
        sightings=[Sighting(name="Ronan Velasquez")],
        confidence=0.5,
        speak=True,
        line="It is in! Ronan Velasquez makes it 4-3.",
    )


def test_the_open_gate_passes_what_the_real_gate_refuses() -> None:
    line, state = a_terrible_line(), MatchState(home="Ashcombe", away="Verity")

    assert not FactGate().judge(line, state, None).passed

    gate = OpenGate()
    verdict = gate.judge(line, state, None)
    assert verdict.passed
    assert verdict.line == line.line
    # Still recorded: the ablation must leave the same rows behind as a real
    # run, or the grader is reading two different kinds of trace.
    assert gate.stats.judged == 1
    assert gate.stats.passed == 1


def test_the_single_voice_director_refuses_the_second_voice() -> None:
    director = CallerOnlyDirector(cfg=DirectorConfig())
    caller_beat = Beat(
        id="b1", voice=Voice.CALLER, text="Away we go.", video_ts=1.0, created_ts=1.0
    )
    analyst_beat = Beat(
        id="b2", voice=Voice.ANALYST, text="Worth noting the shape.", video_ts=2.0, created_ts=2.0
    )

    assert director.submit(caller_beat) is True
    assert director.submit(analyst_beat) is False
    assert director.pending == 1


# -- the suite ----------------------------------------------------------


@pytest.mark.asyncio
async def test_every_variant_runs_and_grades_to_the_same_scorecard(tmp_path: Path) -> None:
    variants = standard_variants(fast())
    cards = await run_suite(
        variants,
        seed=5,
        duration_s=60.0,
        seconds=8.0,
        error_rate=0.3,
        out_dir=tmp_path,
    )

    assert [c.name for c in cards] == [v.name for v in variants]
    for card, variant in zip(cards, variants, strict=True):
        path = trace_file(tmp_path, variant.name)
        assert path.exists(), f"{variant.name} wrote no trace"
        rows = read_trace(path)
        assert rows_of(rows, "caller"), f"{variant.name} never called the caller"
        # Including the ablation that has no gate: a bypass that stopped
        # writing gate rows would not be gradeable by the same code.
        assert rows_of(rows, "gate"), f"{variant.name} wrote no gate rows"

        run = metrics.load_run(path)
        assert run.lines, f"{variant.name} never said anything"
        assert card.lines == len(run.lines)
        assert 0.0 <= card.factual_error_rate <= 1.0
        assert 0.0 <= card.event_recall <= 1.0
        assert 0.0 <= card.gate_rejection_rate <= 1.0
        assert 0.0 <= card.silence_ratio <= 1.0

    rendered = render(cards)
    for variant in variants:
        assert f"| {variant.name} |" in rendered
    assert rendered == render(cards), "rendering is not a pure function of the cards"


@pytest.mark.asyncio
async def test_bypassing_the_gate_lets_more_of_the_oracles_lies_reach_a_voice(
    tmp_path: Path,
) -> None:
    """The project's central claim, reduced to one comparison.

    The oracle is told to lie on every call, so the two runs are handed the
    same invented names, invented scorelines and phantom goals. The only
    difference between them is whether anything checks.
    """
    sim = MatchSim(seed=5, duration_s=90.0)
    settings = fast()

    guarded = await run_variant(full(settings), sim, seconds=10.0, out_dir=tmp_path, error_rate=1.0)
    open_gate = await run_variant(
        no_gate(settings), sim, seconds=10.0, out_dir=tmp_path, error_rate=1.0
    )

    guarded_run = metrics.load_run(trace_file(tmp_path, "full"))
    open_run = metrics.load_run(trace_file(tmp_path, "no-gate"))
    assert guarded_run.gate_rejections, "the gate rejected nothing, so nothing was tested"
    assert not open_run.gate_rejections, "the bypassed gate rejected something"

    assert error_count(open_gate) > error_count(guarded), (
        f"the fact gate bought nothing: {error_count(guarded)} factual errors reached a "
        f"voice with it and {error_count(open_gate)} without it"
    )


@pytest.mark.asyncio
async def test_worldcupvoice_calls_on_a_metronome_and_the_full_system_does_not(
    tmp_path: Path,
) -> None:
    """The baseline's timer against the speak predictor, as two gap patterns.

    Paced rather than flat out: the cadence is measured in video seconds and
    the tick that enforces it sleeps against the wall, so a match running a
    hundred times too fast cannot show a four second rhythm.
    """
    sim = MatchSim(seed=5, duration_s=36.0)
    settings = fast()

    await run_variant(worldcupvoice(settings), sim, seconds=14.0, out_dir=tmp_path, speed=6.0)
    await run_variant(full(settings), sim, seconds=14.0, out_dir=tmp_path, speed=6.0)

    metronome = gaps(caller_call_times(trace_file(tmp_path, "worldcupvoice")))
    predicted = gaps(caller_call_times(trace_file(tmp_path, "full")))
    assert len(metronome) >= 3, "too few calls to see a rhythm"
    assert len(predicted) >= 3, "too few calls to compare against"

    def on_cadence(gap: float) -> bool:
        return WORLDCUPVOICE_CADENCE_S <= gap <= WORLDCUPVOICE_CADENCE_S + TICK_SLOP_S

    assert all(on_cadence(gap) for gap in metronome), (
        f"worldcupvoice's calls are not on a {WORLDCUPVOICE_CADENCE_S:g}s timer: {metronome}"
    )
    assert not all(on_cadence(gap) for gap in predicted), (
        f"the full system is calling on the baseline's timer too: {predicted}"
    )
    assert statistics.pstdev(predicted) > statistics.pstdev(metronome), (
        f"the speak predictor is as regular as a metronome: {predicted}"
    )


@pytest.mark.asyncio
async def test_a_variant_is_graded_only_on_the_match_it_called(tmp_path: Path) -> None:
    """A run stopped early owes nothing for the football it never saw.

    The wall-clock deadline cuts this run off a long way into a long match, so
    most of the fixture is out of scope. Scoring against the whole script would
    count corners from minutes the run was never shown a frame of, and every
    variant's recall would be dragged down by the same large amount — which
    reads as a finding and is an artefact of how long the run was given.
    """
    sim = MatchSim(seed=5, duration_s=240.0)
    card = await run_variant(full(fast()), sim, seconds=4.0, out_dir=tmp_path, speed=8.0)

    assert 0.0 < card.watched_s < sim.duration_s, (
        f"the run should have been cut off partway: watched {card.watched_s:.1f}s "
        f"of {sim.duration_s:.0f}s"
    )
    in_window = [e for e in sim.ground_truth if e.video_ts <= card.watched_s]
    assert len(in_window) < len(sim.ground_truth), "the whole match fitted in the window"

    scored = sum(need for _got, need in card.recall_by_event.values())
    assert scored == len(in_window), (
        f"recall was scored against {scored} events but only {len(in_window)} "
        "happened inside the window the run called"
    )

    rendered = render([card])
    assert "simulator numbers" in rendered.lower()
    assert f"{card.watched_s:.0f}s" in rendered


@pytest.mark.asyncio
async def test_the_delay_sweep_produces_one_card_per_depth(tmp_path: Path) -> None:
    variants = delay_sweep(fast())
    cards = await run_suite(
        variants,
        seed=5,
        duration_s=40.0,
        seconds=6.0,
        error_rate=0.0,
        out_dir=tmp_path,
    )

    assert [c.name for c in cards] == [delay_name(d) for d in DELAY_DEPTHS]
    assert [delay_of(c.name) for c in cards] == list(DELAY_DEPTHS)

    table = sweep_table(cards)
    assert len(table.splitlines()) == len(DELAY_DEPTHS) + 2, table
    assert sweep_table([report.Scorecard(name="full")]) == ""

    # Lag is buffer depth plus generation time, so the sweep should read back
    # as the depths that produced it.
    for card, depth in zip(cards, DELAY_DEPTHS, strict=True):
        if card.lines:
            assert card.lag_p50 == pytest.approx(depth, abs=1.0)


@pytest.mark.asyncio
async def test_the_wire_row_runs_and_writes_what_it_changed(tmp_path: Path) -> None:
    """The ceiling row has to run and be gradeable beside the others."""
    settings = fast(delay_s=4.0)
    sim = MatchSim(seed=5, duration_s=120.0)

    card = await run_variant(
        wire(settings), sim, seconds=12.0, out_dir=tmp_path, error_rate=1.0, speed=6.0
    )

    rows = read_trace(trace_file(tmp_path, card.name))
    assert rows_of(rows, "correction"), "the wire changed nothing it thought worth tracing"
    assert rows_of(rows, "gate"), "the ceiling row has to be gradeable like the rest"


@pytest.mark.asyncio
async def test_only_the_wire_can_put_a_lied_about_score_right() -> None:
    """What the ceiling row is actually for, stated so it cannot be noise.

    The brief asked for this as a comparison of factual error counts between
    the wire row and the full system at ``error_rate=1.0``. At that rate the
    gate rejects nearly everything, so each run speaks one or two lines and
    the two error counts are zero or one — a coin toss, not a measurement.

    The property underneath it is not a coin toss. With the board reader
    lying on every read nothing ever gets three agreeing looks, so the score
    cannot move at all; the wire is the one other source allowed to move it,
    and it moves it to exactly the truth.
    """
    from commentary.wire import ReplayWire

    async def score_at_the_end(with_wire: bool) -> tuple[tuple[int, int], tuple[int, int]]:
        # Run to the end of the clip rather than to a wall-clock deadline: how
        # far a flat-out run gets in six seconds depends on the machine, and
        # this is a claim about the whole clip, not about the scheduler.
        sim = MatchSim(seed=5, duration_s=90.0)
        settings = fast(delay_s=4.0)
        runtime = Runtime(
            source=SimSource(sim, settings.capture, realtime=False),
            backend=SimOracle(sim=sim, error_rate=1.0),
            pack=sim.knowledge_pack,
            settings=settings,
            speaker=LogSpeaker(words_per_second=200),
            wire=ReplayWire.from_truth(sim.ground_truth, 2.0) if with_wire else None,
        )
        await runtime.run()
        seen = [e for e in sim.ground_truth if e.video_ts <= runtime.cursor_ts]
        truth = (seen[-1].home_score, seen[-1].away_score) if seen else (0, 0)
        return (runtime.state.home_score, runtime.state.away_score), truth

    blind, truth = await score_at_the_end(with_wire=False)
    assert blind == (0, 0), "a board that lies every time should move nothing"
    assert truth != (0, 0), "this fixture needs a goal in the watched window"

    told, truth_again = await score_at_the_end(with_wire=True)
    assert told == truth_again, f"the wire left the score at {told}, truth {truth_again}"
