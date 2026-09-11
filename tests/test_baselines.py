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
    worldcupvoice,
)
from commentary.schemas import Beat, CallerLine, Event, MatchState, Scene, Side, Voice
from commentary.sim import MatchSim
from commentary.trace import read_trace, rows_of

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
        capture=CaptureConfig(width=640, height=360, fps=8, delay_s=delay_s, history_s=3.0),
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
        names_read=["Ronan Velasquez"],
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
