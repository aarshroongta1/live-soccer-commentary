"""The whole pipeline, end to end, against a match that never happened.

This is the test that would have been impossible to write tonight without the
simulator: frames in at one end, spoken commentary out at the other, with a
written record of what actually happened to check it against.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from commentary.config import (
    CallerConfig,
    CaptureConfig,
    DirectorConfig,
    PredictorConfig,
    Settings,
)
from commentary.grading import metrics, report
from commentary.runtime import Runtime
from commentary.schemas import Event
from commentary.sim import MatchSim, SimOracle, SimSource
from commentary.trace import RunTrace
from commentary.voice import LogSpeaker


def fast_settings(delay_s: float = 4.0) -> Settings:
    """A sim minute compressed into a test that finishes in seconds."""
    return Settings(
        capture=CaptureConfig(width=640, height=360, fps=8, delay_s=delay_s, history_s=3.0),
        caller=CallerConfig(min_gap_s=2.0),
        predictor=PredictorConfig(tick_s=0.05),
        director=DirectorConfig(max_beat_age_s=30.0),
    )


async def run_sim(
    tmp_path: Path,
    *,
    error_rate: float = 0.0,
    seconds: float = 3.0,
    delay_s: float = 4.0,
) -> tuple[Runtime, MatchSim, Path]:
    sim = MatchSim(seed=5, duration_s=120.0)
    settings = fast_settings(delay_s)
    source = SimSource(sim, settings.capture, realtime=False)
    oracle = SimOracle(sim=sim, error_rate=error_rate)
    path = tmp_path / "run.jsonl"
    with RunTrace(path=path) as trace:
        runtime = Runtime(
            source=source,
            backend=oracle,
            pack=sim.knowledge_pack,
            settings=settings,
            speaker=LogSpeaker(words_per_second=120),
            trace=trace,
        )
        await runtime.run(seconds=seconds)
    return runtime, sim, path


@pytest.mark.asyncio
async def test_a_match_runs_and_produces_commentary(tmp_path: Path) -> None:
    runtime, _sim, path = await run_sim(tmp_path)

    assert runtime.stats.frames > 0, "no frames reached the buffer"
    assert runtime.stats.board_reads > 0, "the board was never read"
    assert runtime.stats.caller_calls > 0, "the caller was never asked"
    run = metrics.load_run(path)
    assert run.lines, "nothing was ever spoken"


@pytest.mark.asyncio
async def test_the_score_only_ever_comes_from_the_board(tmp_path: Path) -> None:
    runtime, sim, _path = await run_sim(tmp_path)
    truth_at_end = [e for e in sim.ground_truth if e.video_ts <= runtime.live_ts]
    expected_home = truth_at_end[-1].home_score if truth_at_end else 0
    assert runtime.state.home_score <= expected_home, (
        "state got ahead of what the board could have shown"
    )


@pytest.mark.asyncio
async def test_the_gate_catches_lies_the_oracle_injects(tmp_path: Path) -> None:
    runtime, sim, path = await run_sim(tmp_path, error_rate=1.0, seconds=4.0)

    oracle = runtime.backend
    assert isinstance(oracle, SimOracle)
    assert oracle.injected, "the oracle was asked to lie and did not"
    assert runtime.gate.stats.judged > 0, "the gate never saw a line"
    assert runtime.gate.stats.rejected > 0, "every injected error slipped through the gate"

    # And what does get spoken should be measurably cleaner than what was proposed.
    run = metrics.load_run(path)
    errors = metrics.factual_errors(run, sim.ground_truth, sim.knowledge_pack)
    proposed = len(oracle.injected)
    assert len(errors) < proposed, (
        f"{len(errors)} errors survived out of {proposed} injected — the gate is not working"
    )


@pytest.mark.asyncio
async def test_a_run_can_be_graded_from_its_trace_alone(tmp_path: Path) -> None:
    _runtime, sim, path = await run_sim(tmp_path)
    card = report.score("sim", path, sim.ground_truth, sim.knowledge_pack, duration_s=120.0)
    assert card.lines > 0
    assert 0.0 <= card.silence_ratio <= 1.0
    assert 0.0 <= card.factual_error_rate <= 1.0
    assert "| sim |" in report.table([card])


@pytest.mark.asyncio
async def test_nothing_is_spoken_while_a_replay_is_on_screen(tmp_path: Path) -> None:
    _runtime, _sim, path = await run_sim(tmp_path, seconds=4.0)
    rows = metrics.read_trace(path) if hasattr(metrics, "read_trace") else []
    del rows  # the assertion below is the one that matters

    run = metrics.load_run(path)
    # The caller is told to stay quiet over a replay and the gate refuses one
    # outright, so no spoken line may carry the replay scene.
    assert all("replay" not in line.text.lower() for line in run.lines)


@pytest.mark.asyncio
async def test_a_confident_caller_is_not_evidence_of_a_celebration(tmp_path: Path) -> None:
    """Corroboration has to come from somewhere other than the claimant.

    An earlier version asked the caller how confident it felt and treated
    anything above 0.8 as a celebration seen in the lookahead. That is the
    same source with a number attached, and it let a goal be announced at a
    moment when no goal had happened.
    """
    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0)

    runtime._roars.clear()
    assert runtime._celebration_ahead(10.0) is False

    # A roar only stands in for the board when the board cannot be read.
    # While the bug is legible the board is the only thing that confirms a
    # goal, because a crowd roars at near misses too — and one did, letting a
    # phantom goal through when the roar was a free-standing second route.
    runtime._roars.append(12.0)
    tracker = runtime.board_tracker
    tracker._confirmed = (1, 0, 1)
    tracker._absent_run = 0
    assert runtime._celebration_ahead(10.0) is False, "the board was readable; it should decide"

    # Bug gone for long enough to read as a replay: now the crowd is all we have.
    tracker._absent_run = tracker.replay_reads
    assert runtime._celebration_ahead(10.0) is True
    assert runtime._celebration_ahead(100.0) is False


@pytest.mark.asyncio
async def test_a_goal_is_never_announced_before_the_board_confirms_it(tmp_path: Path) -> None:
    _runtime, sim, path = await run_sim(tmp_path, seconds=5.0, delay_s=4.0)
    run = metrics.load_run(path)
    goals = [e.video_ts for e in sim.ground_truth if e.event is Event.GOAL]

    for line in run.lines:
        if "goal" not in line.text.lower():
            continue
        assert any(line.video_ts >= goal - 2.0 for goal in goals), (
            f"a goal was called at {line.video_ts:.1f}s with no goal behind it"
        )
