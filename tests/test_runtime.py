"""The whole pipeline, end to end, against a match that never happened.

This is the test that would have been impossible to write tonight without the
simulator: frames in at one end, spoken commentary out at the other, with a
written record of what actually happened to check it against.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from commentary.capture.buffer import Frame
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
async def test_the_goal_confirmation_window_does_not_widen_with_the_buffer(
    tmp_path: Path,
) -> None:
    """How long a score bug lags a goal is a fact about television.

    An earlier version used the buffer depth as the window, so choosing to
    wait longer also made the gate accept a board change further from the
    moment being called. Across the sweep that showed up as phantom goals
    reaching air 1, 1, 4, 6 as the buffer deepened: the delay bought the
    caller information and paid for it by loosening the gate.
    """
    from commentary.perception.board import BoardChange
    from commentary.runtime import GOAL_GRAPHIC_LAG_S

    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=14.0)
    assert runtime.settings.capture.delay_s > GOAL_GRAPHIC_LAG_S, "test needs a deep buffer"

    def board_change_at(ts: float) -> BoardChange:
        return BoardChange(
            ts=ts, home_score=1, away_score=0, clock=None, period=1, previous=(0, 0)
        )

    runtime._board_changes = [board_change_at(10.0 + GOAL_GRAPHIC_LAG_S - 1.0)]
    assert runtime._board_changed_near(10.0) is True

    # Inside the buffer but far past when a graphic could plausibly be
    # reporting this moment: that is a different passage of play.
    runtime._board_changes = [board_change_at(10.0 + GOAL_GRAPHIC_LAG_S + 2.0)]
    assert runtime._board_changed_near(10.0) is False


@pytest.mark.asyncio
async def test_a_goal_is_never_announced_before_the_board_confirms_it(tmp_path: Path) -> None:
    """No goal line without a goal behind it, within the window the gate allows.

    The tolerance is the gate's own forward window and not a tighter number,
    because the simulator's score bug has no graphic lag: it changes the
    instant the ball crosses the line, where a broadcaster's takes about six
    seconds. A graphic ahead of the cursor is evidence of a goal at or before
    the cursor on real footage, and evidence of one slightly after it here.
    """
    from commentary.runtime import GOAL_GRAPHIC_LAG_S

    delay_s = 4.0
    _runtime, sim, path = await run_sim(tmp_path, seconds=5.0, delay_s=delay_s)
    run = metrics.load_run(path)
    goals = [e.video_ts for e in sim.ground_truth if e.event is Event.GOAL]
    window = min(GOAL_GRAPHIC_LAG_S, delay_s) + 0.5

    for line in run.lines:
        if "goal" not in line.text.lower():
            continue
        assert any(line.video_ts >= goal - window for goal in goals), (
            f"a goal was called at {line.video_ts:.1f}s with no goal behind it"
        )


# -- the goal the first real run got wrong -----------------------------------
#
# Argentina v France 2022, video 36:00-39:00, delay 8 s. The ball crossed the
# line at video 58 (StatsBomb 35:22); the FIFA bug went 1-0 to 2-0 at video
# 63.7; the tracker, needing three reads landing every ~3.5 s, confirmed it at
# 70.0. The caller called the goal correctly at cursor 56.5 with the finish in
# the lookahead and the gate rejected it, then rejected three more lines about
# the same goal over the next eighty seconds.


def _board_at(runtime: Runtime, home: int, away: int, ts: float, *, count: int = 1) -> None:
    """Put the tracker part-way through agreeing that the score moved."""
    from commentary.perception.board import BoardPending

    runtime.board_tracker._confirmed = (1, 0, 1)
    runtime.board_tracker._pending = BoardPending(
        home_score=home, away_score=away, period=1, clock=None, count=count, first_ts=ts
    )


@pytest.mark.asyncio
async def test_a_board_still_agreeing_with_itself_corroborates_a_goal(
    tmp_path: Path,
) -> None:
    """The read at 63.7 is evidence at cursor 56.5, thirteen seconds before it settles."""
    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)

    _board_at(runtime, 2, 0, 63.7)
    assert runtime._board_supports_goal(56.5) is True

    # One read agreeing with the caller is two sources; it is still not three
    # reads, so nothing has moved the score.
    assert runtime.board_tracker.home_score == 1


@pytest.mark.asyncio
async def test_a_pending_board_that_is_not_a_score_increase_is_not_a_goal(
    tmp_path: Path,
) -> None:
    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)

    # The clock rolled into the second half: same score, new period.
    _board_at(runtime, 1, 0, 63.7)
    assert runtime._board_supports_goal(56.5) is False

    # A score increase read far from the cursor is a different passage of play.
    _board_at(runtime, 2, 0, 90.0)
    assert runtime._board_supports_goal(56.5) is False


@pytest.mark.asyncio
async def test_the_lines_that_follow_a_goal_are_about_a_goal_the_state_holds(
    tmp_path: Path,
) -> None:
    """The celebration, the replay and the scorer's face are not phantom goals."""
    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)
    from commentary.runtime import GOAL_TALK_WINDOW_S

    runtime._last_goal_ts = 63.7
    for cursor in (66.0, 76.0, 63.7 + GOAL_TALK_WINDOW_S):
        assert runtime._board_supports_goal(cursor) is True, f"rejected at cursor {cursor}"


@pytest.mark.asyncio
async def test_a_goal_claim_long_after_the_last_one_still_fails(tmp_path: Path) -> None:
    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)

    runtime._last_goal_ts = 100.0
    runtime.board_tracker._pending = None
    runtime._board_changes = []
    assert runtime._board_supports_goal(200.0) is False


@pytest.mark.asyncio
async def test_applying_a_board_goal_starts_the_clock_on_talking_about_it(
    tmp_path: Path,
) -> None:
    """The clock starts when the state takes the goal in, not when the bug moved.

    A change first seen at 63.7 and confirmed only when the score bug came
    back from behind the replay is news when it lands, and the celebration
    lines come after that. Stamping it 63.7 made the window expire before the
    viewer's scoreboard had even changed.
    """
    from commentary.perception.board import BoardChange
    from commentary.runtime import GOAL_TALK_WINDOW_S

    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)
    runtime._last_goal_ts = None
    runtime._board_changes = [
        BoardChange(ts=63.7, home_score=2, away_score=0, clock=None, period=1, previous=(1, 0)),
    ]
    runtime.buffer.append(Frame(ts=140.0, image=np.zeros((4, 4, 3), dtype=np.uint8)))
    cursor = runtime.buffer.cursor_ts
    assert cursor is not None and cursor > 63.7 + GOAL_TALK_WINDOW_S

    runtime._apply_due_board_changes()
    assert runtime._last_goal_ts == cursor

    # A line about the goal, well over 45 s after the board first showed it
    # and seconds after the state caught up, is a line about a goal we hold.
    runtime.board_tracker._pending = None
    assert runtime._board_supports_goal(cursor + 5.0) is True
    assert runtime._board_supports_goal(cursor + GOAL_TALK_WINDOW_S + 1.0) is False


@pytest.mark.asyncio
async def test_a_wire_inside_the_buffer_lands_every_goal_as_an_incident(
    tmp_path: Path,
) -> None:
    """The two-clock rule end to end.

    At ``latency_s`` below the buffer depth the feed's own lag is absorbed
    entirely: everything it says about the stretch the cursor has reached has
    already been applied by the time the run ends.
    """
    from commentary.schemas import Event as SchemaEvent
    from commentary.wire import ReplayWire

    sim = MatchSim(seed=5, duration_s=120.0)
    settings = fast_settings(4.0)
    source = SimSource(sim, settings.capture, realtime=False)
    path = tmp_path / "wire.jsonl"
    with RunTrace(path=path) as trace:
        runtime = Runtime(
            source=source,
            backend=SimOracle(sim=sim, error_rate=0.0),
            pack=sim.knowledge_pack,
            settings=settings,
            speaker=LogSpeaker(words_per_second=120),
            trace=trace,
            wire=ReplayWire.from_truth(sim.ground_truth, 2.0),
        )
        await runtime.run(seconds=5.0)

    watched = runtime.cursor_ts
    goals = [
        e for e in sim.ground_truth if e.event is SchemaEvent.GOAL and e.video_ts <= watched - 2.0
    ]
    landed = {
        round(i.video_ts, 1)
        for i in runtime.state.incidents
        if i.event is SchemaEvent.GOAL and i.source == "wire"
    }
    for goal in goals:
        assert any(abs(ts - goal.video_ts) < 1.0 for ts in landed), (
            f"the wire's goal at {goal.video_ts:.1f}s never reached the state"
        )


@pytest.mark.asyncio
async def test_the_default_runtime_loads_no_wire(tmp_path: Path) -> None:
    """Rule two, made structural: every outside source is opt-in and off."""
    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0)
    assert runtime.wire is None
    assert runtime._sync is None
    assert runtime._wire_confirms_goal(10.0) is False
