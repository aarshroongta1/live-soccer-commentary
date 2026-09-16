"""The whole pipeline, end to end, against a match that never happened.

This is the test that would have been impossible to write tonight without the
simulator: frames in at one end, spoken commentary out at the other, with a
written record of what actually happened to check it against.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from commentary.agents.colour import Material
from commentary.capture.buffer import Frame
from commentary.config import (
    CallerConfig,
    CaptureConfig,
    DirectorConfig,
    PredictorConfig,
    Settings,
)
from commentary.grading import metrics, report
from commentary.orchestration import new_turn_state
from commentary.runtime import Runtime
from commentary.schemas import (
    Beat,
    CallerLine,
    Event,
    GateVerdict,
    Scene,
    Side,
    Sighting,
    Voice,
)
from commentary.sim import MatchSim, SimOracle, SimSource
from commentary.trace import RunTrace
from commentary.voice import LogSpeaker


def fast_settings(delay_s: float = 4.0) -> Settings:
    """A sim minute compressed into a test that finishes in seconds."""
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


def watch_the_director(runtime: Runtime) -> list[Any]:
    """Collect the beats the runtime submits, instead of speaking them."""
    submitted: list[Any] = []

    def submit(beat: Any) -> bool:
        submitted.append(beat)
        return True

    runtime.director.submit = submit  # type: ignore[method-assign]
    return submitted


async def one_caller_form(runtime: Runtime, line: CallerLine) -> None:
    """Drive one caller call with a form of our own and let the runtime have it."""

    async def call(*args: Any, **kwargs: Any) -> CallerLine:
        return line

    runtime.caller.call = call  # type: ignore[method-assign]
    await runtime._call([])


def a_replay(event: Event, text: str) -> CallerLine:
    return CallerLine(
        scene=Scene.REPLAY,
        event=event,
        side=Side.HOME,
        confidence=0.9,
        speak=True,
        line=text,
    )


@pytest.mark.asyncio
async def test_a_replay_is_spoken_as_a_replay_and_moves_nothing(tmp_path: Path) -> None:
    """Replay mode, end to end through the runtime.

    This test used to assert the opposite — that nothing is ever spoken while
    a replay is on screen — and that rule is why
    ``runs/trigger/mbappe/file-20260913-185228.jsonl`` runs from 12.9 s to
    49.0 s in silence with four accurate replay forms in it. A replay line may
    now be said. What it may not do is count: a replayed foul is not a second
    foul, and the man on the ball in a replay is not the man on the ball now.
    """
    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)
    runtime.phraser = None  # the caller's own words, judged as they stand
    runtime.state.last_events = [Event.FOUL]
    before = list(runtime.state.last_events)
    was_recent = runtime._recent_event
    submitted = watch_the_director(runtime)

    await one_caller_form(
        runtime, a_replay(Event.FOUL, "The leg was in behind him and down he went.")
    )

    assert [beat.text for beat in submitted] == ["The leg was in behind him and down he went."]
    # Preemptable whatever the event: the whole of what makes a replay line
    # safe is that the moment the game is back on the screen it can be dropped.
    assert submitted[0].preemptable is True
    assert runtime.state.last_events == before
    assert runtime._recent_event == was_recent
    assert runtime.replays.said == 1
    assert runtime.replays.first is False
    # And it takes the dead-ball rate rather than the rate its event would
    # have earned, because the ball is not in play behind the picture.
    assert runtime._last_spoken_replay is True


@pytest.mark.asyncio
async def test_a_replay_of_a_goal_never_opens_a_goal_window(tmp_path: Path) -> None:
    """The second goal of a match must not become the third on a replay."""
    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)
    cursor = runtime.cursor_ts
    assert cursor is not None
    runtime.phraser = None
    runtime._last_goal_ts = cursor
    runtime._restart_ts = None
    runtime.state.last_events = [Event.GOAL]
    watch_the_director(runtime)

    await one_caller_form(
        runtime,
        a_replay(Event.GOAL, "He had swung a leg at it and it had gone in off the post."),
    )

    assert runtime.follow.armed_at is None
    assert runtime._said_a_goal is False


@pytest.mark.asyncio
async def test_a_second_replay_line_too_soon_is_not_said(tmp_path: Path) -> None:
    """Two lines four seconds apart is as fast as the corpus's replay runs go."""
    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)
    cursor = runtime.cursor_ts
    assert cursor is not None
    runtime.phraser = None
    runtime.state.last_events = [Event.FOUL]
    submitted = watch_the_director(runtime)

    await one_caller_form(runtime, a_replay(Event.FOUL, "The trailing leg caught him."))
    # Two seconds on, another angle of the same tackle.
    later = cursor + runtime.settings.capture.delay_s + 2.0
    runtime.buffer.append(Frame(ts=later, image=np.zeros((4, 4, 3), dtype=np.uint8)))
    await one_caller_form(runtime, a_replay(Event.FOUL, "From behind, the foot was never near it."))

    assert len(submitted) == 1
    assert runtime.replays.said == 1


def test_a_replay_sequence_counts_looks_not_lines() -> None:
    """The sequence is a fact about the pictures, not about what was said."""
    from commentary.config import ReplayTalkConfig
    from commentary.runtime import ReplaySequence

    seq = ReplaySequence(cfg=ReplayTalkConfig())

    seq.look(10.0)
    assert seq.first is True
    assert seq.may_speak(10.0) is True
    seq.spoke(10.0)

    # Too soon: the corpus's own replay runs are three and four seconds apart.
    seq.look(12.0)
    assert seq.may_speak(12.0) is False
    assert seq.first is False

    seq.look(14.5)
    assert seq.may_speak(14.5) is True
    seq.spoke(14.5)
    seq.look(19.0)
    seq.spoke(19.0)
    seq.look(23.5)
    assert seq.may_speak(23.5) is False, "three is the longest run in the corpus"

    # A cut back to the game and then another replay is a second sequence.
    seq.look(40.0)
    assert seq.first is True
    assert seq.may_speak(40.0) is True


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
async def test_the_first_read_that_shows_a_goal_prompts_the_caller(tmp_path: Path) -> None:
    """The trigger fires on the first differing read, once, and again on confirmation.

    On the Mbappé penalty the bug flipped at live 86.1 and the caller was not
    prompted until the third agreeing read, at cursor 87.9 — seven seconds
    after the kick. The first read lands at the cursor as the ball is struck,
    with the finish in the lookahead, so that is when to ask.
    """
    from commentary.schemas import BoardRead, Trigger

    def board(home: int, away: int, *, visible: bool = True) -> BoardRead:
        return BoardRead(
            bug_visible=visible, home_score=home, away_score=away, clock="79:26", confidence=0.95
        )

    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)
    runtime.board_tracker._confirmed = (2, 0, 1)
    runtime.board_tracker._pending = None
    runtime._drain_triggers()

    runtime._take_board_read(board(2, 1), 86.1)
    assert runtime._drain_triggers() == [Trigger.BOARD_CHANGE]
    assert runtime.board_tracker.away_score == 0  # evidence, not belief

    # The second agreeing read is more evidence and no news.
    runtime._take_board_read(board(2, 1), 89.5)
    assert runtime._drain_triggers() == []

    # An absent read in the middle changes nothing either way.
    runtime._take_board_read(board(2, 1, visible=False), 91.0)
    assert runtime._drain_triggers() == []

    # The third confirms, moves the score, and fires as it always has.
    runtime._take_board_read(board(2, 1), 93.2)
    assert runtime._drain_triggers() == [Trigger.BOARD_CHANGE]
    assert runtime.board_tracker.away_score == 1

    # A read that is not a score increase prompts nobody.
    runtime._take_board_read(board(2, 0), 96.0)
    assert runtime._drain_triggers() == []


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
    from commentary.runtime import GOAL_TALK_CAP_S

    runtime._last_goal_ts = 63.7
    for cursor in (66.0, 76.0, 130.0, 63.7 + GOAL_TALK_CAP_S):
        assert runtime._board_supports_goal(cursor) is True, f"rejected at cursor {cursor}"


@pytest.mark.asyncio
async def test_the_gate_is_told_which_of_the_three_the_board_gave_it(tmp_path: Path) -> None:
    """A goal already in the state is cover for talking, not for counting.

    ``_board_supports_goal`` answers one question with three signals, and the
    broadest of them — a goal the state already holds — is the one that let
    "Argentina's third" out at two-nil. The gate's arithmetic needs that
    signal on its own: a number is settled or it is arriving, and only an
    arriving one may be a goal ahead of the board.
    """
    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)
    cursor = runtime.cursor_ts
    assert cursor is not None
    runtime._last_goal_ts = cursor
    runtime._restart_ts = None

    seen: dict[str, Any] = {}
    judged = runtime.gate.judge

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return judged(*args, **kwargs)

    runtime.gate.judge = spy  # type: ignore[method-assign]

    async def one_line(*args: Any, **kwargs: Any) -> CallerLine:
        return CallerLine(
            scene=Scene.LIVE_PLAY,
            event=Event.GOAL,
            side=Side.HOME,
            confidence=0.9,
            speak=True,
            line="And it is in!",
        )

    runtime.caller.call = one_line  # type: ignore[method-assign]
    await runtime._call([])

    assert seen["goal_in_state"] is True
    assert seen["board_changed"] is True


@pytest.mark.asyncio
async def test_a_goal_claim_long_after_the_last_one_still_fails(tmp_path: Path) -> None:
    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)

    runtime._last_goal_ts = 100.0
    runtime.board_tracker._pending = None
    runtime._board_changes = []
    assert runtime._board_supports_goal(300.0) is False


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
    from commentary.runtime import GOAL_TALK_CAP_S

    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)
    runtime._last_goal_ts = None
    runtime._board_changes = [
        BoardChange(ts=63.7, home_score=2, away_score=0, clock=None, period=1, previous=(1, 0)),
    ]
    runtime.buffer.append(Frame(ts=300.0, image=np.zeros((4, 4, 3), dtype=np.uint8)))
    cursor = runtime.buffer.cursor_ts
    assert cursor is not None and cursor > 63.7 + GOAL_TALK_CAP_S

    runtime._apply_due_board_changes()
    assert runtime._last_goal_ts == cursor

    # A line about the goal, well over the cap past the board first showing
    # it and seconds after the state caught up, is a line about a goal we
    # hold: the cap runs from when the viewer's scoreline changed.
    runtime.board_tracker._pending = None
    assert runtime._board_supports_goal(cursor + 5.0) is True
    assert runtime._board_supports_goal(cursor + GOAL_TALK_CAP_S + 1.0) is False


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


# -- goal talk runs until the game does --------------------------------------
#
# Second real run, one goal: the state took it in at cursor 66.8, the caller
# wrote a kickoff line at 153.5, and in between the broadcaster showed the
# celebration, the replays and the scorer's face. The two lines about the
# goal at 128.1 and 140.1 were rejected as phantom goals by a 45-second
# window while the picture was still on it.


def _goal_line(event: Event = Event.GOAL, scene: Scene = Scene.LIVE_PLAY) -> CallerLine:
    return CallerLine(
        scene=scene, event=event, confidence=0.8, speak=True, line="They are still celebrating."
    )


@pytest.mark.asyncio
async def test_the_celebration_is_still_about_the_goal_until_play_restarts(
    tmp_path: Path,
) -> None:
    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)
    runtime.board_tracker._pending = None
    runtime._board_changes = []
    runtime._last_goal_ts = 66.8

    for cursor in (128.1, 140.1):
        assert runtime._board_supports_goal(cursor) is True, f"rejected at cursor {cursor}"

    runtime._note_restart(_goal_line(event=Event.KICKOFF), 153.5)
    assert runtime._restart_ts == 153.5
    assert runtime._board_supports_goal(170.0) is False

    # The question is where the cursor is, not when the kickoff was noticed:
    # a line about the goal from before the restart is still about the goal.
    assert runtime._board_supports_goal(140.1) is True


@pytest.mark.asyncio
async def test_a_restart_nobody_saw_is_what_the_cap_is_for(tmp_path: Path) -> None:
    from commentary.runtime import GOAL_TALK_CAP_S

    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)
    runtime.board_tracker._pending = None
    runtime._board_changes = []
    runtime._last_goal_ts = 66.8

    assert runtime._board_supports_goal(66.8 + GOAL_TALK_CAP_S) is True
    assert runtime._board_supports_goal(66.8 + GOAL_TALK_CAP_S + 0.1) is False


@pytest.mark.asyncio
async def test_the_next_goal_starts_the_talking_over(tmp_path: Path) -> None:
    """A restart belongs to the goal it followed, not to the one after it."""
    from commentary.perception.board import BoardChange

    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)
    runtime.board_tracker._pending = None
    runtime._last_goal_ts = 66.8
    runtime._note_restart(_goal_line(event=Event.KICKOFF), 153.5)
    assert runtime._board_supports_goal(200.0) is False

    runtime._board_changes = [
        BoardChange(ts=190.0, home_score=3, away_score=0, clock=None, period=1, previous=(2, 0)),
    ]
    runtime.buffer.append(Frame(ts=200.0, image=np.zeros((4, 4, 3), dtype=np.uint8)))
    runtime._apply_due_board_changes()

    assert runtime._restart_ts is None
    assert runtime._board_supports_goal(runtime.cursor_ts + 30.0) is True


# -- the caller reads the shirt, the registry keeps the name ------------------
#
# On the real clip the local number reader confirmed nothing in three minutes
# and the caller read nine correct number-and-name pairs off the same frames.
# So the caller reports what it read against the tag drawn above the body,
# and this is where the two are joined — with the team sheets as the check,
# because a wrong pair here follows that player until the next cut. The tag
# is a letter: mark "E" is track 4.


def _sighting_line(*sightings: Sighting) -> CallerLine:
    return CallerLine(
        scene=Scene.LIVE_PLAY,
        event=Event.CARRY,
        confidence=0.8,
        speak=True,
        line="He drives at the defence.",
        sightings=list(sightings),
    )


class _Binder:
    """Records what the registry was told to believe, and by which side."""

    def __init__(self) -> None:
        self.bound: list[tuple[Side, int, str]] = []


async def _with_sightings(tmp_path: Path, *sightings: Sighting) -> tuple[Runtime, _Binder]:
    runtime, sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)
    binder = _Binder()
    registry = runtime.state_tracker.registry
    original = registry.believe

    def spy(number: int, name: str, ts: float, *, side: Side = Side.UNKNOWN, **kw: Any) -> None:
        binder.bound.append((side, number, name))
        original(number, name, ts, side=side, **kw)

    registry.believe = spy  # type: ignore[method-assign]
    # The sim run above binds sightings of its own; the counters here are
    # about the ones this test hands over.
    runtime.stats.sightings = runtime.stats.sightings_dropped = 0
    runtime._bind_sightings(_sighting_line(*sightings), runtime.cursor_ts)
    return runtime, binder


def _home_number(runtime: Runtime) -> tuple[int, str]:
    """A real number and name off the sim's own home sheet."""
    assert runtime.pack is not None
    player = next(p for p in runtime.pack.home.squad if p.number is not None)
    assert player.number is not None
    return player.number, player.name


@pytest.mark.asyncio
async def test_a_sighting_names_the_track_and_is_believed(tmp_path: Path) -> None:
    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)
    number, name = _home_number(runtime)

    runtime, binder = await _with_sightings(tmp_path, Sighting(number=number, name=name))

    assert binder.bound == [(Side.HOME, number, name)]
    assert runtime.state_tracker.registry.name_for(number, Side.HOME) == name
    assert runtime.stats.sightings == 1


@pytest.mark.asyncio
async def test_a_first_name_binds_when_side_and_number_settle_the_player(tmp_path: Path) -> None:
    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)
    number, name = _home_number(runtime)
    first_name = name.split()[0]

    runtime, binder = await _with_sightings(
        tmp_path,
        Sighting(number=number, name=first_name, side=Side.HOME),
    )

    assert binder.bound == [(Side.HOME, number, name)]


@pytest.mark.asyncio
async def test_a_first_name_without_a_side_is_not_bound(tmp_path: Path) -> None:
    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)
    number, name = _home_number(runtime)

    _runtime, binder = await _with_sightings(
        tmp_path,
        Sighting(number=number, name=name.split()[0]),
    )

    assert binder.bound == []


@pytest.mark.asyncio
async def test_a_first_name_without_a_number_is_not_bound(tmp_path: Path) -> None:
    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)
    _number, name = _home_number(runtime)

    _runtime, binder = await _with_sightings(
        tmp_path,
        Sighting(name=name.split()[0], side=Side.HOME),
    )

    assert binder.bound == []


@pytest.mark.asyncio
async def test_a_number_alone_names_somebody_only_if_one_squad_wears_it(
    tmp_path: Path,
) -> None:
    """The kit split does not get to say who a shirt belongs to.

    On the real clip it put an Argentina body on France, and a sighting of
    "26" on that track became Marcus Thuram through a passage Argentina
    played the whole of. Both squads had a 26; nothing that saw the shirt
    could tell them apart, so nothing should have claimed to.
    """
    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)
    assert runtime.pack is not None
    home = {p.number for p in runtime.pack.home.squad if p.number is not None}
    away = {p.number for p in runtime.pack.away.squad if p.number is not None}
    shared = sorted(home & away)
    only_home = sorted(home - away)

    if only_home:
        number = only_home[0]
        name = next(p.name for p in runtime.pack.home.squad if p.number == number)
        _runtime, binder = await _with_sightings(tmp_path, Sighting(number=number))
        assert binder.bound == [(Side.HOME, number, name)]

    if shared:
        _runtime, both = await _with_sightings(tmp_path, Sighting(number=shared[0]))
        assert both.bound == [], "two players wear it and the caller did not say which kit"


@pytest.mark.asyncio
async def test_the_caller_says_which_kit_and_a_shared_number_binds(tmp_path: Path) -> None:
    """Both squads wear an 11, and the one looking at the shirt settles it.

    On the penalty clip 18 of 34 sightings were dropped, nearly all of them a
    bare number two players in the match wear. The side is the caller's to
    give — it has the kit strings in its team sheets and it is the thing
    looking at the picture — and it is not the kit split's, which put an
    Argentina body on France and named a France forward off it.
    """
    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)
    assert runtime.pack is not None
    home = {p.number for p in runtime.pack.home.squad if p.number is not None}
    away = {p.number for p in runtime.pack.away.squad if p.number is not None}
    shared = sorted(home & away)
    if not shared:
        pytest.skip("this pack's two squads share no number")
    number = shared[0]

    for side, sheet in ((Side.HOME, runtime.pack.home), (Side.AWAY, runtime.pack.away)):
        name = next(p.name for p in sheet.squad if p.number == number)
        _runtime, binder = await _with_sightings(
            tmp_path, Sighting(number=number, side=side)
        )
        assert binder.bound == [(side, number, name)]


@pytest.mark.asyncio
async def test_a_side_the_number_contradicts_is_dropped(tmp_path: Path) -> None:
    """A number nobody on that side wears is a misread of one or the other.

    The sim's two squads wear the same eleven numbers, so this one builds the
    squads it needs rather than hunting the sim's pack for a number only one
    side has.
    """
    from commentary.schemas import KnowledgePack, Player, TeamSheet

    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)
    runtime.pack = KnowledgePack(
        home=TeamSheet(
            name="Argentina", kit="stripes", starters=[Player(name="Molina", number=26)]
        ),
        away=TeamSheet(name="France", kit="navy", starters=[Player(name="Thuram", number=9)]),
    )

    assert runtime._roster_check(Sighting(number=26, side=Side.AWAY), 0.0) is None
    assert runtime._roster_check(Sighting(number=26, side=Side.HOME), 0.0) == (
        Side.HOME,
        26,
        "Molina",
    )


@pytest.mark.asyncio
async def test_a_name_and_a_kit_that_disagree_are_dropped(tmp_path: Path) -> None:
    """One of the two was misread and nothing here can say which."""
    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)
    number, name = _home_number(runtime)

    _runtime, binder = await _with_sightings(
        tmp_path, Sighting(number=number, name=name, side=Side.AWAY)
    )
    assert binder.bound == []


@pytest.mark.asyncio
async def test_a_number_not_in_the_squad_is_dropped(tmp_path: Path) -> None:
    runtime, binder = await _with_sightings(tmp_path, Sighting(number=98))

    assert binder.bound == []
    assert runtime.stats.sightings_dropped == 1


@pytest.mark.asyncio
async def test_a_name_on_no_roster_is_dropped(tmp_path: Path) -> None:
    _runtime, binder = await _with_sightings(tmp_path, Sighting(name="Zaltimore"))
    assert binder.bound == []


@pytest.mark.asyncio
async def test_a_rejected_sighting_never_pollutes_match_state() -> None:
    """Validation happens before caller observations enter durable facts."""
    runtime = _built_runtime()
    runtime.phraser = None
    submitted = watch_the_director(runtime)

    await one_caller_form(
        runtime,
        CallerLine(
            scene=Scene.LIVE_PLAY,
            event=Event.CARRY,
            side=Side.HOME,
            sightings=[Sighting(number=99, name="Zaltimore", side=Side.HOME)],
            confidence=0.9,
            speak=True,
            line="The runner carries it towards the area.",
        ),
    )

    assert runtime.facts.registry.name_for(99, Side.HOME) is None
    assert "99" not in runtime.state.on_pitch
    assert submitted == []


@pytest.mark.asyncio
async def test_a_sighting_whose_number_and_name_disagree_is_dropped(tmp_path: Path) -> None:
    """Two readings of one shirt that cannot both be right is neither."""
    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)
    number, name = _home_number(runtime)

    _runtime, binder = await _with_sightings(
        tmp_path, Sighting(number=number + 40, name=name)
    )
    assert binder.bound == []


@pytest.mark.asyncio
async def test_a_read_is_believed_by_the_registry(tmp_path: Path) -> None:
    """A name and a number the caller read are what the registry is for."""
    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)
    number, name = _home_number(runtime)

    runtime, binder = await _with_sightings(tmp_path, Sighting(number=number, name=name))

    assert binder.bound == [(Side.HOME, number, name)]
    assert runtime.state_tracker.registry.name_for(number, Side.HOME) == name
    assert runtime.stats.sightings == 1


@pytest.mark.asyncio
async def test_a_goal_from_a_set_piece_is_a_goal_to_the_director(tmp_path: Path) -> None:
    """Ronaldo's free kick was called, named, and dropped on the camera cut.

    "curls it over the wall and into the top corner", tagged `free_kick` by
    the caller because a free kick is what put the ball there. `free_kick` is
    preemptable and `goal` is not, so the cut every broadcaster makes the
    instant a goal goes in killed the best line in the clip. What the line
    says outranks what it was filed under.
    """
    from commentary.gate import claims_goal
    from commentary.schemas import Event as E

    said = "Ronaldo curls it over the wall and into the top corner"
    assert claims_goal(said, E.FREE_KICK)
    assert not claims_goal("Ronaldo stands over the free kick", E.FREE_KICK)
    # The runtime turns that into the beat's event, which is what the
    # director routes and preempts on.
    event = E.GOAL if claims_goal(said, E.FREE_KICK) else E.FREE_KICK
    assert event is E.GOAL
    assert event in (E.GOAL, E.PENALTY)


@pytest.mark.asyncio
async def test_a_name_carries_through_the_run_it_was_read_on(tmp_path: Path) -> None:
    """Molina was named twice on his run and anonymous when he finished it.

    The name is recorded from the line that was actually spoken, not from
    every sighting reported: a name the caller read and did not say is not
    what the commentary was about, and carrying it would invent a subject
    rather than keep one.
    """
    from commentary.schemas import Sighting

    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)
    registry = runtime.state_tracker.registry
    said = CallerLine(
        scene=Scene.LIVE_PLAY,
        event=Event.CARRY,
        side=Side.HOME,
        confidence=0.8,
        speak=True,
        line="Molina bursting up the right",
        sightings=[Sighting(number=26, name="Molina", side=Side.HOME)],
    )
    runtime._remember_on_the_ball(said, said.line, 10.0)
    assert registry.on_the_ball is not None and registry.on_the_ball.name == "Molina"

    later = CallerLine(
        scene=Scene.LIVE_PLAY, event=Event.GOAL, side=Side.HOME,
        confidence=0.8, speak=True, line="and it is in at the near post",
    )
    assert runtime._carried_name(later, 14.0) == "Molina"
    # Past the window the name has to be read again.
    assert runtime._carried_name(later, 30.0) is None
    # A restart ends it whatever the clock says.
    restart = later.model_copy(update={"event": Event.CORNER})
    assert runtime._carried_name(restart, 14.0) is None


@pytest.mark.asyncio
async def test_a_dead_ball_name_is_never_carried(tmp_path: Path) -> None:
    """The one name the caller had at a penalty was the wrong man.

    On the Netherlands clip it reported `21 de Jong` while van Dijk stood
    over the ball, and called him "the Dutch taker in orange" instead. A
    sighting says "I read this number on somebody in this picture", never
    "this is the man on the ball".
    """
    from commentary.schemas import Sighting

    runtime, _sim, _path = await run_sim(tmp_path, seconds=1.0, delay_s=8.0)
    over_the_ball = CallerLine(
        scene=Scene.STOPPAGE,
        event=Event.PENALTY,
        side=Side.HOME,
        confidence=0.8,
        speak=True,
        line="de Jong stands over it",
        sightings=[Sighting(number=21, name="de Jong", side=Side.HOME)],
    )
    runtime._remember_on_the_ball(over_the_ball, over_the_ball.line, 10.0)
    kick = CallerLine(
        scene=Scene.LIVE_PLAY, event=Event.GOAL, side=Side.HOME,
        confidence=0.8, speak=True, line="and it is buried",
    )
    assert runtime._carried_name(kick, 12.0) is None


# -- the colour seat's attribution check, on the live path -------------------


def _built_runtime() -> Runtime:
    """A runtime with frames in the buffer, built rather than run."""
    sim = MatchSim(seed=5, duration_s=120.0)
    settings = fast_settings()
    runtime = Runtime(
        source=SimSource(sim, settings.capture, realtime=False),
        backend=SimOracle(sim=sim),
        pack=sim.knowledge_pack,
        settings=settings,
        speaker=LogSpeaker(words_per_second=120),
    )
    blank = np.zeros((8, 8, 3), dtype=np.uint8)
    for index in range(96):
        runtime.buffer.append(Frame(ts=index / settings.capture.fps, image=blank))
    return runtime


@pytest.mark.asyncio
async def test_the_caller_uses_the_fact_snapshot_captured_for_its_turn() -> None:
    runtime = _built_runtime()
    seen: list[str] = []
    line = CallerLine(
        scene=Scene.LIVE_PLAY,
        event=Event.CARRY,
        side=Side.HOME,
        confidence=0.9,
        speak=True,
        line="Hale carries through midfield.",
    )

    async def call(*args: Any, **kwargs: Any) -> CallerLine:
        seen.append(args[1])
        return line

    runtime.caller.call = call  # type: ignore[method-assign]
    state = new_turn_state(
        match_id="match-1",
        turn_id="turn-1",
        cursor_s=5.0,
        live_s=9.0,
        triggers=[],
        fact_version=3,
        fact_summary="the immutable prompt facts",
        match_state=runtime.state.model_copy(deep=True),
        goal_in_state=False,
    )

    result = await runtime.call_caller(state)

    assert result.form == line
    assert seen == ["the immutable prompt facts"]


def test_a_completed_lead_beat_is_only_committed_once() -> None:
    runtime = _built_runtime()
    runtime.phraser = None
    submitted = watch_the_director(runtime)
    line = CallerLine(
        scene=Scene.LIVE_PLAY,
        event=Event.CARRY,
        side=Side.HOME,
        confidence=0.9,
        speak=True,
        line="Hale carries through midfield.",
    )
    verdict = GateVerdict(passed=True, line=line.line)
    beat = Beat(
        id="lead:match-1:turn-1",
        voice=Voice.CALLER,
        text=line.line,
        video_ts=5.0,
        created_ts=0.0,
        live_ts=9.0,
        event=line.event,
    )
    spoken_before = runtime.stats.spoken

    runtime._commit_lead(line, verdict, beat)
    runtime._commit_lead(line, verdict, beat)

    assert len(submitted) == 1
    assert submitted[0].created_ts > 0.0
    assert runtime.stats.spoken == spoken_before + 1


@pytest.mark.asyncio
async def test_a_colour_utterance_that_blames_the_wrong_man_is_refused_live() -> None:
    """The attribution check, through the runtime rather than the offline pass.

    ``judge_utterance`` takes ``attributed``, ``named_before`` and ``after``,
    and without them ``misattributes`` returns on its first line: the check
    was inert on the live path while the offline rephrase had it. What is
    held here is the wiring — the second utterance is judged knowing who the
    first one named, because the second one says "he" and means him, and
    knowing who the caller's own forms put on the penalty.
    """
    runtime = _built_runtime()
    assert runtime.pack is not None
    on_the_penalty = runtime.pack.home.starters[0]
    somebody_else = runtime.pack.away.starters[0]
    cursor = runtime.cursor_ts
    runtime.colour.saw_form(
        cursor,
        CallerLine(
            scene=Scene.LIVE_PLAY,
            event=Event.PENALTY,
            side=Side.HOME,
            sightings=[
                Sighting(number=on_the_penalty.number, name=on_the_penalty.name, side=Side.HOME)
            ],
            confidence=0.9,
            speak=True,
            line=f"{on_the_penalty.surname} stands over it, the referee pointing to the spot.",
        ),
    )
    judged: list[tuple[str, Any]] = []
    original = runtime._publish

    def spy(topic: Any, ts: float, value: Any = None, **extra: Any) -> None:
        judged.append((str(getattr(topic, "value", topic)), value))
        original(topic, ts, value, **extra)

    runtime._publish = spy  # type: ignore[method-assign]

    await runtime._say_colour(
        [
            f"Well, {somebody_else.surname} is the man here.",
            "And he's just conceded the penalty.",
        ],
        [cursor, cursor],
        "over_a_replay",
    )

    verdicts = [value for topic, value in judged if topic == "gate"]
    assert len(verdicts) == 2
    assert verdicts[0].passed, "naming the man is a line, and it is the antecedent"
    assert not verdicts[1].passed
    assert verdicts[1].reasons[0].startswith("attribution:")
    assert somebody_else.surname in verdicts[1].reasons[0]


@pytest.mark.asyncio
async def test_a_colour_utterance_that_echoes_the_lead_is_refused_live() -> None:
    """Two voices sharing four words eleven seconds apart is one voice.

    ``judge_utterance`` takes ``lead_said`` for this and the live path passed
    nothing, so the check was inert on air while the offline pass had it.
    """
    runtime = _built_runtime()
    assert runtime.pack is not None
    scorer = runtime.pack.home.starters[0].surname
    runtime.colour.saw_lead_line(
        runtime.cursor_ts, f"{scorer} knew it from the moment it left his boot."
    )
    judged: list[tuple[str, Any]] = []
    original = runtime._publish

    def spy(topic: Any, ts: float, value: Any = None, **extra: Any) -> None:
        judged.append((str(getattr(topic, "value", topic)), value))
        original(topic, ts, value, **extra)

    runtime._publish = spy  # type: ignore[method-assign]

    await runtime._say_colour(
        [f"Yeah, {scorer} knew it from the moment it left his boot."],
        [runtime.cursor_ts],
        "after_a_goal",
    )

    verdicts = [value for topic, value in judged if topic == "gate"]
    assert len(verdicts) == 1
    assert not verdicts[0].passed
    assert verdicts[0].reasons[0].startswith("echoes_lead:")


@pytest.mark.asyncio
async def test_a_turn_made_only_of_a_count_has_to_say_it_happened_again() -> None:
    """``only_repeated`` is the other argument the live path was not passing.

    A count is material for one thing and one thing only: the observation
    that it has happened again. A line that takes the count and says
    something else about the man is a line with nothing behind it.
    """
    runtime = _built_runtime()
    # A turn whose whole material is a count: patterns and nothing else.
    runtime.colour.last_material = Material(patterns=("the same man down that side, four times",))
    assert runtime.colour.last_material.only_a_count
    judged: list[tuple[str, Any]] = []
    original = runtime._publish

    def spy(topic: Any, ts: float, value: Any = None, **extra: Any) -> None:
        judged.append((str(getattr(topic, "value", topic)), value))
        original(topic, ts, value, **extra)

    runtime._publish = spy  # type: ignore[method-assign]
    assert runtime.pack is not None
    somebody = runtime.pack.home.starters[0].surname

    await runtime._say_colour(
        [f"Well, {somebody} has been the best of them tonight."],
        [runtime.cursor_ts],
        "in_build_up",
    )

    verdicts = [value for topic, value in judged if topic == "gate"]
    assert len(verdicts) == 1
    assert not verdicts[0].passed
    assert verdicts[0].reasons[0].startswith("pattern_unsaid:")
