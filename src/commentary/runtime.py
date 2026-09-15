"""The match loop: everything wired together, running against a clock.

Five things happen at once. Frames and sound pour into the ring buffers at
the live edge. The board reader glances at the score bug every couple of
seconds. The speak predictor ticks, and when it says so the caller looks at
the cursor and the near future and fills in a form. The fact gate judges that
form. The director decides who says what, and cuts someone off when a goal
goes in.

One subtlety governs the whole file. The board reader reads at the LIVE edge,
because that is how the system can know a goal went in before the narration
cursor has reached it. But a board change is not applied to match state until
the cursor passes the moment it happened, so the commentary can never
announce something the viewer has not seen yet. The early read is used only
as evidence: when the caller claims a goal, the gate asks whether the board
moved around that moment, which is exactly the question a human commentator
answers by glancing up at the graphic.

"Around that moment" is a fixed window, not the buffer depth. Tying it to
the buffer made the gate more permissive the longer we chose to wait, which
is a strange thing for a safety check to do and undid most of what the delay
was for.

The glance is not only at the settled board. Confirmation takes three
agreeing reads and lands well after the goal, so the gate also accepts a
board change the tracker is still gathering evidence for, and accepts a goal
the state has already taken in as cover for the lines that follow one. The
first read that disagrees with the settled score in a goal's direction is
also enough to *prompt* the caller, which is how the kick gets called as it
happens rather than seven seconds on. What the score itself is allowed to
move on does not change: three reads.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from commentary.agents.analyst import Analyst
from commentary.agents.caller import Caller
from commentary.agents.phraser import Phraser
from commentary.bus import Bus, Topic
from commentary.capture.audio import CutDetector
from commentary.capture.buffer import DelayBuffer, Frame
from commentary.config import SETTINGS, Settings
from commentary.director import Director, next_beat_id
from commentary.gate import FactGate, claims_goal, fold, is_the_same_name
from commentary.llm.base import LLMBackend, Usage
from commentary.perception.board import BoardChange, BoardReader, BoardTracker
from commentary.predictor import SpeakPredictor
from commentary.schemas import (
    Beat,
    BoardRead,
    CallerLine,
    Event,
    Incident,
    KnowledgePack,
    MatchState,
    Player,
    Side,
    Sighting,
    SpeakDecision,
    Trigger,
    Voice,
)
from commentary.state import MatchStateTracker, parse_clock, period_for_clock
from commentary.tools import MatchTools
from commentary.trace import RunTrace
from commentary.voice.speaker import WORDS_PER_SECOND, LogSpeaker, Speaker
from commentary.wire import Wire, WireSync

#: How long after the ball crosses the line a broadcaster's score bug
#: catches up. A property of television, not of our buffer, which is the
#: whole point: the window the gate will accept a board change in must not
#: grow when we choose to wait longer.
#:
#: Measured on the first run against real footage (Argentina v France 2022):
#: the ball crossed the line at video 58 (StatsBomb 35:22) and the FIFA bug
#: went 1-0 to 2-0 at video 63.7, 5.7 s later. At the old 5.0 the window
#: [cursor - 2, cursor + 5] closed at 61.5 and missed it, and the caller's
#: correct goal call was rejected as unconfirmed.
GOAL_GRAPHIC_LAG_S = 10.0

#: The backstop on how long a goal stays a thing worth talking about, and
#: only the backstop: what really ends it is play restarting.
#:
#: This was 45 seconds of cursor time and that was wrong in kind. A
#: broadcaster spends 60 to 90 seconds after a goal on the celebration, the
#: replays, the scorer's face and the walk back, and every line about the
#: goal in that stretch is a line about something the state holds. On the
#: second real run the celebration lines at 128.1 and 140.1 were rejected as
#: phantom goals for being more than 45 s past a goal applied at 66.8 — while
#: the picture was still showing it, and the kickoff was still 13 s away.
#:
#: The cap exists because the restart can be missed: no kickoff line
#: written. Past it the picture has moved on whatever the caller says.
GOAL_TALK_CAP_S = 150.0

#: How long a name stays on the ball. The caller reads a shirt, the player
#: turns, and the number is gone while the move it is part of is still going
#: on: Molina was named twice on his run and anonymous when he finished it.
#: Eight seconds is the length of a run, not of a passage of play — past it
#: the name has to be read again.
CARRY_NAME_S = 8.0


def _player_named(pack: KnowledgePack, name: str) -> tuple[Side, Player] | None:
    """Which player on either sheet this is, if it is one of them.

    One matcher, shared with the gate: a read the gate accepts and the
    runtime refuses to bind is a name in a line with nothing holding it up.
    """
    if not fold(name):
        return None
    for side in (Side.HOME, Side.AWAY):
        sheet = pack.team(side)
        if sheet is None:
            continue
        for player in sheet.squad:
            if is_the_same_name(name, player.name):
                return side, player
    return None


def _player_numbered(pack: KnowledgePack, side: Side, number: int) -> Player | None:
    sheet = pack.team(side)
    if sheet is None:
        return None
    for player in sheet.squad:
        if player.number == number:
            return player
    return None


@dataclass
class RuntimeStats:
    frames: int = 0
    board_reads: int = 0
    caller_calls: int = 0
    analyst_calls: int = 0
    ticks: int = 0
    gated_out: int = 0
    sightings: int = 0
    sightings_dropped: int = 0
    spoken: int = 0
    cost_stopped: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Runtime:
    """One match, from first frame to final whistle."""

    source: Any
    backend: LLMBackend
    pack: KnowledgePack | None = None
    settings: Settings = SETTINGS
    speaker: Speaker = field(default_factory=LogSpeaker)
    trace: RunTrace | None = None
    home: str = "Home"
    away: str = "Away"
    #: The second voice. Off by one flag, because "single voice" is one of the
    #: ablations the results table has to report.
    with_analyst: bool = True
    #: A statistician's feed. ``None`` in the default runtime and in every
    #: row of the results table but one: the thesis is the picture, the sound
    #: and notes, and this is here to say what a feed would have bought.
    wire: Wire | None = None

    bus: Bus = field(default_factory=Bus)
    stats: RuntimeStats = field(default_factory=RuntimeStats)

    def __post_init__(self) -> None:
        cap = self.settings.capture
        if self.pack is not None:
            self.home, self.away = self.pack.home.name, self.pack.away.name

        self.buffer = DelayBuffer(cap.fps, cap.delay_s, cap.history_s)

        self.board_reader = BoardReader(self.backend, config=self.settings.board)
        self.board_tracker = BoardTracker(self.settings.board)
        self.state_tracker = (
            MatchStateTracker.from_pack(self.pack)
            if self.pack is not None
            else MatchStateTracker(home=self.home, away=self.away)
        )
        self.caller = Caller(
            self.backend,
            config=self.settings.caller,
            pack=self.pack,
        )
        #: The speaking half of the play-by-play voice, or ``None`` when
        #: ``PHRASER_MODEL=off``. None is not a degraded mode: it is exactly
        #: the runtime that existed before the split, and a test asserts that
        #: the spoken lines are identical with it off.
        phraser = Phraser(
            self.backend,
            config=self.settings.phraser,
            home=self.home,
            away=self.away,
        )
        self.phraser: Phraser | None = phraser if phraser.enabled else None
        self.analyst = Analyst(
            self.backend,
            config=self.settings.analyst,
            tools=MatchTools(state=self.state_tracker.state, pack=self.pack),
            pack=self.pack,
        )
        self.gate = FactGate(self.settings.gate)
        self.predictor = SpeakPredictor(self.settings.predictor, self.settings.caller)
        self.director = Director(speaker=self.speaker, cfg=self.settings.director, bus=self.bus)

        self.cut = CutDetector(self.settings.predictor)

        self._pending: list[Trigger] = []
        self._board_changes: list[BoardChange] = []
        self._cuts: list[float] = []
        #: Cursor time at which the state last took in a board goal — not the
        #: time of the board change itself. See ``_apply_due_board_changes``.
        self._last_goal_ts: float | None = None
        #: Cursor time at which play was seen to restart after that goal, and
        #: the end of talking about it. ``None`` until it is seen.
        self._restart_ts: float | None = None
        self._last_spoken_video_ts: float | None = None
        #: How long that line takes to say. The rate cap is a debt the last
        #: line ran up, and a fragment runs up less of one than a sentence,
        #: so the predictor needs the length and not just the timestamp.
        self._last_spoken_seconds: float | None = None
        #: Whether the last line the caller got past the gate claimed a goal.
        #: The four seconds after one are the scorer's name, the celebration
        #: and the replay arriving together, and the rate cap spent them
        #: silent.
        self._said_a_goal = False
        #: Which side the player on the ball plays for, so a carried name
        #: cannot cross to the other team on the next line.
        self._carry_side = Side.UNKNOWN
        self._last_analyst_ts: float = 0.0
        self._recent_event: tuple[Event, float] | None = None
        self._sync = WireSync(self.wire) if self.wire is not None else None
        self._stop = asyncio.Event()

    # -- what the web layer is allowed to see ----------------------------

    @property
    def state(self) -> MatchState:
        return self.state_tracker.state

    @property
    def usage(self) -> Usage:
        return self.backend.total

    @property
    def present_offset_s(self) -> float:
        """How far behind the cursor the viewer's picture is held."""
        return self.settings.capture.present_offset_s

    def status(self) -> dict[str, Any]:
        stats = self.gate.stats
        return {
            **self.stats.as_dict(),
            **self.director.stats.as_dict(),
            "buffered_frames": len(self.buffer),
            "cursor_ts": self.buffer.cursor_ts,
            "live_ts": self.buffer.live_ts,
            "gate": {
                "judged": stats.judged,
                "passed": stats.passed,
                "trimmed": stats.trimmed,
                "rejected": stats.rejected,
                "by_reason": dict(stats.by_reason),
            },
        }

    # -- the loops -------------------------------------------------------

    async def run(self, seconds: float | None = None) -> RuntimeStats:
        async with self.source:
            # The recorder subscribes before anything can publish, so the
            # trace starts at the first frame rather than the first race.
            recorder = asyncio.create_task(self._record(), name="trace")
            await asyncio.sleep(0)
            tasks = [
                recorder,
                asyncio.create_task(self._ingest_frames(), name="frames"),
                asyncio.create_task(self._read_board(), name="board"),
                asyncio.create_task(self._tick(), name="tick"),
                asyncio.create_task(self.director.run(), name="director"),
            ]
            if seconds is not None:
                tasks.append(asyncio.create_task(self._deadline(seconds), name="deadline"))

            try:
                await self._stop.wait()
            finally:
                # Let whatever is mid-sentence finish, then close the books
                # while the recorder is still listening, so the final tallies
                # land in the trace rather than after it.
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self.director.drain(timeout=2.0), timeout=2.5)
                self._publish(Topic.STATUS, self.live_ts, **self.status())
                self._publish(
                    Topic.COST, self.live_ts, total_usd=round(self.backend.total.cost_usd, 4)
                )
                await asyncio.sleep(0)
                self.director.stop()
                for task in tasks:
                    task.cancel()
                for task in tasks:
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
        return self.stats

    async def _deadline(self, seconds: float) -> None:
        await asyncio.sleep(seconds)
        self.stop()

    def stop(self) -> None:
        self._stop.set()

    @property
    def live_ts(self) -> float:
        return self.buffer.live_ts or 0.0

    @property
    def cursor_ts(self) -> float:
        return self.buffer.cursor_ts or 0.0

    async def _ingest_frames(self) -> None:
        async for frame in self.source.frames():
            self.buffer.append(frame)
            self.stats.frames += 1
            self._apply_due_board_changes()
            self._apply_due_wire()
            if (cut := self.cut.feed(frame)) is not None:
                # Kept with its timestamp: a cut is both a reason to consider
                # speaking and the edge of what counts as "next" for the
                # caller, and only the second of those needs to know when.
                self._cuts.append(cut.ts)
                self._fire(Trigger.CAMERA_CUT)
        # The source ran out: a clip ended, or the stream died. Either way the
        # match is over as far as this process is concerned.
        self.stop()

    async def _read_board(self) -> None:
        """Glance at the score bug, at the live edge, on a fixed interval."""
        while True:
            await asyncio.sleep(self.settings.board.interval_s)
            frame = self._live_frame()
            if frame is None:
                continue
            try:
                read = await self.board_reader.read(frame)
            except Exception as exc:  # a missed glance is not a dead match
                self._publish(Topic.ERROR, frame.ts, where="board", detail=str(exc))
                continue
            self.stats.board_reads += 1
            self._publish(Topic.BOARD, frame.ts, read)
            self._take_board_read(read, frame.ts)

    def _take_board_read(self, read: BoardRead, ts: float) -> None:
        """One glance at the bug, and what the rest of the system makes of it.

        A confirmed change fires the trigger, as it always did. So does the
        *first* read that disagrees with the settled score in a goal's
        direction. The board reader runs a buffer's length ahead of the
        cursor and a broadcaster's graphic runs six seconds behind the ball,
        so that first read lands at the cursor about when the ball is
        crossing the line, with the finish inside the caller's lookahead.
        Waiting for the third agreeing read fired it seven seconds after the
        kick instead — on the Mbappé penalty, that was a line about the
        run-up at 76.6, a lull handed to the analyst at 80.8, and the goal
        called at 87.9. Only the trigger moves early; the score still waits
        for three reads, and the gate already treated one agreeing read as
        corroboration.
        """
        self._observe_clock(read, ts)
        change = self.board_tracker.update(read, ts)
        if change is not None:
            self._board_changes.append(change)
            self._fire(Trigger.BOARD_CHANGE)
        elif self._first_sight_of_a_goal(ts):
            self._fire(Trigger.BOARD_CHANGE)
        self._note_screen()

    def _first_sight_of_a_goal(self, ts: float) -> bool:
        """Did the read just taken open a goal-shaped disagreement with the board?

        Once per pending change: the second and third agreeing reads add
        evidence, not news, and the confirmation fires on its own.
        """
        pending = self.board_tracker.pending_goal
        return pending is not None and pending.count == 1 and pending.first_ts == ts

    def _observe_clock(self, read: BoardRead, ts: float) -> None:
        """Tell the sync where the match clock and the video clock meet.

        The feed counts in match time and everything here counts in video
        time, and the board reader's own glance at the graphic is the only
        bridge between them that exists at runtime. This is the whole of the
        board's involvement: when an event is *released* is a question about
        the live edge, and the frame loop answers that one.
        """
        if self._sync is None or not read.bug_visible or read.clock is None:
            return
        clock_s = parse_clock(read.clock)
        period = period_for_clock(read.clock)
        if clock_s is None or period is None:
            return
        self._sync.observe_clock(clock_s, period, ts)

    def _live_frame(self) -> Frame | None:
        live = self.buffer.live_ts
        return self.buffer.nearest(live) if live is not None else None

    def _note_screen(self) -> None:
        """Push a change in what the board looks like, replay or gone entirely."""
        tracker = self.board_tracker
        seen = (tracker.in_replay, not tracker.bug_missing)
        if seen != (self.state.in_replay, self.state.bug_visible):
            self.state_tracker.apply_board(tracker)
            self._publish(Topic.STATE, self.cursor_ts, self.state)

    def _apply_due_board_changes(self) -> None:
        """Let the state catch up to the cursor, never to the live edge.

        This is what keeps the system from announcing a goal the viewer has
        not been shown. The evidence arrives early; the belief arrives on time.
        """
        cursor = self.buffer.cursor_ts
        if cursor is None:
            return
        due = [c for c in self._board_changes if c.ts <= cursor]
        if not due:
            return
        self._board_changes = [c for c in self._board_changes if c.ts > cursor]
        # Stamped with the cursor, not with the change's own ts. "The score
        # moved recently" is a fact about the viewer's scoreboard, and the
        # viewer's scoreboard moves when the state takes the change in. The
        # two are the same moment when confirmation is prompt and a long way
        # apart when it is not: a change first seen at 62.7 and confirmed
        # only when the bug came back after the replay at 136 is news at 136,
        # and the lines about the goal come after that, not after 62.7.
        if any(c.is_goal for c in due):
            self._last_goal_ts = cursor
            self._restart_ts = None
        for change in due:
            if change.is_goal and change.scoring_side is not None:
                self.state.incidents.append(
                    Incident(
                        event=Event.GOAL,
                        side=change.scoring_side,
                        player=None,
                        video_ts=change.ts,
                        source="board",
                    )
                )
        self.state_tracker.apply_board(self.board_tracker)
        self._publish(Topic.STATE, cursor, self.state)

    def _apply_due_wire(self) -> None:
        """Let the statistician catch up to the cursor, never to the live edge.

        Same rule as the board, and for the same reason: the feed knows
        things before the viewer has seen them, and commentary that used them
        would be describing a match nobody is watching yet.
        """
        cursor = self.buffer.cursor_ts
        if self._sync is None or cursor is None:
            return
        # Polled here rather than off the back of a board read. What the feed
        # has said is a fact about the live edge, which this loop knows
        # exactly and the board reader only samples every couple of seconds;
        # tying release to that sampling made a goal's arrival depend on when
        # the score bug was last glanced at, which is nothing to do with it.
        self._sync.poll(self.live_ts)
        for correction in self._sync.apply_due(cursor, self.state_tracker):
            self._publish(
                Topic.CORRECTION,
                correction.video_ts,
                what=correction.what,
                event=correction.event.event.value,
            )
            self._publish(Topic.STATE, cursor, self.state)

    # -- deciding to speak -----------------------------------------------

    async def _tick(self) -> None:
        interval = self.settings.predictor.tick_s
        while True:
            await asyncio.sleep(interval)
            self.stats.ticks += 1
            if not self.buffer.ready:
                continue
            triggers = self._drain_triggers()
            if not triggers:
                triggers = [Trigger.SCHEDULED]
            decision = self.predictor.decide(
                now_ts=self.cursor_ts,
                triggers=triggers,
                last_spoken_ts=self._last_spoken_video_ts,
                after_goal=self._said_a_goal,
                last_spoken_seconds=self._last_spoken_seconds,
            )
            self._publish(Topic.TRIGGER, self.cursor_ts, decision)
            if self._over_budget():
                continue

            # A lull belongs to the analyst, and it has to be offered one
            # first. The predictor's job is to never let the broadcast go
            # mute, so left alone it will always send the caller to fill a
            # silence — and the analyst, which by design only speaks into
            # silences, would never once get a turn.
            if self._is_a_lull(decision) and await self._maybe_analyst():
                continue
            if decision.should_call:
                await self._call(decision.triggers)

    def _is_a_lull(self, decision: SpeakDecision) -> bool:
        """Nothing has happened; the only reason to speak is that nobody has."""
        real = set(decision.triggers) - {Trigger.SCHEDULED, Trigger.SILENCE_PRESSURE}
        return not real

    async def _maybe_analyst(self) -> bool:
        """Offer the analyst a turn. Returns whether it took one."""
        if not self.with_analyst:
            return False
        cursor = self.cursor_ts
        silence = (
            float("inf")
            if self._last_spoken_video_ts is None
            else cursor - self._last_spoken_video_ts
        )
        allowed, reason = self.analyst.should_speak(
            silence_s=silence,
            last_analyst_ts=self._last_analyst_ts,
            now_ts=cursor,
            last_event=self._recent_event_within(6.0),
        )
        if not allowed:
            return False

        line = await self.analyst.call(self.buffer, self.state_tracker.summary(cursor), reason)
        if line is None:
            self._publish(Topic.ERROR, cursor, where="analyst", detail=self.analyst.last_reason)
            return False
        self._publish(Topic.ANALYST, cursor, line)
        if not line.speak or not line.line.strip():
            return False

        self.director.submit(
            Beat(
                id=next_beat_id("a"),
                voice=Voice.ANALYST,
                text=line.line,
                video_ts=cursor,
                created_ts=time.monotonic(),
                live_ts=self.live_ts,
                preemptable=True,
            )
        )
        self._last_analyst_ts = cursor
        self._mark_spoken(cursor, line.line)
        self.stats.analyst_calls += 1
        return True

    def _recent_event_within(self, seconds: float) -> Event | None:
        """Whatever just happened, while it is still 'just'."""
        if self._recent_event is None:
            return None
        event, ts = self._recent_event
        return event if self.cursor_ts - ts <= seconds else None

    def _over_budget(self) -> bool:
        spent = self.backend.total.cost_usd
        if spent < self.settings.cost.max_usd_per_match:
            return False
        if not self.stats.cost_stopped:
            self.stats.cost_stopped = True
            self._publish(Topic.STATUS, self.cursor_ts, reason="cost_cap", spent_usd=spent)
        return True

    async def _call(self, triggers: list[Trigger]) -> None:
        cursor = self.cursor_ts
        self.stats.caller_calls += 1
        line = await self.caller.call(
            self.buffer,
            self.state_tracker.summary(cursor),
            triggers,
            lookahead_until=self._next_cut_after(cursor),
        )
        if line is None:
            self._publish(Topic.ERROR, cursor, where="caller", detail=self.caller.last_reason)
            return

        self._publish(Topic.CALLER, cursor, line)
        self.state_tracker.apply_caller(line, cursor)
        self._note_restart(line, cursor)
        self._bind_sightings(line, cursor)
        if not line.speak or not line.line.strip():
            return

        # Seeing is done; speaking is a separate call. What the gate judges
        # is whatever is actually going to the speaker, so the phrased line
        # is checked against the roster and the scoreline exactly as the
        # caller's would have been. Nothing the phraser writes gets past a
        # check the caller's line had to pass.
        judged, excitement = await self._phrase(line, cursor)

        verdict = self.gate.judge(
            judged,
            self.state,
            self.pack,
            board_changed=self._board_supports_goal(cursor),
            wire_confirmed=self._wire_confirms_goal(cursor),
            # The narrow half of the three, passed on its own: a goal the
            # state already holds is a goal the score already counts, and the
            # gate's arithmetic needs to know that the number is settled
            # rather than arriving.
            goal_in_state=self._goal_already_in_the_state(cursor),
            carried=self._carried_name(line, cursor),
            at=cursor,
        )
        self._publish(Topic.GATE, cursor, verdict, event=line.event.value)
        if not verdict.passed:
            self.stats.gated_out += 1
            return

        self.caller.gate.accept(verdict.line)
        # What the line says outranks what the caller filed it under. Ronaldo's
        # free kick was called correctly — "curls it over the wall and into the
        # top corner" — tagged `free_kick`, and dropped by the director on the
        # camera cut that every broadcaster makes the instant a goal goes in.
        # A goal is a goal whatever put the ball there.
        event = Event.GOAL if claims_goal(verdict.line, line.event) else line.event
        self._said_a_goal = event is Event.GOAL
        beat = Beat(
            id=next_beat_id(),
            voice=Voice.CALLER,
            text=verdict.line,
            video_ts=cursor,
            created_ts=time.monotonic(),
            live_ts=self.live_ts,
            event=event,
            excitement=excitement,
            triggers=triggers,
            preemptable=event not in (Event.GOAL, Event.PENALTY),
        )
        self.director.submit(beat)
        if self.phraser is not None:
            self.phraser.accept(verdict.line)
        self._remember_on_the_ball(line, verdict.line, cursor)
        self._mark_spoken(cursor, verdict.line)
        if line.event is not Event.NONE:
            self._recent_event = (line.event, cursor)
        self.stats.spoken += 1
        self._publish(Topic.COST, cursor, total_usd=round(self.backend.total.cost_usd, 4))

    async def _phrase(self, line: CallerLine, cursor: float) -> tuple[CallerLine, float]:
        """Say the caller's form the way a commentator would, or keep its words.

        Returns the form the gate should judge and the excitement to hang on
        the beat. With the stage off, or when it fails, that is the caller's
        own line untouched: a line the caller wrote and the gate has yet to
        see is worth more spoken badly than not spoken at all, so a phraser
        that errors or comes back empty is an error row on the bus and never
        a silent drop.
        """
        if self.phraser is None:
            return line, 0.0
        phrased = await self.phraser.phrase(
            line,
            self.state_tracker.summary(cursor),
            on_the_ball=self._carried_name(line, cursor),
        )
        if phrased is None or not phrased.line.strip():
            self._publish(
                Topic.ERROR,
                cursor,
                where="phraser",
                detail=self.phraser.last_reason or "the phraser returned nothing",
            )
            return line, 0.0
        self._publish(
            Topic.PHRASED,
            cursor,
            original=line.line,
            line=phrased.line,
            excitement=phrased.excitement,
        )
        return line.model_copy(update={"line": phrased.line}), phrased.excitement

    def _mark_spoken(self, cursor: float, text: str) -> None:
        """Remember when the voice was last given a line, and how long a one.

        The length is the speaker's own arithmetic — words over
        ``WORDS_PER_SECOND`` — rather than the ``seconds`` the director
        records once the line has been said. It has to be: the next tick can
        come half a second after the beat is submitted, before a word of it
        has been spoken, and a cap that waits for the measurement would spend
        every short line's window using the previous line's length.
        """
        self._last_spoken_video_ts = cursor
        words = len(text.split())
        self._last_spoken_seconds = words / WORDS_PER_SECOND if words else None

    def _next_cut_after(self, cursor: float) -> float | None:
        """Where the near future stops being the same passage of play.

        A broadcast cuts away every few seconds, and everything past the cut
        belongs to a different picture: a replay, the bench, a face in the
        crowd. Those frames are not what happens next, so the caller does not
        get to see them as though they were.
        """
        ahead = [ts for ts in self._cuts if ts > cursor]
        return min(ahead) if ahead else None

    def _board_supports_goal(self, cursor: float) -> bool:
        """Is there anything on the scoreboard behind a goal claimed here?

        Three things count, and they answer three different questions the
        first real run asked in its first three minutes.
        """
        return (
            self._board_changed_near(cursor)
            or self._board_pending_near(cursor)
            or self._goal_already_in_the_state(cursor)
        )

    def _board_pending_near(self, cursor: float) -> bool:
        """Is the board *in the middle of* agreeing with this claim?

        The tracker needs three agreeing reads and they land every three and
        a half seconds, so on the real run a goal the bug showed at 63.7 was
        not confirmed until 70.0 — thirteen seconds after the ball crossed
        the line, which no delay we would run at covers.

        One board read agreeing with a caller that has independently claimed
        a goal is two sources, and that is what the gate wants. It is not
        enough to move the score: the state still waits for three reads. The
        difference matters because a single misread digit that corroborates
        nothing decays away, while one that happens to line up with a real
        claim was probably not a misread.
        """
        pending = self.board_tracker.pending_goal
        if pending is None:
            return False
        window = min(GOAL_GRAPHIC_LAG_S, self.settings.capture.delay_s)
        return cursor - 2.0 <= pending.first_ts <= cursor + window

    def _bind_sightings(self, line: CallerLine, cursor: float) -> None:
        """Put the numbers and names the caller read into the registry.

        The team sheets are the check: a number has to be in that side's
        squad and a name has to be on the roster, because a hallucinated pair
        here would not just be one wrong line, it would be carried into the
        next one. What survives goes to the registry, which is what a later
        line reads a name back out of.
        """
        if not line.sightings:
            return
        seen: list[dict[str, Any]] = []
        for sighting in line.sightings:
            found = self._roster_check(sighting, cursor)
            row: dict[str, Any] = {
                "number": sighting.number,
                "name": sighting.name,
                "side": sighting.side.value,
                "bound": found is not None,
            }
            if found is not None:
                side, number, name = found
                self.state_tracker.registry.believe(number, name, cursor, side=side)
                row["as"] = f"{number} {name}"
                self.stats.sightings += 1
            else:
                self.stats.sightings_dropped += 1
            seen.append(row)
        self._publish(Topic.SIGHTING, cursor, sightings=seen)

    def _roster_check(self, sighting: Sighting, cursor: float) -> tuple[Side, int, str] | None:
        """The player this sighting is about, or None if it does not stand up.

        A name settles which side it is and which number goes with it, so a
        number given alongside has to agree.

        A number needs a side, because both squads wear a 5, a 7 and an 11.
        The side comes from the caller: it is looking at the kit and it has
        both kit strings in its team sheets, and a model reading the picture
        does not confuse white stripes with navy. It used to come from the
        kit split, which is the weakest link in the chain by its own
        docstring: on the real clip it put an Argentina body on France, and a
        sighting of "26" on it became Marcus Thuram in a passage Argentina
        played the whole of. A number with no side still names nobody — that
        is the case that was throwing away half of every read, and the answer
        to it is for the caller to say which kit, not for anything here to
        guess.
        """
        if self.pack is None:
            return None
        if sighting.name:
            found = _player_named(self.pack, sighting.name)
            if found is None:
                return None
            side, player = found
            if player.number is None:
                return None
            if sighting.number is not None and sighting.number != player.number:
                return None
            if sighting.side is not Side.UNKNOWN and sighting.side is not side:
                # The name and the kit disagree, so one of them was misread
                # and there is no way to tell which.
                return None
            return side, player.number, player.name
        if sighting.number is None:
            return None
        if sighting.side is not Side.UNKNOWN:
            wearer = _player_numbered(self.pack, sighting.side, sighting.number)
            return (sighting.side, sighting.number, wearer.name) if wearer else None
        wearing = [
            (side, wearer)
            for side in (Side.HOME, Side.AWAY)
            if (wearer := _player_numbered(self.pack, side, sighting.number)) is not None
        ]
        if len(wearing) != 1:
            return None
        one_side, wearer = wearing[0]
        return one_side, sighting.number, wearer.name

    def _remember_on_the_ball(self, line: CallerLine, spoken: str, cursor: float) -> None:
        """Record who this line was about, so the next one may keep the name.

        The name has to be in the line that was actually spoken: a sighting
        the caller reported and did not say is not what the commentary was
        about, and carrying it would be inventing a subject rather than
        keeping one.
        """
        registry = self.state_tracker.registry
        dead = line.event in (Event.PENALTY, Event.FREE_KICK, Event.CORNER, Event.THROW_IN)
        for sighting in line.sightings:
            name = sighting.name
            if not name:
                continue
            if fold(name.rsplit(" ", 1)[-1]) in fold(spoken):
                registry.name_on_the_ball(name, cursor, dead_ball=dead)
                self._carry_side = sighting.side
                return

    def _carried_name(self, line: CallerLine, cursor: float) -> str | None:
        """The name the last line had on the ball, if this line may keep it.

        One case only. The last line named the player on the ball within
        :data:`CARRY_NAME_S`, the phase of play has not changed — no restart
        and no change of possession — and this line is about the same run or
        shot. Then the name is still the name of the man being described, and
        the caller does not have to read a number that is facing away.

        The dead-ball case was specified and is not implemented. On the
        Netherlands clip the one name the caller had at a penalty was de Jong
        and the taker was van Dijk: a sighting says "I read this number on
        somebody in this picture", never "this is the man on the ball".
        """
        registry = self.state_tracker.registry
        held = registry.on_the_ball
        if held is None or held.dead_ball:
            return None
        if line.event in (Event.KICKOFF, Event.THROW_IN, Event.CORNER, Event.FREE_KICK):
            return None
        if line.side is not Side.UNKNOWN and self._carry_side not in (Side.UNKNOWN, line.side):
            return None
        return registry.carried_name(cursor, window_s=CARRY_NAME_S)

    def _note_restart(self, line: CallerLine, cursor: float) -> None:
        """Has the game gone again since the goal the state is holding?

        The caller writing a kickoff. There used to be a second way, a
        referee's whistle heard since the goal and then a live picture, and
        on real broadcast the whistle detector fired once in 58 runs, so it
        never was a way. Past the cap the talk ends anyway.

        A kickoff does not count while the score bug is away or the board
        reader thinks we are in a replay. That is the broadcaster's own answer to "has the
        game started again", it does not depend on the caller getting the
        scene right, and the caller does not: it wrote a kickoff over a replay
        of the goal.
        """
        if self._last_goal_ts is None or self._restart_ts is not None:
            return
        if not self.state.bug_visible or self.state.in_replay:
            # The broadcaster pulls the score bug for the replays and brings
            # it back when the game does. Until it is back we are watching
            # the goal, whatever the caller calls the scene — on the second
            # real run it wrote "kickoff, live play" over a replay at cursor
            # 89.5 and ended goal talk sixty seconds early, which cost the
            # celebration lines that followed.
            return
        if line.event is Event.KICKOFF:
            self._restart_ts = cursor

    def _goal_already_in_the_state(self, cursor: float) -> bool:
        """Is this a line about a goal the state has already taken in?

        The celebration, the replay, the scorer's face, the walk back: all of
        them are lines about a goal, and none of them is near the moment the
        board moved. Asking only whether the board moved *at the cursor*
        rejected four correct lines about one goal over eighty seconds.

        It runs until the game does. A fixed window cannot be the answer
        because a broadcaster's celebration is not a fixed length — the
        second real run lost two correct lines to a 45-second one while the
        picture was still on the scorer — and the thing that ends it is not
        a clock but the ball being kicked off again. The cap is for the
        restart nobody saw.
        """
        if self._last_goal_ts is None:
            return False
        since = cursor - self._last_goal_ts
        # A line from just before the state caught up is a line about the same
        # goal: the caller watched the ball cross the line and the graphic
        # followed it, which is the whole of what GOAL_GRAPHIC_LAG_S measures.
        # The third run lost a correct goal call at cursor 59.3 to a state
        # that applied the board at 62.2.
        if not -GOAL_GRAPHIC_LAG_S <= since <= GOAL_TALK_CAP_S:
            return False
        return self._restart_ts is None or cursor < self._restart_ts

    def _board_changed_near(self, cursor: float) -> bool:
        """Did the scoreboard move around the moment being called?

        A goal shows on the graphic a beat after the ball crosses the line, so
        the change sits slightly ahead of the cursor, and seeing it early is
        what the delay buys. But the window is a fact about broadcast
        graphics, not about our buffer: it must not widen just because we
        chose to wait longer.

        The first version used ``delay_s`` as the window, and so made the gate
        more permissive the deeper the buffer got — at eight seconds it would
        accept a board change eight seconds after the cursor as proof of a
        goal being called now, which is often a different passage of play
        entirely. Measured across the sweep, phantom goals reaching air went
        1, 1, 4, 6 as the buffer deepened. The delay was buying the caller
        information and paying for it by loosening the gate, which is most of
        why the delay chart showed nothing.
        """
        window = min(GOAL_GRAPHIC_LAG_S, self.settings.capture.delay_s)
        recent = [c for c in self._board_changes if c.is_goal]
        return any(cursor - 2.0 <= c.ts <= cursor + window for c in recent)

    def _wire_confirms_goal(self, cursor: float) -> bool:
        """Has the statistician said a goal went in around this moment?

        Known rather than applied: the feed's own latency is the thing being
        modelled, and a goal it has reported is evidence from the instant it
        reports it even though the state waits for the cursor.
        """
        if self._sync is None:
            return False
        return any(
            event.event is Event.GOAL
            and event.video_ts is not None
            and cursor - GOAL_GRAPHIC_LAG_S <= event.video_ts <= cursor + 2.0
            for event in self._sync.known
        )

    # -- plumbing --------------------------------------------------------

    def _fire(self, trigger: Trigger) -> None:
        """Park a trigger until the next tick reads them all at once.

        Triggers arrive on the frame loop, which runs far faster than the
        system can speak. Collecting them and letting the tick decide is what
        keeps a burst of three cuts from becoming three separate attempts to
        say something.
        """
        self._pending.append(trigger)

    def _drain_triggers(self) -> list[Trigger]:
        drained = list(dict.fromkeys(self._pending))
        self._pending.clear()
        return drained

    def _publish(self, topic: Topic, ts: float, value: Any = None, **extra: Any) -> None:
        self.bus.publish(topic, ts, value, **extra)

    async def _record(self) -> None:
        """Write everything on the bus to the trace.

        The trace subscribes rather than being written to directly, because
        the director publishes on its own — a line that was spoken, or cut off
        mid-word, is the director's news, not the runtime's. Anything written
        on a second path is a thing the eval would silently never see.
        """
        if self.trace is None:
            return
        async for message in self.bus.subscribe():
            self.trace.write(message)


def trace_path(root: Path, name: str) -> Path:
    return root / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
