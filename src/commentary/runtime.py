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
the state has already taken in as cover for the lines that follow one. What
the score itself is allowed to move on does not change: three reads.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from commentary.agents.analyst import Analyst
from commentary.agents.caller import Caller
from commentary.bus import Bus, Topic
from commentary.capture.audio import CutDetector, RoarDetector, WhistleDetector
from commentary.capture.buffer import AudioRing, DelayBuffer, Frame
from commentary.config import SETTINGS, Settings
from commentary.director import Director, next_beat_id
from commentary.gate import FactGate
from commentary.llm.base import LLMBackend, Usage
from commentary.perception.board import BoardChange, BoardReader, BoardTracker
from commentary.perception.players import NullTracker, Track, Tracker
from commentary.predictor import SpeakPredictor
from commentary.prompts.caller import MARK_TOLERANCE_S
from commentary.schemas import (
    Beat,
    BoardRead,
    Event,
    Incident,
    KnowledgePack,
    MatchState,
    SpeakDecision,
    Trigger,
    Voice,
)
from commentary.state import MatchStateTracker, parse_clock, period_for_clock
from commentary.tools import MatchTools
from commentary.trace import RunTrace
from commentary.voice.speaker import LogSpeaker, Speaker
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

#: Detection runs on every other frame. Fifteen frames a second is more than
#: identity needs — a body does not become a different body in 130 ms — and
#: halving the rate halves what the executor thread has to do.
TRACK_EVERY = 2

#: How long a goal stays a thing worth talking about. The celebration, the
#: replay, the scorer's face and the restart all belong to a goal the state
#: already holds, and on the same run four lines about one goal were rejected
#: as phantom goals over the eighty seconds after it because the only
#: question being asked was whether the board had moved *near the cursor*.
GOAL_TALK_WINDOW_S = 45.0


@dataclass
class RuntimeStats:
    frames: int = 0
    tracked_frames: int = 0
    audio_chunks: int = 0
    board_reads: int = 0
    caller_calls: int = 0
    analyst_calls: int = 0
    ticks: int = 0
    gated_out: int = 0
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
    #: Who is on the pitch, read off the picture. A :class:`NullTracker` by
    #: default: naming players from vision is opt-in, and the no-marks row of
    #: the ablation table is this same field with the null one in it.
    tracker: Tracker = field(default_factory=NullTracker)
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
        self.audio_ring = AudioRing()

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
            tracks_for=self.tracks_for,
        )
        self.analyst = Analyst(
            self.backend,
            config=self.settings.analyst,
            tools=MatchTools(state=self.state_tracker.state, pack=self.pack),
            pack=self.pack,
        )
        self.gate = FactGate(self.settings.gate)
        self.predictor = SpeakPredictor(self.settings.predictor, self.settings.caller)
        self.director = Director(speaker=self.speaker, cfg=self.settings.director, bus=self.bus)

        self.whistle = WhistleDetector(self.settings.predictor)
        self.roar = RoarDetector(self.settings.predictor)
        self.cut = CutDetector(self.settings.predictor)

        self._pending: list[Trigger] = []
        self._board_changes: list[BoardChange] = []
        self._roars: list[float] = []
        self._cuts: list[float] = []
        #: Tracks by the video time of the frame they were read from, bounded
        #: to the buffer's own reach: a mark is only ever drawn on a frame the
        #: caller is being shown, and those never come from further back.
        self._tracks: deque[tuple[float, list[Track]]] = deque(
            maxlen=int((cap.delay_s + cap.history_s) * cap.fps / TRACK_EVERY) + 1
        )
        self._last_goal_ts: float | None = None
        self._last_spoken_video_ts: float | None = None
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
            if hasattr(self.source, "audio"):
                tasks.append(asyncio.create_task(self._ingest_audio(), name="audio"))
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
                # A cut is a different camera on a different part of the
                # pitch, so no body in it continues a body from before.
                self.tracker.reset()
                self._fire(Trigger.CAMERA_CUT)
            await self._track(frame)
        # The source ran out: a clip ended, or the stream died. Either way the
        # match is over as far as this process is concerned.
        self.stop()

    async def _track(self, frame: Frame) -> None:
        """Find the bodies, off the event loop.

        Detection is tens of milliseconds of numpy and, with real weights, a
        neural network; run inline it would stall the frame, audio and
        caller loops that share this thread. In an executor it costs the
        ingest loop only the hop.
        """
        if self.stats.frames % TRACK_EVERY:
            return
        loop = asyncio.get_running_loop()
        tracks = await loop.run_in_executor(None, self.tracker.update, frame)
        self.stats.tracked_frames += 1
        if not tracks:
            return
        self._tracks.append((frame.ts, tracks))
        # The registry is state, and state has one writer. The tracker's job
        # ends at "that shirt says 11 and it is an Argentina shirt"; what
        # that is worth ten minutes later is the registry's decay to decide.
        for track in tracks:
            if track.number is not None and track.name is not None:
                self.state_tracker.registry.believe(
                    track.number, track.name, frame.ts, side=track.side
                )

    def tracks_for(self, ts: float) -> list[Track]:
        """Who was where when this frame was captured, for drawing on it.

        Nearest tracked frame rather than an exact hit, because detection
        runs on every other frame and the caller is shown whichever frames
        the buffer sampled. Past the tolerance the bodies have moved and a
        name would be drawn over the wrong player, which is the one failure
        this whole chain exists to avoid.
        """
        best: tuple[float, list[Track]] | None = None
        for tracked_ts, tracks in self._tracks:
            gap = abs(tracked_ts - ts)
            if gap <= MARK_TOLERANCE_S and (best is None or gap < best[0]):
                best = (gap, tracks)
        return [] if best is None else best[1]

    async def _ingest_audio(self) -> None:
        async for chunk in self.source.audio():
            self.audio_ring.append(chunk)
            self.stats.audio_chunks += 1
            if self.whistle.feed(chunk) is not None:
                self._fire(Trigger.WHISTLE)
            if (roar := self.roar.feed(chunk)) is not None:
                # Kept with its timestamp, not just as a trigger: the fact
                # gate needs to know WHEN the crowd went up, to decide
                # whether it corroborates a goal claimed at the cursor.
                self._roars.append(roar.ts)
                self._fire(Trigger.ROAR)

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
            self._observe_clock(read, frame.ts)
            change = self.board_tracker.update(read, frame.ts)
            if change is not None:
                self._board_changes.append(change)
                self._fire(Trigger.BOARD_CHANGE)
            self._note_screen()

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
        goals = [c.ts for c in due if c.is_goal]
        if goals:
            self._last_goal_ts = max(goals)
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
        self._last_spoken_video_ts = cursor
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
        if not line.speak or not line.line.strip():
            return

        verdict = self.gate.judge(
            line,
            self.state,
            self.pack,
            board_changed=self._board_supports_goal(cursor),
            lookahead_celebration=self._celebration_ahead(cursor),
            wire_confirmed=self._wire_confirms_goal(cursor),
        )
        self._publish(Topic.GATE, cursor, verdict, event=line.event.value)
        if not verdict.passed:
            self.stats.gated_out += 1
            return

        self.caller.gate.accept(verdict.line)
        beat = Beat(
            id=next_beat_id(),
            voice=Voice.CALLER,
            text=verdict.line,
            video_ts=cursor,
            created_ts=time.monotonic(),
            live_ts=self.live_ts,
            event=line.event,
            triggers=triggers,
            preemptable=line.event not in (Event.GOAL, Event.PENALTY),
        )
        self.director.submit(beat)
        self._last_spoken_video_ts = cursor
        if line.event is not Event.NONE:
            self._recent_event = (line.event, cursor)
        self.stats.spoken += 1
        self._publish(Topic.COST, cursor, total_usd=round(self.backend.total.cost_usd, 4))

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
        pending = self.board_tracker.pending
        if pending is None:
            return False
        home, away = self.board_tracker.home_score, self.board_tracker.away_score
        if home is None or away is None:
            return False  # the first board we ever settle on is not a goal
        if pending.home_score <= home and pending.away_score <= away:
            return False
        window = min(GOAL_GRAPHIC_LAG_S, self.settings.capture.delay_s)
        return cursor - 2.0 <= pending.first_ts <= cursor + window

    def _goal_already_in_the_state(self, cursor: float) -> bool:
        """Is this a line about a goal the state has already taken in?

        The celebration, the replay, the scorer's face, the restart: all of
        them are lines about a goal, and none of them is near the moment the
        board moved. Asking only whether the board moved *at the cursor*
        rejected four correct lines about one goal over eighty seconds.
        """
        if self._last_goal_ts is None:
            return False
        return 0.0 <= cursor - self._last_goal_ts <= GOAL_TALK_WINDOW_S

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

    def _celebration_ahead(self, cursor: float) -> bool:
        """Is there a crowd celebration between the cursor and the live edge?

        Evidence has to come from something other than the model that is
        making the claim. The first version of this asked the caller how
        confident it felt and let anything above 0.8 through, which is not a
        second source at all — it is the same source with a number attached,
        and a confidently wrong model is precisely the failure the gate
        exists to stop. It let a goal be announced at a moment when no goal
        had happened.

        A sustained roar is independent: it comes off the audio, which the
        caller never sees. But it is weak evidence, because crowds roar at
        near misses too, and a run with this as a free-standing second route
        to a goal let a phantom one through on exactly that. So the roar only
        counts when the board cannot be read at all — a replay, a graphic
        over the bug, a broadcaster who has hidden it. When the board is
        legible it is the only thing that confirms a goal, because it is the
        only source that is actually about the score.
        """
        if not self.board_tracker.in_replay and self.board_tracker.home_score is not None:
            return False
        window = self.settings.capture.delay_s
        return any(cursor - 1.0 <= ts <= cursor + window for ts in self._roars)

    # -- plumbing --------------------------------------------------------

    def _fire(self, trigger: Trigger) -> None:
        """Park a trigger until the next tick reads them all at once.

        Triggers arrive on the frame and audio loops, which run far faster
        than the system can speak. Collecting them and letting the tick decide
        is what keeps a burst of three cuts and a roar from becoming four
        separate attempts to say something.
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
