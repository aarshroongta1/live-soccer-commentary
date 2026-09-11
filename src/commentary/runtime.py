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
from commentary.bus import Bus, Topic
from commentary.capture.audio import CutDetector, RoarDetector, WhistleDetector
from commentary.capture.buffer import AudioRing, DelayBuffer, Frame
from commentary.config import SETTINGS, Settings
from commentary.director import Director, next_beat_id
from commentary.gate import FactGate
from commentary.llm.base import LLMBackend, Usage
from commentary.perception.board import BoardChange, BoardReader, BoardTracker
from commentary.predictor import SpeakPredictor
from commentary.schemas import (
    Beat,
    Event,
    KnowledgePack,
    MatchState,
    SpeakDecision,
    Trigger,
    Voice,
)
from commentary.state import MatchStateTracker
from commentary.tools import MatchTools
from commentary.trace import RunTrace
from commentary.voice.speaker import LogSpeaker, Speaker

#: How long after the ball crosses the line a broadcaster's score bug
#: catches up. A property of television, not of our buffer, which is the
#: whole point: the window the gate will accept a board change in must not
#: grow when we choose to wait longer.
GOAL_GRAPHIC_LAG_S = 5.0


@dataclass
class RuntimeStats:
    frames: int = 0
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
        self.caller = Caller(self.backend, config=self.settings.caller, pack=self.pack)
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
        self._last_spoken_video_ts: float | None = None
        self._last_analyst_ts: float = 0.0
        self._recent_event: tuple[Event, float] | None = None
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
            if (cut := self.cut.feed(frame)) is not None:
                # Kept with its timestamp: a cut is both a reason to consider
                # speaking and the edge of what counts as "next" for the
                # caller, and only the second of those needs to know when.
                self._cuts.append(cut.ts)
                self._fire(Trigger.CAMERA_CUT)
        # The source ran out: a clip ended, or the stream died. Either way the
        # match is over as far as this process is concerned.
        self.stop()

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
            change = self.board_tracker.update(read, frame.ts)
            if change is not None:
                self._board_changes.append(change)
                self._fire(Trigger.BOARD_CHANGE)
            self._note_replay()

    def _live_frame(self) -> Frame | None:
        live = self.buffer.live_ts
        return self.buffer.nearest(live) if live is not None else None

    def _note_replay(self) -> None:
        if self.board_tracker.in_replay != self.state.in_replay:
            self.state_tracker.apply_board(self.board_tracker)
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
        self.state_tracker.apply_board(self.board_tracker)
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

        line = await self.analyst.call(self.buffer, self.state_tracker.summary(), reason)
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
            self.state_tracker.summary(),
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
            board_changed=self._board_changed_near(cursor),
            lookahead_celebration=self._celebration_ahead(cursor),
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
