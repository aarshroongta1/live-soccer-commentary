"""Watch a run that already happened, for nothing.

Every run leaves a trace: one JSON object a line, each one exactly what the
bus published at the moment it happened. So a finished run is already a
complete recording of the broadcast the web page showed — everything except
the picture, and the picture is the clip, still on disk.

This module puts the two back together. The clip goes through the same
:class:`FileCapture` and :class:`DelayBuffer` a live run uses, so ``/api/video``
serves frames behind the narration cursor exactly as it did on the night; each
trace row is republished onto a fresh :class:`Bus` when the cursor reaches the
point the run published it at. The page cannot tell the difference, and neither
can anything else that speaks to a
:class:`~commentary.server.RuntimeHandle`.

"The point the run published it at" is not always the row's own timestamp, and
:func:`cues` is where that is worked out. It matters more than it looks: the
picture is now held ``present_offset_s`` behind the cursor to meet the model's
round trip, so a replay that fired every line at the moment it describes would
show each one several seconds before the play it is about.

No model is called and no key is needed. That is the point: a run costs real
money and can be watched once, live, by whoever happened to be at the screen.
Replayed it costs nothing and can be watched by anybody, as often as it takes
to work out why the gate rejected the line about the penalty.

Give it a speaker and it is also the only free way to *hear* a change. The
trace's ``beat`` rows go to a real :class:`~commentary.director.Director`
instead of being republished, so the queueing, the ageing-out and the
mid-word cut on a goal all happen again, now, against whatever voice is being
tried — on lines a model was already paid for once. What the trace says was
spoken is then dropped: it is last time's account of the speaking, and this
time is different by construction.

``start_s`` is the offset into the clip the original run began at. A
``--source file`` run starts at zero and needs nothing; the screen runs played
a clip in a video player and started capturing part-way in, so their traces
count from that instant and the clip has to be seeked to meet them. Frames are
stamped from the seek, which is what keeps the trace's own timestamps — the
ones in the transcript and in every panel — the clock the whole page reads.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from commentary.bus import Bus, Message, Topic
from commentary.capture import DelayBuffer, FileCapture, FrameSource
from commentary.config import SETTINGS, Settings
from commentary.director import Director
from commentary.llm.base import Usage
from commentary.schemas import Beat, MatchState
from commentary.trace import read_trace
from commentary.voice.speaker import Speaker

#: The three topics that are the director's own account of the speaking. With
#: a voice attached the speaking happens again, so the trace's copies are
#: dropped and the live director's are published instead.
SPEAKING = (Topic.BEAT, Topic.SPOKEN, Topic.PREEMPTED)

#: How long to let the queue empty after the clip ends, when a voice is on.
#: The last thing a run said is usually the thing somebody put the clip on to
#: hear, and cutting it at the final frame would be a worse lie than a second
#: of black.
VOICE_DRAIN_S = 8.0


@dataclass(frozen=True)
class Cue:
    """One trace row, waiting for the cursor to reach it."""

    ts: float
    topic: Topic
    payload: dict[str, Any]
    #: Cursor time the original run actually published this row at. Usually
    #: ``ts``; see :func:`cues` for the rows where it is not.
    at: float = 0.0

    def message(self) -> Message:
        return Message(topic=self.topic, ts=self.ts, payload=self.payload)


def cues(rows: list[dict[str, Any]], *, delay_s: float = 0.0) -> list[Cue]:
    """Trace rows as cues, in the order the cursor will reach them.

    Sorted rather than left in file order, because a trace is written in
    arrival order and the board reader runs a buffer's length ahead of
    everything else: the first line of a real trace is a board read stamped
    two seconds in, above a trigger stamped at a quarter of a second.
    Replaying file order would put the match slightly out of sequence for no
    reason. The sort is stable, so rows scheduled together keep the order the
    run produced them in.

    **A row is not always published at its own timestamp.** A ``beat`` is
    stamped ``video_ts`` — the cursor when the caller's call *began* — but it
    only exists once the model answers, a measured median 3.4 s later, and
    that is when it went out over the bus and reached the page. Those rows
    carry the live edge they were published at, so their cursor time is
    ``live_ts - delay_s``. Replaying them at ``ts`` would put every line on
    the page three seconds early, which the presentation offset then makes
    three seconds *earlier* still: the picture is held back to meet the line,
    so the line must arrive when it really did.

    Rows without a ``live_ts`` stay on ``ts``. A ``caller`` form, a ``gate``
    verdict, a ``sighting``: these are the panels' account of a moment rather
    than something a viewer hears, and the moment is what they are stamped
    with.

    The rule moves a ``status`` row backwards rather than forwards, and that
    is right too: the runtime stamps those with the live edge instead of the
    cursor, so ``live_ts - delay_s`` is the cursor time they went out at in
    exactly the same way.

    A row whose topic is not one the bus knows is dropped. Traces outlive the
    code that wrote them; ``tracks`` and ``gallery`` rows from before the
    player tracker was removed are in ``runs/`` to this day.
    """
    found: list[Cue] = []
    for row in rows:
        try:
            topic = Topic(row.get("topic", ""))
        except ValueError:
            continue
        payload = {key: value for key, value in row.items() if key not in ("topic", "ts")}
        ts = float(row.get("ts", 0.0))
        live_ts = payload.get("live_ts")
        at = float(live_ts) - delay_s if isinstance(live_ts, int | float) else ts
        found.append(Cue(ts=ts, topic=topic, payload=payload, at=at))
    found.sort(key=lambda cue: cue.at)
    return found


def _state_of(payload: dict[str, Any]) -> MatchState | None:
    """A ``state`` row back into the model, or None if it is not one any more."""
    try:
        return MatchState.model_validate(payload)
    except Exception:
        return None


def _beat_of(payload: dict[str, Any]) -> Beat | None:
    """A ``beat`` row back into the model a director can be handed.

    ``created_ts`` is restamped, and that is not tidying. It is a
    ``time.monotonic()`` reading from a process that exited weeks ago, and the
    director measures staleness against ``time.monotonic()`` now: left as
    written, every line in the file is hours past its age limit and would be
    dropped before it reached a voice. Everything else is kept, ``video_ts``
    and ``preemptable`` and the event above all — a goal has to still be a
    goal here, or the cut this whole design is about never happens.
    """
    try:
        return Beat.model_validate({**payload, "created_ts": time.monotonic()})
    except Exception:
        return None


@dataclass
class Replay:
    """A finished run, played back from its trace and its clip.

    Satisfies :class:`~commentary.server.RuntimeHandle`, so ``create_app``
    serves a replay without knowing it is one.
    """

    rows: list[dict[str, Any]]
    source: FrameSource
    settings: Settings = SETTINGS
    start_s: float = 0.0
    #: For the status line, so the page says which run it is showing.
    label: str = ""
    #: Once the last cue has played, seek the clip back to the start and play
    #: the trace again, rather than ending the process.
    loop: bool = False
    #: A voice to say the trace's lines again, or None to stay silent and
    #: republish what the run said the first time.
    speaker: Speaker | None = None

    bus: Bus = field(default_factory=Bus)
    #: The clip on disk, so the watch page can play it natively — the real
    #: frames at the real rate through a ``<video>`` element seeked to the
    #: cursor — instead of the delay buffer's re-encoded JPEG stream at the
    #: capture's 15 fps. The first person to watch a lap said the picture and
    #: the frame rate were poor, and on a replay every frame is on disk.
    clip_path: Path | None = None

    def __post_init__(self) -> None:
        cap = self.settings.capture
        # The director is built here rather than passed in so that a replay
        # with a voice is one argument at every level above this. It gets the
        # replay's own bus, which is what puts its lines on the page and in
        # the trace exactly where the original run's were.
        self.director = (
            None
            if self.speaker is None
            else Director(speaker=self.speaker, cfg=self.settings.director, bus=self.bus)
        )
        self.buffer = DelayBuffer(cap.fps, cap.delay_s, cap.history_s)
        self.cues = cues(self.rows, delay_s=cap.delay_s)
        self._played = 0
        self._usage = Usage()
        self._stop = asyncio.Event()
        self._pass = 1
        # The teams are known before the first state row the same way a live
        # runtime knows them before kickoff: they come off the team sheets,
        # not off the screen. Everything else — the score, the clock, the
        # replay flag — stays at its default until a state row says otherwise.
        first = next(
            (state for cue in self.cues if (state := self._teams_of(cue)) is not None), None
        )
        self._initial_state = first if first is not None else MatchState(home="Home", away="Away")
        self._state = self._initial_state

    @staticmethod
    def _teams_of(cue: Cue) -> MatchState | None:
        if cue.topic is not Topic.STATE:
            return None
        state = _state_of(cue.payload)
        return None if state is None else MatchState(home=state.home, away=state.away)

    # -- what the web layer is allowed to see ----------------------------

    @property
    def state(self) -> MatchState:
        return self._state

    @property
    def usage(self) -> Usage:
        return self._usage

    @property
    def present_offset_s(self) -> float:
        return self.settings.capture.present_offset_s

    def status(self) -> dict[str, Any]:
        return {
            "message": f"replaying {self.label}" if self.label else "replaying a saved run",
            "source": "replay",
            "start_s": self.start_s,
            "rows": len(self.cues),
            "replayed": self._played,
            "buffered_frames": len(self.buffer),
            "cursor_ts": self.buffer.cursor_ts,
            "live_ts": self.buffer.live_ts,
            # Where the cursor is in the clip's own seconds, and whether the
            # clip itself is on offer, so the page can play the file natively
            # and keep it seeked to the narration.
            "clip_ts": self.clip_ts,
            "clip": self.clip_path is not None,
            # Said out loud so that a page showing lines nobody can hear, and
            # a page showing lines coming out of the speakers right now, are
            # not the same page.
            "speaking": self.director is not None,
        }

    # -- the clock -------------------------------------------------------

    @property
    def cursor_ts(self) -> float:
        return self.buffer.cursor_ts or 0.0

    @property
    def clip_ts(self) -> float | None:
        """Where in the clip the narration cursor is, in the clip's own seconds.

        The trace counts from the moment the run started; the clip counts
        from its first frame. ``start_s`` is the whole of the difference.
        """
        cursor = self.buffer.cursor_ts
        return None if cursor is None else cursor + self.start_s

    @property
    def played(self) -> int:
        """Rows republished so far."""
        return self._played

    @property
    def remaining(self) -> int:
        """Rows the cursor never reached. Non-zero means the clip ran short."""
        return len(self.cues) - self._played

    # -- playing it ------------------------------------------------------

    async def run(self, seconds: float | None = None) -> None:
        tasks = [asyncio.create_task(self._ingest(), name="frames")]
        if self.director is not None:
            tasks.append(asyncio.create_task(self.director.run(), name="director"))
        if seconds is not None:
            tasks.append(asyncio.create_task(self._deadline(seconds), name="deadline"))
        try:
            await self._stop.wait()
            await self._finish_speaking()
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            if self.speaker is not None:
                await self.speaker.aclose()

    async def _finish_speaking(self) -> None:
        """Let the voice finish before the clip's last frame ends everything.

        A live run drains at the final whistle for the same reason: the queue
        is almost never empty when the video stops, and a demo that cuts its
        own goal call off mid-word looks like the preemption misfiring rather
        than like the file running out.

        Only at the end. A ``--loop`` replay does not come through here at the
        seam between passes: the queue there is the tail of the pass that just
        finished, and the right thing to do with it is carry on speaking into
        the new one, exactly as the director would at any other moment.
        """
        director = self.director
        if director is None:
            return
        await director.drain(timeout=VOICE_DRAIN_S)
        director.stop()

    async def _deadline(self, seconds: float) -> None:
        await asyncio.sleep(seconds)
        self.stop()

    def stop(self) -> None:
        self._stop.set()

    async def _ingest(self) -> None:
        while True:
            async with self.source:
                async for frame in self.source.frames():
                    self.buffer.append(frame)
                    self.publish_due()
            # The clip ended. Anything left is a row the original run made
            # about video this clip does not contain; firing it now would put
            # it on the page at the wrong moment, so it is counted and
            # dropped, whether or not another pass follows.
            if not self.loop:
                break
            self._restart_pass()
        self.stop()

    def _restart_pass(self) -> None:
        """Seek the clip back to the start and play the trace again.

        The Bus and the server keep running; only the source is reopened
        (the next ``async with self.source`` in ``_ingest`` calls its
        ``__aenter__`` again) and the buffer and the replayed state go back
        to what they were before the first frame of the first pass.

        A voice needs nothing said to it here, and that is the point of
        winding ``_played`` back rather than tracking passes anywhere else:
        every cue is emitted again, so every ``beat`` is submitted to the
        director again, with its ``created_ts`` stamped at this pass rather
        than the last one. A looping demo therefore says the lines out loud
        on every lap, not only the first.
        """
        self._pass += 1
        self._played = 0
        self.buffer.clear()
        self._state = self._initial_state
        self._usage = Usage()
        print(f"pass {self._pass}: {len(self.cues)} rows")

    def publish_due(self) -> None:
        """Publish every cue the cursor has reached, in order, once each."""
        cursor = self.buffer.cursor_ts
        if cursor is None:
            return
        while self._played < len(self.cues) and self.cues[self._played].at <= cursor:
            cue = self.cues[self._played]
            self._played += 1
            self._apply(cue)
            self._emit(cue)

    def _emit(self, cue: Cue) -> None:
        """Put one cue back out — through the voice, if there is one.

        Silent, this is a straight republish: the trace holds what the run
        said and when, and that is exactly what the page wants.

        With a voice it is not, because the speaking is happening again. A
        ``beat`` is handed to the director, which publishes its own ``beat``
        as it queues it and its own ``spoken`` or ``preempted`` when it finds
        out how the line actually went. The trace's three copies are dropped
        rather than published alongside, or the page would be showing two
        accounts of the same line at once — and only one of them would be
        about the audio in the room.
        """
        if self.director is None or cue.topic not in SPEAKING:
            self.bus.publish(cue.topic, cue.ts, **cue.payload)
            return
        if cue.topic is Topic.BEAT:
            beat = _beat_of(cue.payload)
            if beat is not None:
                self.director.submit(beat)
                return
            # A row too old or too broken to be a Beat any more. Putting it
            # back on the bus unspoken is better than losing the line: the
            # transcript panel still shows it, with nothing claiming it was
            # heard.
            self.bus.publish(cue.topic, cue.ts, **cue.payload)

    def _apply(self, cue: Cue) -> None:
        """Rebuild what the snapshot endpoint answers with, row by row.

        ``/api/state`` is what the page reads before its first event arrives,
        so it has to say what the run believed at this instant and not what
        it believed at the end.
        """
        if cue.topic is Topic.STATE:
            state = _state_of(cue.payload)
            if state is not None:
                self._state = state
        elif cue.topic is Topic.COST:
            total = cue.payload.get("total_usd")
            if isinstance(total, int | float):
                # Only the total was ever published; the token counts stayed
                # in the backend and are not in the trace to recover.
                self._usage = Usage(cost_usd=float(total))


def from_files(
    trace: str | Path,
    path: str | Path,
    *,
    start_s: float = 0.0,
    settings: Settings = SETTINGS,
    loop: bool = False,
    speaker: Speaker | None = None,
) -> Replay:
    """A replay of one trace against one clip, ready to run or to serve."""
    trace_path, clip = Path(trace), Path(path)
    rows = read_trace(trace_path)
    source = FileCapture(clip, settings.capture, start_s=start_s)
    return Replay(
        rows=rows,
        source=source,
        settings=settings,
        start_s=start_s,
        label=f"{trace_path.name} over {clip.name}",
        loop=loop,
        speaker=speaker,
        clip_path=clip,
    )
