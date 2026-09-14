"""Watch a run that already happened, for nothing.

Every run leaves a trace: one JSON object a line, each one exactly what the
bus published at the moment it happened. So a finished run is already a
complete recording of the broadcast the web page showed — everything except
the picture, and the picture is the clip, still on disk.

This module puts the two back together. The clip goes through the same
:class:`FileCapture` and :class:`DelayBuffer` a live run uses, so ``/api/video``
serves frames at the narration cursor exactly as it did on the night; each
trace row is republished onto a fresh :class:`Bus` when the cursor reaches its
timestamp. The page cannot tell the difference, and neither can anything else
that speaks to a :class:`~commentary.server.RuntimeHandle`.

No model is called and no key is needed. That is the point: a run costs real
money and can be watched once, live, by whoever happened to be at the screen.
Replayed it costs nothing and can be watched by anybody, as often as it takes
to work out why the gate rejected the line about the penalty.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from commentary.bus import Bus, Message, Topic
from commentary.capture import DelayBuffer, FileCapture, FrameSource
from commentary.config import SETTINGS, Settings
from commentary.llm.base import Usage
from commentary.schemas import MatchState
from commentary.trace import read_trace


@dataclass(frozen=True)
class Cue:
    """One trace row, waiting for the cursor to reach it."""

    ts: float
    topic: Topic
    payload: dict[str, Any]

    def message(self) -> Message:
        return Message(topic=self.topic, ts=self.ts, payload=self.payload)


def cues(rows: list[dict[str, Any]]) -> list[Cue]:
    """Trace rows as cues, oldest first.

    Sorted by ``ts`` rather than left in file order, because a trace is
    written in arrival order and the board reader runs a buffer's length
    ahead of everything else: the first line of a real trace is a board read
    stamped two seconds in, above a trigger stamped at a quarter of a second.
    Replaying file order would put the match slightly out of sequence for no
    reason. The sort is stable, so rows sharing a timestamp — a gate verdict
    and the beat it let through — keep the order the run produced them in.

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
        found.append(Cue(ts=float(row.get("ts", 0.0)), topic=topic, payload=payload))
    found.sort(key=lambda cue: cue.ts)
    return found


def _state_of(payload: dict[str, Any]) -> MatchState | None:
    """A ``state`` row back into the model, or None if it is not one any more."""
    try:
        return MatchState.model_validate(payload)
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

    bus: Bus = field(default_factory=Bus)

    def __post_init__(self) -> None:
        cap = self.settings.capture
        self.buffer = DelayBuffer(cap.fps, cap.delay_s, cap.history_s)
        self.cues = cues(self.rows)
        self._played = 0
        self._usage = Usage()
        self._stop = asyncio.Event()
        # The teams are known before the first state row the same way a live
        # runtime knows them before kickoff: they come off the team sheets,
        # not off the screen. Everything else — the score, the clock, the
        # replay flag — stays at its default until a state row says otherwise.
        first = next(
            (state for cue in self.cues if (state := self._teams_of(cue)) is not None), None
        )
        self._state = first if first is not None else MatchState(home="Home", away="Away")

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
        async with self.source:
            tasks = [asyncio.create_task(self._ingest(), name="frames")]
            if seconds is not None:
                tasks.append(asyncio.create_task(self._deadline(seconds), name="deadline"))
            try:
                await self._stop.wait()
            finally:
                for task in tasks:
                    task.cancel()
                for task in tasks:
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

    async def _deadline(self, seconds: float) -> None:
        await asyncio.sleep(seconds)
        self.stop()

    def stop(self) -> None:
        self._stop.set()

    async def _ingest(self) -> None:
        async for frame in self.source.frames():
            self.buffer.append(frame)
            self.publish_due()
        # The clip ended. Anything left is a row the original run made about
        # video this clip does not contain; firing it now would put it on the
        # page at the wrong moment, so it is counted and dropped.
        self.stop()

    def publish_due(self) -> None:
        """Publish every cue the cursor has reached, in order, once each."""
        cursor = self.buffer.cursor_ts
        if cursor is None:
            return
        while self._played < len(self.cues) and self.cues[self._played].ts <= cursor:
            cue = self.cues[self._played]
            self._played += 1
            self._apply(cue)
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
    )
