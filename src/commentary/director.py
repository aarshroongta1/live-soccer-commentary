"""Who speaks, when, and who gets cut off.

Two voices sharing one channel need someone deciding between them, and the
decision is mostly about time. A caller line about a shot is worthless four
seconds later; an analyst aside is worthless the instant a goal goes in. So
the director holds a shallow queue, drops anything that has aged out, keeps
the voices from talking over each other, and hard-preempts on the events
where a human producer would cut the mic.

This is the reason the project has no agent framework in it. LangGraph,
CrewAI and the rest are built on the assumption that an agent finishes its
turn. Here the most important thing the system does is stop one mid-sentence.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import time
from dataclasses import dataclass, field

from commentary.bus import Bus, Topic
from commentary.config import DirectorConfig
from commentary.schemas import Beat, Event, Voice
from commentary.voice.speaker import LogSpeaker, Speaker, Utterance

_ids = itertools.count(1)


def next_beat_id(prefix: str = "b") -> str:
    return f"{prefix}{next(_ids)}"


@dataclass
class DirectorStats:
    queued: int = 0
    spoken: int = 0
    preempted: int = 0
    dropped_stale: int = 0
    dropped_full: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "queued": self.queued,
            "spoken": self.spoken,
            "preempted": self.preempted,
            "dropped_stale": self.dropped_stale,
            "dropped_full": self.dropped_full,
        }


@dataclass
class Director:
    """A single speaking channel with preemption.

    ``clock`` is injectable so tests can run a whole match of scheduling in
    milliseconds without pretending that ``asyncio.sleep`` is precise.
    """

    speaker: Speaker = field(default_factory=LogSpeaker)
    cfg: DirectorConfig = field(default_factory=DirectorConfig)
    bus: Bus | None = None
    stats: DirectorStats = field(default_factory=DirectorStats)
    last_spoken_ts: float = 0.0
    last_voice: Voice | None = None
    _queue: list[Beat] = field(default_factory=list)
    _current: Beat | None = None
    _cancel: asyncio.Event = field(default_factory=asyncio.Event)
    _wake: asyncio.Event = field(default_factory=asyncio.Event)
    _running: bool = False

    # -- submission ------------------------------------------------------

    @property
    def preempting_events(self) -> frozenset[Event]:
        return frozenset(Event(name) for name in self.cfg.preempt_on)

    def submit(self, beat: Beat) -> bool:
        """Offer a beat. Returns whether it was accepted into the queue."""
        if beat.event in self.preempting_events:
            return self._submit_urgent(beat)
        if len(self._queue) >= self.cfg.queue_depth:
            # The queue being full means the system has more to say than time
            # to say it. Losing the oldest is right: it is the most stale.
            self._queue.pop(0)
            self.stats.dropped_full += 1
        self._queue.append(beat)
        self.stats.queued += 1
        self._publish(Topic.BEAT, beat)
        self._wake.set()
        return True

    def _submit_urgent(self, beat: Beat) -> bool:
        """A goal does not wait behind an aside about pressing triggers."""
        self._queue = [b for b in self._queue if not b.preemptable]
        self._queue.insert(0, beat)
        self.stats.queued += 1
        self._publish(Topic.BEAT, beat, urgent=True)
        if self._current is not None and self._current.preemptable:
            self._cancel.set()
        self._wake.set()
        return True

    # -- the loop --------------------------------------------------------

    async def run(self) -> None:
        self._running = True
        try:
            while self._running:
                beat = self._take()
                if beat is None:
                    await self._idle()
                    continue
                await self._speak(beat)
        finally:
            self._running = False

    async def _idle(self) -> None:
        self._wake.clear()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._wake.wait(), timeout=0.25)

    def _take(self) -> Beat | None:
        """The next beat still worth saying, discarding anything that aged out."""
        now = time.monotonic()
        while self._queue:
            beat = self._queue.pop(0)
            urgent = beat.event in self.preempting_events
            if not urgent and now - beat.created_ts > self.cfg.max_beat_age_s:
                self.stats.dropped_stale += 1
                self._publish(Topic.PREEMPTED, beat, reason="stale")
                continue
            return beat
        return None

    async def _speak(self, beat: Beat) -> None:
        self._current = beat
        self._cancel = asyncio.Event()
        try:
            utterance = await self.speaker.say(beat, self._cancel)
        finally:
            self._current = None
        self.last_spoken_ts = time.monotonic()
        self.last_voice = beat.voice
        if utterance.completed:
            self.stats.spoken += 1
            self._publish(Topic.SPOKEN, beat, spoken=utterance.spoken, seconds=utterance.seconds)
        else:
            self.stats.preempted += 1
            self._publish(
                Topic.PREEMPTED,
                beat,
                reason="cut",
                spoken=utterance.spoken,
                seconds=utterance.seconds,
            )

    async def drain(self, timeout: float = 5.0) -> None:
        """Wait for the queue to empty. Used by tests and at the final whistle."""
        deadline = time.monotonic() + timeout
        while (self._queue or self._current) and time.monotonic() < deadline:
            await asyncio.sleep(0.01)

    def stop(self) -> None:
        self._running = False
        self._cancel.set()
        self._wake.set()

    # -- introspection ---------------------------------------------------

    @property
    def busy(self) -> bool:
        return self._current is not None

    @property
    def pending(self) -> int:
        return len(self._queue)

    def silence_for(self, now: float | None = None) -> float:
        """How long nothing has been said, which is what builds speak pressure."""
        if self.busy:
            return 0.0
        return (now or time.monotonic()) - self.last_spoken_ts

    def _publish(self, topic: Topic, beat: Beat, **extra: object) -> None:
        if self.bus is not None:
            self.bus.publish(topic, beat.video_ts, beat, **extra)


def last_utterance(speaker: Speaker) -> Utterance | None:
    said = getattr(speaker, "said", None)
    return said[-1] if said else None
