"""The play-by-play wire: a statistician in the ear, off by default.

A feed knows everything the picture does not — who passed to whom, who was
fouled, whose card it is — and it knows it in match time, several seconds
after it happened. Both of those are the whole difficulty. The thesis of
this project is that the picture, the sound and notes prepared before
kickoff are enough; the wire exists so the writeup can show what a feed
would have bought, as the ceiling row of the ablation table and never as
part of the default runtime.

Two things live here. :class:`ReplayWire` is a saved match released as if
it were arriving live, delayed by a modelled feed latency. :class:`WireSync`
puts those events on the video clock using the board reader's own clock
readings, and decides when each one may be applied.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Protocol

from commentary.schemas import GroundTruthEvent, WireEvent

#: Board readings kept per period before the median is taken. Ten and a
#: median rather than one reading: passes are two seconds apart and the
#: board reader misreads a digit now and then, and a single bad read must
#: not rename every touch for the next two seconds.
READS_PER_PERIOD = 10


class Wire(Protocol):
    """A source of play-by-play events, arriving late."""

    latency_s: float
    #: The sync resolves ``video_ts`` on these in place; an event it has not
    #: reached yet is not releasable, so the two must be the same objects.
    events: list[WireEvent]

    def due(self, live_ts: float) -> list[WireEvent]: ...


class Tracker(Protocol):
    """The part of match state the wire is allowed to move.

    Stated structurally so this module does not import state: the dependency
    runs the other way, and a test can hand in anything with this method.
    """

    def apply_wire(self, event: WireEvent, ts: float) -> str | None: ...


@dataclass(frozen=True)
class Correction:
    """One thing the wire changed that the trace should show."""

    video_ts: float
    what: str
    event: WireEvent


class ReplayWire:
    """A saved match's events, released once each as the clock reaches them.

    ``latency_s`` is how long the modelled feed takes to say anything: an
    event is released when the live edge has passed ``video_ts + latency_s``,
    which is where a real statistician's lag goes.
    """

    def __init__(self, events: list[WireEvent], latency_s: float) -> None:
        self.events = list(events)
        self.latency_s = latency_s
        self._sent: set[int] = set()

    @classmethod
    def from_truth(cls, events: list[GroundTruthEvent], latency_s: float) -> ReplayWire:
        """A wire over the simulator's ground truth, which is already on video time.

        The sim has no passes, so this wire carries goals, cards, subs and
        whatever else the sim scripted — enough for the ablation row to run
        offline, not enough to say anything about possession.
        """
        return cls(
            [
                WireEvent(
                    event=truth.event,
                    side=truth.side,
                    player=truth.player,
                    home_score=truth.home_score,
                    away_score=truth.away_score,
                    clock_s=truth.video_ts,
                    period=1,
                    video_ts=truth.video_ts,
                )
                for truth in events
            ],
            latency_s,
        )

    def due(self, live_ts: float) -> list[WireEvent]:
        """Everything the feed has said by ``live_ts`` and has not said before."""
        ready: list[tuple[float, WireEvent]] = []
        for index, event in enumerate(self.events):
            ts = event.video_ts
            if index in self._sent or ts is None or ts + self.latency_s > live_ts:
                continue
            self._sent.add(index)
            ready.append((ts, event))
        ready.sort(key=lambda pair: pair[0])
        return [event for _, event in ready]


class WireSync:
    """The wire on the video clock, and the two clocks that govern it.

    An event is *known* when the live edge passes ``video_ts + latency_s``,
    and *applied* when the cursor — the live edge less the buffer delay —
    passes ``video_ts``. So a correction lands at cursor time::

        video_ts + max(0, latency_s - delay_s)

    With ``latency_s <= delay_s`` the buffer absorbs the feed's latency
    entirely and the statistician is right at the cursor, telling the caller
    who has the ball in the frame it is looking at. At ``delay_s = 0`` the
    same wire is ``latency_s`` stale and names the player who had the ball
    ten seconds ago. That is the argument for running behind the live edge.

    The events themselves arrive on the match clock, which is the only one a
    feed knows. :meth:`observe_clock` is what bridges them, from the board
    reader's readings.
    """

    def __init__(self, wire: Wire) -> None:
        self.wire = wire
        self.offsets: dict[int, float] = {}
        #: Everything the feed has said, whether or not the cursor has reached
        #: it. The gate reads this: a goal the statistician has already called
        #: corroborates a caller claiming one, and it does so from the moment
        #: the feed says it rather than from the moment the state moves.
        self.known: list[WireEvent] = []
        #: Events that arrived already on the video clock — the simulator's
        #: own truth. The board's reading of a match clock cannot improve on
        #: a time that was never on a match clock to begin with, and applying
        #: an offset to one moves it by the whole kickoff offset.
        self._fixed = {id(event) for event in wire.events if event.video_ts is not None}
        self._reads: dict[int, list[float]] = {}
        self._queue: list[WireEvent] = []

    def observe_clock(self, clock_s: float, period: int, video_ts: float) -> None:
        """Take one board reading, and restamp that period's events with it.

        Each period is fitted on its own: the interval sits between them, so
        one offset would split the difference and be wrong in both.
        """
        reads = self._reads.setdefault(period, [])
        reads.append(video_ts - clock_s)
        del reads[:-READS_PER_PERIOD]
        offset = statistics.median(reads)
        self.offsets[period] = offset
        for event in self.wire.events:
            if event.period == period and id(event) not in self._fixed:
                event.video_ts = event.clock_s + offset

    def poll(self, live_ts: float) -> None:
        """Take whatever the feed has said by now; it waits for the cursor."""
        said = self.wire.due(live_ts)
        self.known.extend(said)
        self._queue.extend(said)

    def apply_due(self, cursor_ts: float, tracker: Tracker) -> list[Correction]:
        """Apply everything the cursor has reached, oldest first.

        A correction is returned for each event the tracker says changed
        something. Possession changes answer with nothing: they are two a
        second, and a trace row for each would bury the ones that matter.
        """
        due: list[tuple[float, WireEvent]] = []
        waiting: list[WireEvent] = []
        for event in self._queue:
            ts = event.video_ts
            if ts is None or ts > cursor_ts:
                waiting.append(event)
            else:
                due.append((ts, event))
        self._queue = waiting
        due.sort(key=lambda pair: pair[0])

        corrections: list[Correction] = []
        for ts, event in due:
            what = tracker.apply_wire(event, ts)
            if what is not None:
                corrections.append(Correction(video_ts=ts, what=what, event=event))
        return corrections
