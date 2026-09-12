"""The wire: what the feed has said, and when the cursor may hear it."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from commentary.schemas import Event, GroundTruthEvent, Side, WireEvent
from commentary.wire import ReplayWire, WireSync


def event(
    kind: Event = Event.PASS,
    *,
    clock_s: float = 0.0,
    period: int = 1,
    video_ts: float | None = None,
    player: str = "Lionel Messi",
) -> WireEvent:
    return WireEvent(
        event=kind,
        side=Side.HOME,
        player=player,
        clock_s=clock_s,
        period=period,
        video_ts=video_ts,
    )


@dataclass
class FakeTracker:
    """Match state as far as the wire can see it.

    Answers with a description for the things a trace should show and with
    nothing for possession, which is how the real tracker behaves.
    """

    applied: list[tuple[float, Event]] = field(default_factory=list)

    def apply_wire(self, wire_event: WireEvent, ts: float) -> str | None:
        self.applied.append((ts, wire_event.event))
        if wire_event.event is Event.PASS:
            return None
        return f"{wire_event.event.value} {wire_event.player}"


def test_the_feed_says_each_thing_once_and_in_order():
    wire = ReplayWire(
        [
            event(Event.GOAL, video_ts=30.0),
            event(Event.CARD, video_ts=10.0),
            event(Event.PASS, video_ts=20.0),
        ],
        latency_s=5.0,
    )
    assert [e.video_ts for e in wire.due(25.0)] == [10.0, 20.0]
    assert wire.due(25.0) == []
    assert [e.video_ts for e in wire.due(40.0)] == [30.0]


def test_an_event_not_yet_on_the_video_clock_is_never_released():
    wire = ReplayWire([event(Event.GOAL, clock_s=600.0)], latency_s=0.0)
    assert wire.due(100_000.0) == []


def landing_cursor(latency_s: float, delay_s: float) -> float:
    """The cursor time at which a wire goal at video 100 lands, ticking live."""
    sync = WireSync(ReplayWire([event(Event.GOAL, clock_s=100.0, video_ts=100.0)], latency_s))
    tracker = FakeTracker()
    live = 100.0
    while live < 200.0:
        sync.poll(live)
        if sync.apply_due(live - delay_s, tracker):
            return live - delay_s
        live += 0.1
    raise AssertionError("the wire never landed")


@pytest.mark.parametrize(
    ("latency_s", "delay_s", "expected"),
    [(2.0, 4.0, 100.0), (4.0, 4.0, 100.0), (10.0, 4.0, 106.0)],
)
def test_the_buffer_absorbs_the_feeds_latency_up_to_the_delay(
    latency_s: float, delay_s: float, expected: float
):
    # A correction lands at video_ts + max(0, latency_s - delay_s): under the
    # delay the statistician is right at the cursor, over it they are stale.
    assert landing_cursor(latency_s, delay_s) == pytest.approx(expected, abs=0.11)


def test_one_misread_board_clock_does_not_move_every_touch():
    touch = event(Event.PASS, clock_s=600.0)
    sync = WireSync(ReplayWire([touch], latency_s=0.0))
    for i in range(9):
        clock = 100.0 + i * 2
        sync.observe_clock(clock, 1, clock + 20.0)
    # "07:12" read as "87:12": eighty minutes of offset in one reading.
    sync.observe_clock(432.0 + 4800.0, 1, 452.0)

    assert sync.offsets[1] == pytest.approx(20.0)
    assert touch.video_ts == pytest.approx(620.0)


def test_each_period_is_fitted_on_its_own():
    first, second = event(clock_s=600.0, period=1), event(clock_s=3000.0, period=2)
    sync = WireSync(ReplayWire([first, second], latency_s=0.0))
    for i in range(3):
        sync.observe_clock(100.0 + i, 1, 120.0 + i)
        sync.observe_clock(2800.0 + i, 2, 3700.0 + i)

    assert (sync.offsets[1], sync.offsets[2]) == (20.0, 900.0)
    assert (first.video_ts, second.video_ts) == (620.0, 3900.0)


def test_what_the_cursor_has_reached_is_applied_oldest_first():
    wire = ReplayWire(
        [
            event(Event.PASS, video_ts=10.0),
            event(Event.CARD, video_ts=7.0, player="Adrien Rabiot"),
            event(Event.GOAL, video_ts=5.0, player="Ángel Di María"),
            event(Event.SAVE, video_ts=40.0),
        ],
        latency_s=0.0,
    )
    sync, tracker = WireSync(wire), FakeTracker()
    sync.poll(20.0)
    corrections = sync.apply_due(12.0, tracker)

    assert tracker.applied == [(5.0, Event.GOAL), (7.0, Event.CARD), (10.0, Event.PASS)]
    # The pass answered with nothing: possession is too frequent to trace.
    assert [(c.video_ts, c.what) for c in corrections] == [
        (5.0, "goal Ángel Di María"),
        (7.0, "card Adrien Rabiot"),
    ]


def test_nothing_the_cursor_has_not_reached_is_applied_yet():
    sync = WireSync(ReplayWire([event(Event.GOAL, video_ts=50.0)], latency_s=0.0))
    tracker = FakeTracker()
    sync.poll(60.0)

    assert sync.apply_due(40.0, tracker) == []
    assert [c.video_ts for c in sync.apply_due(55.0, tracker)] == [50.0]


def test_the_sims_own_truth_arrives_already_on_the_video_clock():
    truth = [
        GroundTruthEvent(video_ts=12.0, event=Event.GOAL, side=Side.HOME, player="Dot 9"),
        GroundTruthEvent(video_ts=30.0, event=Event.CARD, side=Side.AWAY, player="Dot 4"),
    ]
    wire = ReplayWire.from_truth(truth, latency_s=5.0)

    stamps = [(e.video_ts, e.clock_s, e.period) for e in wire.events]
    assert stamps == [(12.0, 12.0, 1), (30.0, 30.0, 1)]
    assert [e.player for e in wire.due(20.0)] == ["Dot 9"]
