import numpy as np
import pytest

from commentary.capture import DelayBuffer, Frame


def fill(buf: DelayBuffer, fps: int, seconds: float, start: float = 0.0) -> None:
    for i in range(int(fps * seconds)):
        image = np.full((2, 2, 3), i % 256, dtype=np.uint8)
        buf.append(Frame(ts=start + i / fps, image=image))


def test_cursor_trails_the_live_edge():
    buf = DelayBuffer(fps=15, delay_s=8.0)
    fill(buf, 15, 12)
    assert buf.live_ts == pytest.approx(11 + 14 / 15)
    assert buf.cursor_ts == pytest.approx(buf.live_ts - 8.0)


def test_not_ready_until_the_buffer_fills_past_the_cursor():
    buf = DelayBuffer(fps=15, delay_s=8.0)
    assert not buf.ready
    fill(buf, 15, 4)
    assert not buf.ready
    fill(buf, 15, 6, start=4.0)
    assert buf.ready


def test_at_cursor_returns_oldest_first_ending_at_the_cursor():
    buf = DelayBuffer(fps=15, delay_s=4.0)
    fill(buf, 15, 10)
    frames = buf.at_cursor(count=4, spacing_s=1.0)
    assert len(frames) == 4
    assert [f.ts for f in frames] == sorted(f.ts for f in frames)
    assert frames[-1].ts == pytest.approx(buf.cursor_ts, abs=1 / 15)
    assert frames[0].ts == pytest.approx(buf.cursor_ts - 3.0, abs=1 / 15)


def test_lookahead_is_strictly_after_the_cursor():
    buf = DelayBuffer(fps=15, delay_s=4.0)
    fill(buf, 15, 10)
    cursor = buf.cursor_ts
    assert cursor is not None
    ahead = buf.lookahead(count=2)
    assert len(ahead) == 2
    assert all(f.ts > cursor for f in ahead)
    assert ahead[-1].ts == pytest.approx(buf.live_ts, abs=1 / 15)


def test_lookahead_stops_at_a_cut():
    """Past a cut it is a different picture, not what happens next.

    Broadcasts cut away every few seconds. Handing the caller frames from the
    far side of one while calling them the near future invites exactly the
    confident wrong line the lookahead exists to prevent.
    """
    buf = DelayBuffer(fps=15, delay_s=4.0)
    fill(buf, 15, 10)
    cursor = buf.cursor_ts
    assert cursor is not None

    cut_at = cursor + 1.5
    ahead = buf.lookahead(count=2, until_ts=cut_at)
    assert ahead, "there is still a second and a half of the same play to see"
    assert all(cursor < f.ts <= cut_at + 1 / 15 for f in ahead)

    # A cut on top of the cursor leaves nothing legitimate to show.
    assert buf.lookahead(count=2, until_ts=cursor) == []


def test_zero_delay_means_no_lookahead():
    buf = DelayBuffer(fps=15, delay_s=0.0)
    fill(buf, 15, 5)
    assert buf.lookahead() == []
    assert buf.cursor_ts == buf.live_ts


def test_capacity_is_bounded():
    buf = DelayBuffer(fps=15, delay_s=8.0, history_s=6.0)
    fill(buf, 15, 60)
    assert len(buf) == 15 * 14 + 1


def test_negative_delay_rejected():
    with pytest.raises(ValueError):
        DelayBuffer(fps=15, delay_s=-1.0)
