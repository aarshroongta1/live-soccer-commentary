"""A ring buffer with a narration cursor that trails the live edge.

Perception happens at the live edge; the cursor sits ``delay_s`` behind it, so
whatever the caller describes has already resolved. The viewer watches the
video from the cursor too, which is what makes the delay invisible.
"""

from __future__ import annotations

import time
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Frame:
    """One captured frame. ``ts`` is monotonic seconds since capture start."""

    ts: float
    image: np.ndarray


@dataclass(frozen=True)
class AudioChunk:
    """A slice of broadcast sound, mono float32 in [-1, 1].

    Sound is the one input that resolves faster than the picture: a whistle is
    unambiguous within a tenth of a second, while the frame it belongs to still
    looks like twenty-two people running. The chunks are short so a trigger can
    fire before the caller's next tick.
    """

    ts: float
    samples: np.ndarray
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def rms(self) -> float:
        if self.samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(self.samples, dtype=np.float64))))


class AudioRing:
    """The last few seconds of sound, addressed by time like the frame buffer."""

    def __init__(self, seconds: float = 20.0, chunk_s: float = 0.1) -> None:
        self.seconds = seconds
        self._chunks: deque[AudioChunk] = deque(maxlen=max(1, int(seconds / chunk_s)))

    def __len__(self) -> int:
        return len(self._chunks)

    def append(self, chunk: AudioChunk) -> None:
        self._chunks.append(chunk)

    @property
    def latest(self) -> AudioChunk | None:
        return self._chunks[-1] if self._chunks else None

    def since(self, ts: float) -> list[AudioChunk]:
        return [c for c in self._chunks if c.ts >= ts]

    def window(self, end_ts: float, seconds: float) -> list[AudioChunk]:
        return [c for c in self._chunks if end_ts - seconds <= c.ts <= end_ts]


class DelayBuffer:
    """Fixed-capacity frame store addressed by time rather than by index."""

    def __init__(self, fps: int, delay_s: float, history_s: float = 6.0) -> None:
        if delay_s < 0:
            raise ValueError("delay_s must be >= 0")
        self.fps = fps
        self.delay_s = delay_s
        self.history_s = history_s
        capacity = int((delay_s + history_s) * fps) + 1
        self._frames: deque[Frame] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._frames)

    def append(self, frame: Frame) -> None:
        self._frames.append(frame)

    @property
    def live_ts(self) -> float | None:
        """Timestamp of the newest frame, the live edge."""
        return self._frames[-1].ts if self._frames else None

    @property
    def cursor_ts(self) -> float | None:
        """Where narration is reading: ``delay_s`` behind the live edge."""
        live = self.live_ts
        return None if live is None else live - self.delay_s

    @property
    def ready(self) -> bool:
        """True once the buffer has filled past the cursor."""
        if not self._frames:
            return False
        cursor = self.cursor_ts
        assert cursor is not None
        return self._frames[0].ts <= cursor

    def nearest(self, ts: float) -> Frame | None:
        """The frame closest in time to ``ts``."""
        if not self._frames:
            return None
        frames = list(self._frames)
        stamps = [f.ts for f in frames]
        i = bisect_left(stamps, ts)
        if i == 0:
            return frames[0]
        if i >= len(frames):
            return frames[-1]
        before, after = frames[i - 1], frames[i]
        return after if (after.ts - ts) < (ts - before.ts) else before

    def _sample(self, end_ts: float, count: int, spacing_s: float) -> list[Frame]:
        picked: list[Frame] = []
        for k in range(count - 1, -1, -1):
            frame = self.nearest(end_ts - k * spacing_s)
            if frame is None:
                continue
            if not picked or frame.ts != picked[-1].ts:
                picked.append(frame)
        return picked

    def at_cursor(self, count: int = 4, spacing_s: float = 1.0) -> list[Frame]:
        """Frames leading up to the cursor, oldest first. What is being called."""
        cursor = self.cursor_ts
        if cursor is None:
            return []
        return self._sample(cursor, count, spacing_s)

    def lookahead(self, count: int = 2, until_ts: float | None = None) -> list[Frame]:
        """Frames between the cursor and the live edge, oldest first.

        This is the future the caller gets to see before it commits to a line:
        the shot it is describing has already gone in, or has not.

        ``until_ts`` cuts the window short, and exists because the future is
        only the future of *this* passage of play. Broadcasts cut away
        constantly — to a replay, to the bench, to a face in the crowd — and a
        frame from the far side of a cut is not what happens next, it is a
        different picture entirely. Handing those to the caller while telling
        it they are the near future invites exactly the confident, wrong line
        the lookahead was added to prevent. When the window closes to nothing
        the caller simply gets none, which it is told how to handle.
        """
        cursor, live = self.cursor_ts, self.live_ts
        if cursor is None or live is None or self.delay_s <= 0 or count <= 0:
            return []
        end = live if until_ts is None else min(live, until_ts)
        if end <= cursor:
            return []
        step = (end - cursor) / count
        return self._sample(end, count, step)

    def drop_before(self, ts: float) -> None:
        """Discard frames older than ``ts``. Used when the stream stalls."""
        while self._frames and self._frames[0].ts < ts:
            self._frames.popleft()


def now() -> float:
    return time.monotonic()
