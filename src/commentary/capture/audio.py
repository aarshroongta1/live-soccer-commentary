"""Triggers: the instants when something might be worth saying.

A human commentator does not narrate on a timer. The director cuts to a
tight shot and the voice moves. One cue, and it is the cheapest in the
system: a subtraction per frame, run alongside the capture loop, never the
thing that stalls it.

There were two more. A whistle detector and a crowd-roar detector rode the
broadcast audio here, and on 58 runs of real footage the whistle fired once
and the roar fired every five seconds whatever was happening — including
never once on the goal of the tuned clip. Both are gone, and the audio with
them; ``docs/HANDOFF.md`` section 8 has the numbers.

The detector is edge-triggered: one cut is one event, not every frame of the
new shot, so it only re-arms once the difference has fallen away *and* a
hold-off has passed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from commentary.capture.buffer import Frame
from commentary.config import SETTINGS, PredictorConfig
from commentary.schemas import Trigger


@dataclass(frozen=True)
class TriggerEvent:
    """One detected cue, stamped with the video time it happened at."""

    trigger: Trigger
    ts: float
    strength: float


class _EdgeGate:
    """Turns a continuous condition into single events.

    Fires on the rising edge, then stays quiet until the condition has dropped
    back below a release level and ``refractory_s`` has elapsed. Without the
    release requirement a slow dissolve would fire every time the hold-off
    expired; without the hold-off a level hovering at the threshold would
    chatter.
    """

    def __init__(self, refractory_s: float, release: float = 0.8) -> None:
        self.refractory_s = refractory_s
        self.release = release
        self._armed = True
        self._last_fire_ts: float | None = None

    def step(self, value: float, threshold: float, ts: float) -> bool:
        if value < threshold * self.release:
            self._armed = True
        if value < threshold or not self._armed:
            return False
        if self._last_fire_ts is not None and ts - self._last_fire_ts < self.refractory_s:
            return False
        self._armed = False
        self._last_fire_ts = ts
        return True


def thumbnail(image: np.ndarray, size: int = 32) -> np.ndarray:
    """Greyscale, area-averaged down to ``size`` square.

    Downscaling before differencing is the whole trick: a handheld camera
    shaking, a player running across frame, and film grain all survive a
    pixel-wise difference at full resolution and all disappear once the frame
    is thirty-two blocks wide. What survives is a genuine change of shot.
    """
    raw = np.asarray(image)
    if raw.ndim not in (2, 3) or raw.size == 0:
        return np.zeros((1, 1), dtype=np.float64)
    # Decimate with a stride first — a free view — so the float conversion and
    # the averaging run over a few thousand pixels instead of a million. Four
    # samples per output block either way is plenty to average grain out.
    rough = raw[:: max(1, raw.shape[0] // (size * 4)), :: max(1, raw.shape[1] // (size * 4))]
    grey = rough.astype(np.float64)
    if grey.ndim == 3:
        grey = grey.mean(axis=2)
    height, width = grey.shape
    rows = np.linspace(0, height, min(size, height) + 1).astype(int)
    cols = np.linspace(0, width, min(size, width) + 1).astype(int)
    sums = np.add.reduceat(np.add.reduceat(grey, rows[:-1], axis=0), cols[:-1], axis=1)
    counts = np.outer(np.diff(rows), np.diff(cols)).astype(np.float64)
    return np.asarray(sums / counts, dtype=np.float64)


class CutDetector:
    """Camera cut: mean absolute difference between consecutive thumbnails.

    A cut is the director telling you the story moved — to the scorer's face,
    to the bench, to the replay. It is the cheapest signal in the system and it
    costs one subtraction per frame.
    """

    def __init__(
        self,
        cfg: PredictorConfig = SETTINGS.predictor,
        *,
        size: int = 32,
        refractory_s: float = 0.4,
    ) -> None:
        self.cfg = cfg
        self.size = size
        self._gate = _EdgeGate(refractory_s, release=0.5)
        self._previous: np.ndarray | None = None

    def feed(self, frame: Frame) -> TriggerEvent | None:
        """One frame in, at most one cut event out. The first frame never fires."""
        small = thumbnail(frame.image, self.size)
        previous, self._previous = self._previous, small
        if previous is None or previous.shape != small.shape:
            return None
        diff = float(np.mean(np.abs(small - previous)))
        if not self._gate.step(diff, self.cfg.cut_threshold, frame.ts):
            return None
        return TriggerEvent(trigger=Trigger.CAMERA_CUT, ts=frame.ts, strength=diff)
