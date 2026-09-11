"""Triggers: the instants when something might be worth saying.

A human commentator does not narrate on a timer. The whistle goes, the crowd
lifts, the director cuts to a tight shot, and the voice moves. Those three
cues are cheap to detect and they resolve faster than the picture does, which
is the whole point: sound tells you a foul happened a second before the frame
showing twenty-two people standing still tells you anything at all.

Everything here is numpy over one short window, because this runs every
100 ms alongside the capture loop and must never be the thing that stalls it.
No librosa, no model. A whistle is a narrow tone against broadband crowd
noise, so the share of energy inside the whistle band, divided by the share of
the spectrum that band occupies, separates it from noise in one FFT. A roar is
a step up in loudness relative to how loud this stadium has been for the last
few seconds, which is why the baseline is rolling rather than fixed: the same
absolute level is a lull at Anfield and a riot at an empty ground.

Each detector is edge-triggered. One whistle is one event, not the thirty
consecutive chunks that the blast actually spans, so a detector only re-arms
once the condition has fallen away *and* a hold-off has passed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from commentary.capture.buffer import AudioChunk, Frame
from commentary.config import SETTINGS, PredictorConfig
from commentary.schemas import Trigger


@dataclass(frozen=True)
class TriggerEvent:
    """One detected cue. ``ts`` is when the thing happened, not when we noticed.

    The distinction matters for the roar: by the time the level has clearly
    risen the ball is already in the net, and the beat the caller should be
    describing is the onset, half a second earlier.
    """

    trigger: Trigger
    ts: float
    strength: float


class _EdgeGate:
    """Turns a continuous condition into single events.

    Fires on the rising edge, then stays quiet until the condition has dropped
    back below a release level and ``refractory_s`` has elapsed. Without the
    release requirement a two-second whistle would fire every time the hold-off
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


class WhistleDetector:
    """Referee's whistle: a narrow tone poking out of broadband crowd noise.

    The measure is the energy inside ``whistle_band_hz`` as a share of total
    energy, divided by the share of the spectrum that band occupies. Flat noise
    scores about 1 whatever its loudness; crowd noise, which is weighted low,
    scores below 1; a tone inside the band drives it towards the reciprocal of
    the band's width. That normalisation is what lets one threshold work across
    broadcasts with wildly different mixes.
    """

    def __init__(
        self,
        cfg: PredictorConfig = SETTINGS.predictor,
        *,
        refractory_s: float = 1.0,
        min_rms: float = 5e-3,
    ) -> None:
        self.cfg = cfg
        self.min_rms = min_rms
        self._gate = _EdgeGate(refractory_s)

    def band_ratio(self, chunk: AudioChunk) -> float:
        """Band energy share over band width share. Exposed so traces can plot it."""
        samples = np.asarray(chunk.samples, dtype=np.float64).ravel()
        if samples.size < 32 or chunk.sample_rate <= 0:
            return 0.0
        spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size))) ** 2
        total = float(spectrum.sum())
        if total <= 0.0:
            return 0.0
        freqs = np.fft.rfftfreq(samples.size, 1.0 / chunk.sample_rate)
        low, high = self.cfg.whistle_band_hz
        in_band = (freqs >= low) & (freqs < high)
        width_share = float(in_band.mean())
        if width_share <= 0.0:
            return 0.0
        return float(spectrum[in_band].sum() / total) / width_share

    def feed(self, chunk: AudioChunk) -> TriggerEvent | None:
        """One chunk in, at most one whistle event out."""
        if chunk.rms < self.min_rms:
            return None
        ratio = self.band_ratio(chunk)
        if not self._gate.step(ratio, self.cfg.whistle_ratio, chunk.ts):
            return None
        return TriggerEvent(trigger=Trigger.WHISTLE, ts=chunk.ts, strength=ratio)


class RoarDetector:
    """Crowd lift: loudness against the rolling median of the last few seconds.

    A median rather than a mean, because the thing we are measuring is exactly
    the kind of outlier that would drag a mean up and hide itself. The reported
    timestamp is the onset — where the level started climbing — since that is
    the moment the caller should be describing, not the moment the climb became
    undeniable.
    """

    def __init__(
        self,
        cfg: PredictorConfig = SETTINGS.predictor,
        *,
        baseline_s: float = 6.0,
        chunk_s: float = 0.1,
        refractory_s: float = 3.0,
        min_rms: float = 1e-3,
    ) -> None:
        self.cfg = cfg
        self.baseline_s = baseline_s
        self.min_rms = min_rms
        self._history: deque[tuple[float, float]] = deque(maxlen=max(4, int(baseline_s / chunk_s)))
        self._gate = _EdgeGate(refractory_s)

    @property
    def baseline(self) -> float:
        """How loud this stadium has been lately. Floored, so silence cannot divide."""
        if not self._history:
            return self.min_rms
        return max(float(np.median([rms for _, rms in self._history])), self.min_rms)

    @property
    def ready(self) -> bool:
        """True once there is enough history for the baseline to mean anything."""
        return len(self._history) >= max(4, self._history.maxlen or 0) // 2

    def _onset_ts(self, baseline: float, fallback_ts: float) -> float:
        """Walk back while the level was already climbing, and return where it began."""
        onset_level = baseline * (1.0 + (self.cfg.roar_ratio - 1.0) / 3.0)
        onset = fallback_ts
        for ts, rms in reversed(self._history):
            if rms < onset_level:
                break
            onset = ts
        return onset

    def feed(self, chunk: AudioChunk) -> TriggerEvent | None:
        """One chunk in, at most one roar event out, stamped at its onset."""
        rms = chunk.rms
        baseline = self.baseline
        ready = self.ready
        self._history.append((chunk.ts, rms))
        if not ready or rms < self.min_rms:
            return None
        ratio = rms / baseline
        if not self._gate.step(ratio, self.cfg.roar_ratio, chunk.ts):
            return None
        return TriggerEvent(
            trigger=Trigger.ROAR, ts=self._onset_ts(baseline, chunk.ts), strength=ratio
        )


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
