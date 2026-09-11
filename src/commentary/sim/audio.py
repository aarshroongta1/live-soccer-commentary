"""Synthetic broadcast sound on the same clock as the picture.

Sound is in the sim for one reason: the speak predictor fires on whistles and
roars, and until there is audio with those things in it at known times, there
is no way to tell a working trigger from a lucky one. So the crowd is filtered
noise well below a kilohertz, the roars are that same crowd getting louder for
a few seconds, and the whistle is a narrow tone at three kilohertz. The bands
do not overlap, which means a test can assert on band energy rather than on
whether something sounds right.

Chunks are addressed by index and generated from a seeded stream, so asking
for the sound at ninety minutes costs the same as asking for it at ten
seconds.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from commentary.capture.buffer import AudioChunk
from commentary.schemas import Event
from commentary.sim.match import MatchSim

#: Crowd noise is rolled off here, leaving the whistle band clear above it.
CROWD_CUTOFF_HZ = 1100.0
WHISTLE_HZ = 3000.0
WHISTLE_LEN_S = 0.28
CROWD_LEVEL = 0.055

#: Peak extra gain on the crowd, and how long it takes to fall away.
_ROAR: dict[Event, tuple[float, float]] = {
    Event.GOAL: (9.0, 2.8),
    Event.SAVE: (3.6, 1.1),
    Event.SHOT: (3.0, 0.8),
    Event.PENALTY: (6.0, 2.0),
    Event.CARD: (2.4, 1.0),
    Event.CORNER: (1.2, 0.7),
}
_ROAR_ATTACK_S = 0.25


def _lowpass_kernel(cutoff_hz: float, sample_rate: int, taps: int = 63) -> np.ndarray:
    n = np.arange(taps) - (taps - 1) / 2.0
    kernel = np.sinc(2.0 * cutoff_hz / sample_rate * n) * np.hanning(taps)
    return np.asarray(kernel / kernel.sum(), dtype=np.float64)


@dataclass
class MatchAudio:
    """The sound of one simulated match, sliced into fixed chunks.

    ``sim`` supplies the times: every roar and every whistle is anchored to a
    ground truth event, so the audio and the video cannot drift apart.
    """

    sim: MatchSim
    sample_rate: int = 16000
    chunk_s: float = 0.1
    seed: int = 5
    _kernel: np.ndarray = field(init=False, repr=False)
    _gain: float = field(init=False, repr=False)
    _roars: list[tuple[float, float, float]] = field(init=False, repr=False)
    _whistles: list[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._kernel = _lowpass_kernel(CROWD_CUTOFF_HZ, self.sample_rate)
        # Filtering a unit-variance stream costs it variance. Put it back, so
        # the crowd floor sits where CROWD_LEVEL says it does.
        self._gain = 1.0 / float(np.sqrt(np.sum(self._kernel**2)))
        self._roars = []
        for g in self.sim.ground_truth:
            spec = _ROAR.get(g.event)
            if spec is not None:
                self._roars.append((g.video_ts, spec[0], spec[1]))
        self._whistles = self.sim.whistle_times

    @property
    def samples_per_chunk(self) -> int:
        return int(round(self.sample_rate * self.chunk_s))

    @property
    def chunk_count(self) -> int:
        return int(np.ceil(self.sim.duration_s / self.chunk_s))

    def roar_gain(self, ts: float) -> float:
        """How much louder the crowd is than its floor at ``ts``."""
        total = 0.0
        for start, peak, decay in self._roars:
            dt = ts - start
            if dt < 0.0 or dt > decay * 6.0 + _ROAR_ATTACK_S:
                continue
            if dt < _ROAR_ATTACK_S:
                total += peak * dt / _ROAR_ATTACK_S
            else:
                total += peak * float(np.exp(-(dt - _ROAR_ATTACK_S) / decay))
        return total

    def _crowd(self, index: int, n: int) -> np.ndarray:
        rng = np.random.default_rng((self.seed * 1_000_003 + index) & 0xFFFFFFFF)
        taps = len(self._kernel)
        raw = rng.standard_normal(n + taps - 1)
        return np.asarray(np.convolve(raw, self._kernel, mode="valid") * self._gain)

    def _whistle(self, t: np.ndarray) -> np.ndarray:
        out = np.zeros_like(t)
        for start in self._whistles:
            if t[-1] < start or t[0] > start + WHISTLE_LEN_S:
                continue
            dt = t - start
            live = (dt >= 0.0) & (dt <= WHISTLE_LEN_S)
            if not live.any():
                continue
            # Anchoring the phase to the whistle's own start keeps the tone
            # continuous across a chunk boundary rather than clicking.
            phase = 2.0 * np.pi * WHISTLE_HZ * dt + 0.9 * np.sin(2.0 * np.pi * 7.0 * dt)
            window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.clip(dt / WHISTLE_LEN_S, 0.0, 1.0))
            out += np.where(live, 0.4 * window * np.sin(phase), 0.0)
        return out

    def chunk(self, index: int) -> AudioChunk:
        """The ``index``-th chunk of broadcast sound."""
        n = self.samples_per_chunk
        ts = index * self.chunk_s
        t = ts + np.arange(n, dtype=np.float64) / self.sample_rate
        level = CROWD_LEVEL * (1.0 + self.roar_gain(ts + self.chunk_s / 2.0))
        samples = self._crowd(index, n) * level + self._whistle(t)
        return AudioChunk(
            ts=ts,
            samples=np.clip(samples, -1.0, 1.0).astype(np.float32),
            sample_rate=self.sample_rate,
        )

    def at(self, ts: float) -> AudioChunk:
        return self.chunk(int(ts / self.chunk_s))

    def chunks(self) -> Iterator[AudioChunk]:
        for i in range(self.chunk_count):
            yield self.chunk(i)


def band_energy(chunk: AudioChunk, low_hz: float, high_hz: float) -> float:
    """Mean power in a frequency band. What the whistle detector will key on."""
    if chunk.samples.size == 0:
        return 0.0
    spectrum = np.fft.rfft(chunk.samples.astype(np.float64))
    freqs = np.fft.rfftfreq(chunk.samples.size, 1.0 / chunk.sample_rate)
    band = (freqs >= low_hz) & (freqs <= high_hz)
    if not band.any():
        return 0.0
    return float(np.mean(np.abs(spectrum[band]) ** 2))
