"""Synthetic sound and synthetic pictures, so the detectors are pinned down
without a broadcast to hand."""

import numpy as np
import pytest

from commentary.capture.audio import CutDetector, RoarDetector, TriggerEvent, WhistleDetector
from commentary.capture.buffer import AudioChunk, Frame
from commentary.schemas import Trigger

SR = 22050
CHUNK_S = 0.1
N = int(SR * CHUNK_S)


def chunk(index: int, samples: np.ndarray) -> AudioChunk:
    return AudioChunk(ts=index * CHUNK_S, samples=samples.astype(np.float32), sample_rate=SR)


def tone(index: int, freq: float, amp: float) -> np.ndarray:
    """A continuous tone, phase-correct across chunk boundaries."""
    samples = np.arange(index * N, (index + 1) * N)
    return amp * np.sin(2.0 * np.pi * freq * samples / SR)


def test_a_tone_burst_in_noise_fires_the_whistle_once():
    detector = WhistleDetector()
    rng = np.random.default_rng(0)
    fired: list[TriggerEvent] = []
    for i in range(30):
        signal = rng.normal(0.0, 0.02, N)
        if 10 <= i < 20:  # a full second of whistle, thirty chunks it is not
            signal = signal + tone(i, 3000.0, 0.6)
        event = detector.feed(chunk(i, signal))
        if event is not None:
            fired.append(event)
    assert len(fired) == 1
    assert fired[0].trigger is Trigger.WHISTLE
    assert fired[0].ts == pytest.approx(1.0)
    assert fired[0].strength >= 3.5


def test_broadband_noise_alone_does_not_fire_the_whistle():
    detector = WhistleDetector()
    rng = np.random.default_rng(1)
    for i in range(60):
        assert detector.feed(chunk(i, rng.normal(0.0, 0.05, N))) is None


def test_near_silence_does_not_fire_the_whistle():
    detector = WhistleDetector()
    for i in range(20):
        assert detector.feed(chunk(i, tone(i, 3000.0, 1e-4))) is None


def test_a_step_up_in_level_fires_the_roar_at_its_onset():
    detector = RoarDetector(chunk_s=CHUNK_S)
    rng = np.random.default_rng(2)
    fired: list[TriggerEvent] = []
    for i in range(120):
        level = 0.05 if i < 80 else 0.25
        event = detector.feed(chunk(i, rng.normal(0.0, level, N)))
        if event is not None:
            fired.append(event)
    assert len(fired) == 1
    assert fired[0].trigger is Trigger.ROAR
    assert fired[0].ts == pytest.approx(8.0, abs=0.2)
    assert fired[0].strength > 2.2


def test_a_slow_drift_in_level_does_not_fire_the_roar():
    detector = RoarDetector(chunk_s=CHUNK_S)
    rng = np.random.default_rng(3)
    for i in range(600):
        level = 0.05 + 0.15 * i / 600.0
        assert detector.feed(chunk(i, rng.normal(0.0, level, N))) is None


def test_the_roar_baseline_is_relative_to_this_stadium():
    """The same absolute level is a lull in one ground and a roar in another."""
    quiet = RoarDetector(chunk_s=CHUNK_S)
    loud = RoarDetector(chunk_s=CHUNK_S)
    rng = np.random.default_rng(4)
    fired_quiet, fired_loud = [], []
    for i in range(80):
        test_level = 0.12
        base_quiet = 0.02 if i < 60 else test_level
        base_loud = 0.30 if i < 60 else test_level
        if (event := quiet.feed(chunk(i, rng.normal(0.0, base_quiet, N)))) is not None:
            fired_quiet.append(event)
        if (event := loud.feed(chunk(i, rng.normal(0.0, base_loud, N)))) is not None:
            fired_loud.append(event)
    assert fired_quiet and not fired_loud


def frame(index: int, image: np.ndarray) -> Frame:
    return Frame(ts=index / 15.0, image=image)


def test_a_hard_frame_change_fires_the_cut():
    detector = CutDetector()
    before = np.full((90, 160, 3), 40, dtype=np.uint8)
    after = np.full((90, 160, 3), 200, dtype=np.uint8)
    fired = []
    for i in range(20):
        event = detector.feed(frame(i, before if i < 10 else after))
        if event is not None:
            fired.append(event)
    assert len(fired) == 1
    assert fired[0].trigger is Trigger.CAMERA_CUT
    assert fired[0].ts == pytest.approx(10 / 15.0)


def test_the_first_frame_never_fires():
    detector = CutDetector()
    assert detector.feed(frame(0, np.full((90, 160, 3), 200, dtype=np.uint8))) is None


def test_grain_and_jitter_do_not_fire_the_cut():
    detector = CutDetector()
    rng = np.random.default_rng(5)
    base = np.tile(np.linspace(0, 255, 160, dtype=np.uint8), (90, 1))
    for i in range(40):
        noisy = np.clip(base.astype(np.int16) + rng.integers(-6, 7, base.shape), 0, 255)
        image = np.repeat(noisy.astype(np.uint8)[:, :, None], 3, axis=2)
        assert detector.feed(frame(i, image)) is None
