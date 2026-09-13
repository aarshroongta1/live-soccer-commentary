"""Synthetic sound and synthetic pictures, so the detectors are pinned down
without a broadcast to hand."""

import numpy as np
import pytest

from commentary.capture.audio import CutDetector
from commentary.capture.buffer import Frame
from commentary.schemas import Trigger


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
