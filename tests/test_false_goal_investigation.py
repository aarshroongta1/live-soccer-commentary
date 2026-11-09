"""Diagnostic ablations must isolate their change and preserve input evidence."""

import importlib.util

import cv2
import numpy as np
import pytest

from commentary.recorded_demo import Frame


def diagnostic():
    spec = importlib.util.spec_from_file_location(
        "goal_diagnostic", "scripts/investigate_false_goals.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_neutral_condition_changes_one_word_only():
    d = diagnostic()
    system = "prefix\nThis is one recorded highlights sequence, prepared before playback.\nsuffix"
    blocks = [{"type": "text", "text": "unchanged"}]
    actual_system, actual_blocks = d.transformed(system, blocks, "neutral_word", [], None, [])
    assert actual_system == system.replace("highlights", "football")
    assert actual_blocks is blocks
    with pytest.raises(ValueError, match="phrase changed"):
        d.transformed("wrong baseline", blocks, "neutral_word", [], None, [])


def test_magnification_preserves_full_frames_and_original_timestamps():
    d = diagnostic()
    ok, jpeg = cv2.imencode(".jpg", np.zeros((60, 100, 3), dtype=np.uint8))
    assert ok
    frames = [Frame(at_s=6, jpeg=jpeg.tobytes())]
    blocks = [{"type": "text", "text": "original context"}]
    system, result = d.transformed(
        "same system", blocks, "magnified", frames, [0, 0.2, 0.6, 0.9], [6]
    )
    assert system == "same system"
    assert result[0] == blocks[0]
    assert "at_s=6" in result[1]["text"]
    assert result[2]["type"] == "image"
    assert blocks == [{"type": "text", "text": "original context"}]
    with pytest.raises(ValueError, match="exactly match"):
        d.crop_blocks(frames, [0, 0.2, 0.6, 0.9], [6.001])
