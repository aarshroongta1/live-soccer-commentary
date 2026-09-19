"""The benchmark must obey the playback clock, including when models are slow."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from commentary.llm.base import Usage
from commentary.recorded_demo import Frame

_SPEC = importlib.util.spec_from_file_location(
    "try_streaming", Path(__file__).parents[1] / "scripts/try_streaming.py"
)
assert _SPEC and _SPEC.loader
trial = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = trial
_SPEC.loader.exec_module(trial)


def test_frame_reader_never_reads_after_elapsed_window() -> None:
    positions = []

    class Capture:
        def set(self, _property, value):
            positions.append(value)

        def read(self):
            return True, np.zeros((32, 32, 3), dtype=np.uint8)

    frames = trial.read_frames(Capture(), source_start=8, begin=4, end=8)
    assert len(frames) == 8
    assert positions == [12000 + 500 * index for index in range(8)]
    assert all(4 <= frame.at_s < 8 for frame in frames)


@pytest.mark.asyncio
async def test_slow_trial_skips_backlog_without_sending_future_frames(tmp_path, monkeypatch):
    clock = [0.0]
    received = []
    monkeypatch.setattr(trial.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(trial, "validate_inputs", lambda *_args: 60)
    monkeypatch.setattr(
        trial.cv2, "VideoCapture", lambda *_args: SimpleNamespace(release=lambda: None)
    )
    monkeypatch.setattr(trial, "openai_factory", lambda **_kwargs: SimpleNamespace(total=Usage()))
    monkeypatch.setattr(
        trial,
        "read_frames",
        lambda _capture, **kw: [Frame(kw["begin"], b"jpg"), Frame(kw["end"] - 0.1, b"jpg")],
    )

    async def sleep(seconds):
        clock[0] += seconds

    class Commentator:
        def __init__(self, *_args, **_kwargs):
            pass

        async def process(self, frames, *, playback_s):
            assert all(frame.at_s <= playback_s <= clock[0] for frame in frames)
            received.append(playback_s)
            clock[0] += 6
            return {"line": {"voice": "caller", "text": "Argentina advance."}}

    monkeypatch.setattr(trial.asyncio, "sleep", sleep)
    monkeypatch.setattr(trial, "StreamingCommentator", Commentator)
    output = tmp_path / "trial.jsonl"
    args = SimpleNamespace(
        clip=Path("clip.mp4"),
        start=8,
        duration=16,
        interval=4,
        observer_model="gpt-5.6-terra",
        writer_model="gpt-5.6-terra",
        out=output,
        pack=None,
        research=None,
        spend=True,
    )
    await trial.trial(args)
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert received == [4, 10, 16]
    assert summary["median_processing_s"] == 6
    assert summary["skipped_footage_s"] == 4
    assert summary["unprocessed_tail_s"] == 0
    assert summary["batches_within_interval"] == 0
