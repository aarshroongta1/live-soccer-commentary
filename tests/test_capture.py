"""Reading a file the way a match will be read: pictures and sound together.

The file is built here rather than checked in, so the test says exactly what
it is asserting about — thirty rendered frames and a second and a half of
tone, muxed the way a downloaded broadcast arrives.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from commentary.capture import FileCapture
from commentary.capture.source import ScreenCapture
from commentary.config import CaptureConfig

SR = 16000
SECONDS = 2.0
CFG = CaptureConfig(width=320, height=180, fps=15)


def _write_clip(tmp_path: Path) -> Path:
    """A tiny mp4 with a tone on it, muxed by ffmpeg like a real download."""
    import subprocess

    import cv2

    from commentary.sim import MatchSim
    from commentary.sim.render import BroadcastRenderer

    sim = MatchSim(seed=3, duration_s=30.0)
    renderer = BroadcastRenderer(sim.knowledge_pack, width=CFG.width, height=CFG.height)
    silent = tmp_path / "video.mp4"
    writer = cv2.VideoWriter(
        str(silent), int(cv2.VideoWriter.fourcc(*"mp4v")), CFG.fps, (CFG.width, CFG.height)
    )
    try:
        for index in range(int(SECONDS * CFG.fps)):
            writer.write(renderer.frame(sim.at(index / CFG.fps)))
    finally:
        writer.release()

    tone = tmp_path / "tone.wav"
    t = np.arange(int(SR * SECONDS)) / SR
    samples = (0.5 * np.sin(2.0 * np.pi * 440.0 * t) * 32767).astype("<i2")
    with wave.open(str(tone), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SR)
        wav.writeframes(samples.tobytes())

    clip = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(silent),
         "-i", str(tone), "-c:v", "copy", "-c:a", "aac", "-shortest", str(clip)],
        check=True,
    )
    return clip


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_clip(tmp_path_factory.mktemp("clip"))


async def test_a_file_yields_frames_at_the_capture_size(clip: Path):
    async with FileCapture(clip, CFG, realtime=False) as source:
        frames = [frame async for frame in source.frames()]
    assert len(frames) > 10
    assert frames[0].ts == 0.0
    assert frames[0].image.shape == (CFG.height, CFG.width, 3)


def test_the_screen_source_pins_the_output_rate_so_the_buffer_holds_the_delay():
    cfg = CaptureConfig(width=320, height=180, fps=15)
    command = ScreenCapture(cfg)._command()
    i_index = command.index("-i")
    after_i = command[i_index + 2 : i_index + 4]
    assert after_i == ["-r", str(cfg.fps)]
    assert command.index("-framerate") < i_index
