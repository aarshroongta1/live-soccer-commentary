"""Screen capture through ffmpeg's avfoundation input.

Any video playing on screen counts as a live stream: a broadcast, a stream in a
browser tab, a highlight reel. ffmpeg writes raw BGR frames to stdout and we
read them a frame at a time, so end-to-end latency stays around 100 ms.
"""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import AsyncIterator

import numpy as np

from commentary.capture.buffer import Frame, now
from commentary.config import CAPTURE, CaptureConfig


def ffmpeg_command(cfg: CaptureConfig) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "avfoundation",
        "-capture_cursor",
        "0",
        "-framerate",
        str(cfg.fps),
        "-i",
        cfg.device,
        "-vf",
        f"scale={cfg.width}:{cfg.height}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]


class ScreenCapture:
    """Async iterator of :class:`Frame` objects off the screen."""

    def __init__(self, cfg: CaptureConfig = CAPTURE) -> None:
        self.cfg = cfg
        self._proc: asyncio.subprocess.Process | None = None

    @property
    def frame_bytes(self) -> int:
        return self.cfg.width * self.cfg.height * 3

    async def __aenter__(self) -> ScreenCapture:
        self._proc = await asyncio.create_subprocess_exec(
            *ffmpeg_command(self.cfg),
            stdout=asyncio.subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            limit=self.frame_bytes * 4,
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()
            await self._proc.wait()
        self._proc = None

    async def frames(self) -> AsyncIterator[Frame]:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("use ScreenCapture as an async context manager")
        start = now()
        while True:
            try:
                raw = await self._proc.stdout.readexactly(self.frame_bytes)
            except asyncio.IncompleteReadError:
                return
            image = np.frombuffer(raw, dtype=np.uint8).reshape(self.cfg.height, self.cfg.width, 3)
            yield Frame(ts=now() - start, image=image)
