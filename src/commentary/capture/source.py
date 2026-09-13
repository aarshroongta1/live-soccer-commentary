"""Where frames come from.

The rest of the system never asks. A screen capture, a video file, and the
simulator all present the same thing: an async iterator of timestamped frames
with a wall-clock pace. That is what lets the whole pipeline run tonight
against generated video and against a real broadcast tomorrow, unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

import numpy as np

from commentary.capture.buffer import Frame, now
from commentary.config import SETTINGS, CaptureConfig


def _release(proc: asyncio.subprocess.Process) -> None:
    """Close the subprocess transport, which asyncio never does by itself.

    Left alone it is closed by the garbage collector, which by then is often
    running after the event loop has gone and raises out of ``__del__``. The
    transport is reachable only as a private attribute; there is no public
    way to close one, which is why this is read defensively.
    """
    transport = getattr(proc, "_transport", None)
    if transport is not None:
        transport.close()


async def _stop(proc: asyncio.subprocess.Process | None) -> None:
    """Terminate an ffmpeg process and wait for it, without deadlocking.

    Draining stdout first is not tidiness. asyncio pauses a pipe's reader once
    its buffer passes ``limit``, and a paused reader never sees EOF, so a
    process whose output was abandoned mid-stream never reports its pipe
    closed and ``wait()`` hangs forever. That happens the moment one of the
    two streams of a file is consumed and the other is not.
    """
    if proc is None:
        return
    if proc.returncode is not None:
        _release(proc)
        return
    proc.terminate()
    if proc.stdout is not None:
        with contextlib.suppress(Exception):
            await proc.stdout.read()
    await proc.wait()
    _release(proc)


class FrameSource(Protocol):
    """An async context manager yielding frames in real time."""

    async def __aenter__(self) -> FrameSource: ...

    async def __aexit__(self, *exc: object) -> None: ...

    def frames(self) -> AsyncIterator[Frame]: ...


class FFmpegSource:
    """Frames out of any ffmpeg input, decoded to raw BGR on stdout."""

    def __init__(
        self,
        args: list[str],
        cfg: CaptureConfig = SETTINGS.capture,
        *,
        realtime: bool = True,
    ) -> None:
        self.cfg = cfg
        self._args = args
        self._realtime = realtime
        self._proc: asyncio.subprocess.Process | None = None

    @property
    def frame_bytes(self) -> int:
        return self.cfg.width * self.cfg.height * 3

    def _command(self) -> list[str]:
        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            *self._args,
            "-vf",
            f"scale={self.cfg.width}:{self.cfg.height}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]

    async def __aenter__(self) -> FFmpegSource:
        self._proc = await asyncio.create_subprocess_exec(
            *self._command(),
            stdout=asyncio.subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            limit=self.frame_bytes * 4,
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        await _stop(self._proc)
        self._proc = None

    async def frames(self) -> AsyncIterator[Frame]:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("use the source as an async context manager")
        start = now()
        index = 0
        while True:
            try:
                raw = await self._proc.stdout.readexactly(self.frame_bytes)
            except asyncio.IncompleteReadError:
                return
            image = np.frombuffer(raw, dtype=np.uint8).reshape(self.cfg.height, self.cfg.width, 3)
            ts = index / self.cfg.fps
            index += 1
            if self._realtime:
                # A file decodes far faster than it plays. Pace it, so the
                # delay buffer and the rate caps mean the same thing they will
                # mean against a live screen.
                behind = ts - (now() - start)
                if behind > 0:
                    await asyncio.sleep(behind)
            yield Frame(ts=ts, image=image)


class ScreenCapture(FFmpegSource):
    """Whatever is on screen: a broadcast, a browser tab, a highlight reel."""

    def __init__(self, cfg: CaptureConfig = SETTINGS.capture) -> None:
        super().__init__(
            [
                "-f",
                "avfoundation",
                "-capture_cursor",
                "0",
                "-framerate",
                str(cfg.fps),
                "-i",
                cfg.device,
            ],
            cfg,
            realtime=False,  # the screen already runs at wall-clock speed
        )

    async def frames(self) -> AsyncIterator[Frame]:
        # A live capture stamps frames by arrival, not by index: dropped frames
        # must show up as a gap, not as drift.
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("use the source as an async context manager")
        start = now()
        while True:
            try:
                raw = await self._proc.stdout.readexactly(self.frame_bytes)
            except asyncio.IncompleteReadError:
                return
            image = np.frombuffer(raw, dtype=np.uint8).reshape(self.cfg.height, self.cfg.width, 3)
            yield Frame(ts=now() - start, image=image)


class FileCapture(FFmpegSource):
    """A recorded match, played at its real speed.

    This is the development loop: the same clip, repeatedly, so prompt and
    threshold changes are comparable between runs. Picture only: the audio
    triggers that used to ride a second ffmpeg process were measured on real
    broadcast and removed (see ``docs/HANDOFF.md``, section 8).
    """

    def __init__(
        self,
        path: str | Path,
        cfg: CaptureConfig = SETTINGS.capture,
        *,
        start_s: float = 0.0,
        realtime: bool = True,
    ) -> None:
        args = ["-ss", str(start_s), "-re"] if realtime else ["-ss", str(start_s)]
        super().__init__([*args, "-i", str(path), "-r", str(cfg.fps)], cfg, realtime=realtime)
        self.path = Path(path)
        self.start_s = start_s
