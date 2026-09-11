"""The simulator dressed up as a capture device.

This is the point of the whole package. :class:`SimSource` satisfies the same
``FrameSource`` protocol as the screen capture, so the pipeline that will run
against a real broadcast in October runs tonight against a match that does not
exist, and neither the delay buffer nor the caller can tell.

Frames and sound come off one clock. They are separate iterators because the
audio triggers have to fire faster than the caller ticks, and both are meant
to be driven from a single ``asyncio.gather``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from commentary.capture.buffer import AudioChunk, Frame, now
from commentary.config import CAPTURE, CaptureConfig
from commentary.schemas import KnowledgePack
from commentary.sim.audio import MatchAudio
from commentary.sim.match import MatchSim
from commentary.sim.render import BroadcastRenderer


class SimSource:
    """Rendered frames and synthetic sound, paced like a live capture or not.

    ``realtime=True`` sleeps between frames so a demo runs at watching speed
    and the rate caps mean what they will mean live. ``realtime=False`` runs
    flat out, which is the only way a test of a three minute match finishes in
    a couple of seconds.
    """

    def __init__(
        self,
        sim: MatchSim | None = None,
        cfg: CaptureConfig = CAPTURE,
        *,
        realtime: bool = False,
        sample_rate: int = 16000,
        chunk_s: float = 0.1,
    ) -> None:
        self.sim = sim if sim is not None else MatchSim()
        self.cfg = cfg
        self.realtime = realtime
        self.renderer = BroadcastRenderer(
            self.sim.knowledge_pack, width=cfg.width, height=cfg.height
        )
        self.audio_track = MatchAudio(self.sim, sample_rate=sample_rate, chunk_s=chunk_s)
        self._started: float | None = None
        self._open = False

    @property
    def pack(self) -> KnowledgePack:
        return self.sim.knowledge_pack

    @property
    def duration_s(self) -> float:
        return self.sim.duration_s

    async def __aenter__(self) -> SimSource:
        self._started = now()
        self._open = True
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._open = False
        self._started = None

    def _require_open(self) -> float:
        if not self._open or self._started is None:
            raise RuntimeError("use the source as an async context manager")
        return self._started

    async def _pace(self, ts: float, start: float) -> None:
        if self.realtime:
            behind = ts - (now() - start)
            if behind > 0:
                await asyncio.sleep(behind)
                return
        # Even flat out, yield: frames and audio share one event loop and the
        # consumer is usually gathering both.
        await asyncio.sleep(0)

    async def frames(self) -> AsyncIterator[Frame]:
        start = self._require_open()
        index = 0
        while self._open:
            ts = index / self.cfg.fps
            if ts > self.sim.duration_s:
                return
            index += 1
            await self._pace(ts, start)
            yield Frame(ts=ts, image=self.renderer.frame(self.sim.at(ts)))

    async def audio(self) -> AsyncIterator[AudioChunk]:
        start = self._require_open()
        index = 0
        while self._open:
            chunk = self.audio_track.chunk(index)
            if chunk.ts > self.sim.duration_s:
                return
            index += 1
            await self._pace(chunk.ts, start)
            yield chunk
