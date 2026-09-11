"""Where audio bytes go, and how they are silenced.

The director already owns the queue: it holds three beats, drops the stale
ones, puts a goal at the front, and speaks one beat at a time. What it cannot
see is that a cancelled line does not stop being audible when ``say`` returns.
Between the model and the listener sits a pipe, a decoder and a sound-card
buffer holding a few hundred milliseconds of already-rendered speech, and
those keep playing after the stream is abandoned. A "preemption" that leaves
the analyst audible under the goal call is not a preemption.

So this module draws the line the director cannot: a sink that can be told to
stop and be silent *now*, discarding whatever it is holding, and separately to
finish, meaning play out what it has been given and only then come back. The
first is what a cancel needs; the second is what keeps the caller and the
analyst from overlapping, because the director's next ``say`` cannot start
until the previous one has returned.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from typing import Protocol

log = logging.getLogger(__name__)

#: How long to wait for a finished line to play out of the sound-card buffer
#: before giving up on it. Generous: a normal line drains in well under a
#: second, and hanging here would stall the whole director loop.
DRAIN_TIMEOUT_S = 5.0

#: How long to wait for a killed player to actually die. Very short — this is
#: on the preemption path, where every millisecond is audible.
KILL_TIMEOUT_S = 0.5


class AudioSink(Protocol):
    """Somewhere to put audio chunks that can be silenced mid-chunk."""

    async def start(self) -> None:
        """Open the output. Called once before the first chunk."""
        ...

    async def write(self, chunk: bytes) -> None:
        """Hand over one chunk. May block while the output catches up."""
        ...

    async def finish(self) -> None:
        """Let everything written play out, then return."""
        ...

    async def stop(self) -> None:
        """Go silent immediately, dropping whatever is still buffered."""
        ...


class NullSink:
    """Keeps the bytes, makes no sound.

    This is what runs in tests and on a machine with no ffplay. It still
    records what it was given, so a test can assert on how much audio actually
    made it out before a cancel landed.
    """

    def __init__(self) -> None:
        self.chunks: list[bytes] = []
        self.started = False
        self.finished = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def write(self, chunk: bytes) -> None:
        self.chunks.append(chunk)

    async def finish(self) -> None:
        self.finished = True

    async def stop(self) -> None:
        self.stopped = True

    @property
    def written(self) -> int:
        return sum(len(c) for c in self.chunks)


class FFplaySink:
    """Plays a stream by piping it into ``ffplay``.

    ffmpeg is already a dependency of this project for capture, which is the
    whole reason to prefer it here over a Python audio library: no new wheel,
    no PortAudio, no CoreAudio binding to argue with, and — the part that
    matters — the buffer that has to be thrown away on a preemption belongs to
    a separate process that can simply be killed. Killing a subprocess is a
    reliable way to stop sound; asking a library to drop its own queued frames
    usually is not.

    The flags fight ffplay's default appetite for buffering. It is built to
    play files smoothly, so it would rather collect a second of audio before
    starting; here a line that begins a second late has missed the moment.
    """

    def __init__(self, output_format: str = "mp3_22050_32", executable: str = "ffplay") -> None:
        self.output_format = output_format
        self.executable = executable
        self._proc: asyncio.subprocess.Process | None = None

    def _args(self) -> list[str]:
        """ffplay's input flags for one of ElevenLabs' output formats.

        Telling ffplay the container up front is not a micro-optimisation: left
        to itself it probes the stream, and probing means waiting for bytes it
        does not need before it plays the ones it has.
        """
        args = [
            self.executable,
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
        ]
        kind, _, rate = self.output_format.partition("_")
        if kind == "pcm":
            # Raw signed 16-bit mono, exactly as ElevenLabs sends it. Nothing
            # to probe and nothing to decode: the shortest path to a speaker.
            args += ["-f", "s16le", "-ar", rate or "24000", "-ac", "1"]
        elif kind == "mp3":
            args += ["-f", "mp3", "-probesize", "32", "-analyzeduration", "0"]
        elif kind == "ulaw":
            args += ["-f", "mulaw", "-ar", rate or "8000", "-ac", "1"]
        elif kind == "alaw":
            args += ["-f", "alaw", "-ar", rate or "8000", "-ac", "1"]
        # Anything else (opus, and whatever ElevenLabs adds later) is left to
        # ffplay to work out. Slower to start, but it plays.
        return [*args, "-i", "pipe:0"]

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            *self._args(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def write(self, chunk: bytes) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            return
        try:
            proc.stdin.write(chunk)
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            # ffplay died — the speaker's audio is gone but the match is not.
            log.warning("ffplay closed the pipe mid-line; dropping the rest of it")
            self._proc = None

    async def finish(self) -> None:
        """Close the pipe and wait for the last of the audio to be heard.

        The wait is the point. ``say`` returning is the director's signal that
        the channel is free, so returning while the tail of the line is still
        coming out of the speakers would let the other voice start on top of
        it. ``-autoexit`` makes ffplay quit once it has played everything, so
        waiting for the process is waiting for silence.
        """
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.stdin is not None:
            try:
                proc.stdin.close()
                await proc.stdin.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=DRAIN_TIMEOUT_S)
        except TimeoutError:
            log.warning("ffplay did not exit after the line ended; killing it")
            await self._kill(proc)

    async def stop(self) -> None:
        """Kill the player. Buffered audio dies with it, which is the idea."""
        proc, self._proc = self._proc, None
        if proc is not None:
            await self._kill(proc)

    @staticmethod
    async def _kill(proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        try:
            proc.kill()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=KILL_TIMEOUT_S)
        except TimeoutError:
            # A zombie ffplay is noise in `ps`, not noise in the commentary.
            log.warning("ffplay ignored SIGKILL")


def default_sink(output_format: str = "mp3_22050_32") -> AudioSink:
    """Where a real voice plays. Requires ffplay, and says so if it is missing.

    Degrading to :class:`NullSink` here would mean asking for sound and
    getting a silent match with one line in the log, which is a thing to
    discover at kickoff. Anyone who wants the silent path can ask for the
    printed voice instead.
    """
    if shutil.which("ffplay") is None:
        raise RuntimeError(
            "ffplay is not on PATH, so the voice cannot be heard. "
            "Install ffmpeg, or run with --voice log."
        )
    return FFplaySink(output_format=output_format)
