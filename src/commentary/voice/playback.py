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

There are two of them, and the difference is where the buffer lives. The
ffplay sink hands the bytes to another process; the PCM sink keeps them in
this one. Both can be silenced, but only one of them can say when the sound
has actually stopped, and that turns out to be what the rate limiter reads.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import shutil
import time
from typing import Any, Protocol

log = logging.getLogger(__name__)

#: How long to wait for a finished line to play out of the sound-card buffer
#: before giving up on it. Generous: a normal line drains in well under a
#: second, and hanging here would stall the whole director loop.
DRAIN_TIMEOUT_S = 5.0

#: How long to wait for a killed player to actually die. Very short — this is
#: on the preemption path, where every millisecond is audible.
KILL_TIMEOUT_S = 0.5

#: What ElevenLabs means by ``pcm_<rate>``: signed 16-bit little-endian, one
#: channel. Two bytes to a sample, and a sample is a frame.
PCM_SAMPLE_BYTES = 2
PCM_CHANNELS = 1

#: The format the voice asks for by default, spelled out here rather than
#: imported so that this module stays below ``elevenlabs`` in the import
#: order. 22 050 Hz is the middle of the three raw rates a free account may
#: have, and plenty for one close-mic voice.
DEFAULT_FORMAT = "pcm_22050"

#: How long to wait before asking the sound card again whether it has room.
#: Short enough never to be the reason a chunk is late, long enough that a
#: full buffer is not a spin loop.
POLL_S = 0.005

#: How long to keep offering audio to a card that is not taking any before
#: giving up on the line. A healthy card refuses for as long as its buffer
#: holds, which is a fraction of a second; one that refuses for this long has
#: stopped playing. macOS's audio stack really does wedge — killing ``say``
#: mid-utterance is one way to do it — and a director waiting politely on a
#: dead device never says anything again, which is a silent match rather than
#: a missing line.
STALL_S = 5.0

#: A little longer than the arithmetic says, before a finished line's stream
#: is torn down. Covers the few milliseconds between handing the last sample
#: to PortAudio and it reaching the speaker, which the byte count cannot see.
TAIL_S = 0.05


class AudioStalled(RuntimeError):
    """The output stopped taking audio and did not start again."""


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


def pcm_rate(output_format: str) -> int | None:
    """The sample rate of a ``pcm_<rate>`` format, or None if it is not one."""
    kind, _, rate = output_format.partition("_")
    if kind != "pcm" or not rate.isdigit():
        return None
    return int(rate)


class PcmStream(Protocol):
    """The part of a PortAudio output stream this module uses.

    Narrow on purpose: a test satisfies it in twenty lines with no sound card,
    which is the only way the arithmetic below is checkable at all.
    """

    @property
    def write_available(self) -> int:
        """Frames that can be written without waiting."""
        ...

    def start(self) -> None: ...

    def write(self, data: Any) -> Any: ...

    def stop(self) -> None:
        """Stop, once what is buffered has played."""
        ...

    def abort(self) -> None:
        """Stop now, discarding what is buffered."""
        ...

    def close(self) -> None: ...


class PcmSink:
    """Plays raw PCM through the sound card, in this process.

    The ffplay sink was measured costing about three seconds a line, and
    neither second of it was audio. One went on starting a process; the other
    went on :data:`DRAIN_TIMEOUT_S`, because ``-autoexit`` fires when an input
    *ends*, and a pipe that has merely been closed is not an input that has
    ended — so every finished line waited out the whole timeout and was then
    killed. The director holds the channel for as long as ``say`` takes and
    the rate limiter reads the seconds it reports, so that overhead was both a
    silence in the commentary and lines the run never earned.

    Keeping the audio in-process removes all of it. ``finish`` here is
    arithmetic — bytes handed over, over bytes a second — rather than a hope
    that another process will notice it has run out of input. There is no
    timeout because there is nothing to time out on.

    What it costs is that the buffer to throw away on a preemption is now
    ours. ``abort`` is PortAudio's answer to that: it drops what it is holding
    rather than playing it out, which is the whole difference between
    :meth:`stop` and :meth:`finish`.

    No thread blocks. Every write is sized to the room the card reports, so a
    full buffer is a short sleep on the event loop rather than a worker stuck
    inside PortAudio that a cancel cannot reach and that ``abort`` would be
    pulling the stream out from under.
    """

    def __init__(
        self,
        output_format: str = DEFAULT_FORMAT,
        *,
        open_stream: Any = None,
        poll_s: float = POLL_S,
        stall_s: float = STALL_S,
    ) -> None:
        rate = pcm_rate(output_format)
        if rate is None:
            raise ValueError(f"{output_format!r} is not a raw PCM format; use the ffplay sink")
        self.output_format = output_format
        self.rate = rate
        self.frame_bytes = PCM_SAMPLE_BYTES * PCM_CHANNELS
        self.bytes_per_second = float(rate * self.frame_bytes)
        self.poll_s = poll_s
        self.stall_s = stall_s
        self._open = open_stream if open_stream is not None else open_portaudio
        self._stream: PcmStream | None = None
        #: Bytes handed to the card so far.
        self.written = 0
        #: Monotonic time everything handed over will have finished playing
        #: at. Zero until the first chunk; the whole of :meth:`finish`'s wait.
        self.plays_until = 0.0
        self._remainder = b""

    @property
    def seconds_written(self) -> float:
        """How much speech has been handed over, in seconds of audio."""
        return self.written / self.bytes_per_second

    async def start(self) -> None:
        stream: PcmStream = self._open(self.rate)
        stream.start()
        self._stream = stream
        self.written = 0
        self.plays_until = 0.0
        self._remainder = b""

    async def write(self, chunk: bytes) -> None:
        """Hand a chunk to the card, a frame boundary at a time.

        A partial frame is held back rather than written. PortAudio counts in
        frames and would refuse the odd byte, and a chunk boundary landing
        mid-sample is a fact about how the HTTP response was split up on the
        way here, not about the speech.

        Raises :class:`AudioStalled` if the card takes nothing for
        :data:`STALL_S`. The pump turns that into a cut line, which is the
        only honest ending: a device that has stopped playing is not going to
        play this one, and waiting on it holds the channel for the rest of
        the match.
        """
        stream = self._stream
        if stream is None or not chunk:
            return
        data = self._remainder + chunk
        usable = len(data) - len(data) % self.frame_bytes
        self._remainder, data = data[usable:], data[:usable]
        refusing_since: float | None = None
        while data:
            room = stream.write_available * self.frame_bytes
            if room <= 0:
                # The card is full, which usually means it is still playing
                # what it already has. Waiting here is the back-pressure that
                # stops this reading a whole line into memory ahead of the
                # speaker — but only for as long as a full buffer could
                # plausibly last.
                now = time.monotonic()
                if refusing_since is None:
                    refusing_since = now
                elif now - refusing_since > self.stall_s:
                    raise AudioStalled(
                        f"the output took no audio for {self.stall_s:g}s; "
                        "the device has stopped playing"
                    )
                await asyncio.sleep(self.poll_s)
                continue
            refusing_since = None
            piece, data = data[:room], data[room:]
            stream.write(piece)
            self.written += len(piece)
            self.plays_until = (
                max(self.plays_until, time.monotonic()) + len(piece) / self.bytes_per_second
            )

    async def finish(self) -> None:
        """Wait for the audio to actually come out, then close the stream.

        The wait is ``plays_until``, maintained on the way in: each piece is
        queued behind whatever was still playing when it was written, or
        starts from now if the card had run dry in between. That second case
        is why this is not simply total bytes over the rate — a stream slower
        than real time underruns, and the sound then ends later than the
        bytes on their own would say.

        The stream is then aborted rather than stopped, even though stopping
        is the one that means "play out what is left". By this point there is
        nothing left: the sleep has already covered every byte handed over,
        plus :data:`TAIL_S` for the arithmetic being a few milliseconds out.
        And ``stop`` is a blocking drain — on a device that has wedged it
        never returns, and it would be holding the event loop, not a thread.
        A tail that cannot be clipped is worth less than a match that cannot
        hang.
        """
        stream, self._stream = self._stream, None
        if stream is None:
            return
        remaining = self.plays_until - time.monotonic()
        if remaining > 0:
            await asyncio.sleep(remaining + TAIL_S)
        try:
            stream.abort()
        finally:
            stream.close()

    async def stop(self) -> None:
        """Go silent now. What the card is holding is dropped, not played."""
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.abort()
        finally:
            stream.close()


def open_portaudio(rate: int) -> Any:
    """A real 16-bit mono output stream at ``rate``, through PortAudio.

    Imported here rather than at module scope because ``sounddevice`` is an
    optional extra and importing ``commentary.voice`` has to work without it.
    """
    import sounddevice

    return sounddevice.RawOutputStream(samplerate=rate, channels=PCM_CHANNELS, dtype="int16")


class FFplaySink:
    """Plays a stream by piping it into ``ffplay``.

    ffmpeg is already a dependency of this project for capture, which is why
    this was the first sink: no new wheel, no PortAudio, no CoreAudio binding
    to argue with, and — the part that looked decisive — the buffer to throw
    away on a preemption belongs to a separate process that can simply be
    killed.

    What that turned out to cost is in :class:`PcmSink`'s docstring. This one
    stays because it is the only thing here that can play an mp3, and
    ``mp3_22050_32`` is a quarter of the bytes of raw PCM: over a connection
    that delivers audio slower than it plays, that is the difference between a
    line and a stutter. :func:`default_sink` picks between the two by format.

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
        waiting for the process is waiting for silence — in principle. In
        practice it waits for an *input* to end, and a closed pipe is not one
        it recognises as ended, so a piped line reliably burns the whole
        timeout and is then killed. That measurement is what :class:`PcmSink`
        exists for.
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


def default_sink(output_format: str = DEFAULT_FORMAT) -> AudioSink:
    """Where a real voice plays: PortAudio for raw PCM, ffplay for the rest.

    Both branches refuse rather than degrade. Falling back to
    :class:`NullSink` would mean asking for sound and getting a silent match
    with one line in the log, which is a thing to discover at kickoff. Anyone
    who wants the silent path can ask for the printed voice instead.
    """
    if pcm_rate(output_format) is not None:
        if importlib.util.find_spec("sounddevice") is None:
            raise RuntimeError(
                "sounddevice is not installed, so raw PCM has nowhere to go. "
                "Install it with `uv sync --extra audio`, or run with --voice log."
            )
        return PcmSink(output_format)
    if shutil.which("ffplay") is None:
        raise RuntimeError(
            "ffplay is not on PATH, so the voice cannot be heard. "
            "Install ffmpeg, or run with --voice log."
        )
    return FFplaySink(output_format=output_format)
