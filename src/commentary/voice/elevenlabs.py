"""ElevenLabs, streamed, with the ability to stop mid-word.

The whole reason this speaker streams is the director. It hands a line over
and keeps the right to take it back a quarter of a second later, because a
goal has gone in and whatever the analyst was halfway through is now wrong.
Rendering a clip and then playing it would honour that request only between
lines, which is the same as not honouring it. So the audio arrives in chunks,
the cancel event is checked between every one of them, and the player is
killed rather than asked politely.

Everything else here follows from that. The model is the fast one, not the
pretty one. The audio goes to a subprocess because a subprocess can be killed.
A line that fails mid-flight returns a cut-off :class:`Utterance` instead of
raising, because a match that goes quiet is a disappointment and a match that
stops is a bug. A speaker with no key, on the other hand, cannot ever say
anything, so it refuses to be built rather than going silently mute.

How hard a line is said is the other thing this file decides. Every beat
carries an excitement from the phrasing stage, and until it was read here a
goal went out at the same library defaults as a throw-in.
:class:`~commentary.config.VoiceConfig` turns that number into the
``voice_settings`` on the request, per seat, and the settings that were sent
are recorded on the :class:`Utterance` so the trace can answer afterwards why
a line came out the way it did.

The ``elevenlabs`` package is an optional extra and is imported inside the
function that needs it, so importing ``commentary.voice`` works on a machine
that has never installed it. Note that this module is itself called
``elevenlabs``: absolute imports mean ``from elevenlabs.client import ...``
below reaches the installed package, not this file.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from commentary.config import VoiceConfig
from commentary.schemas import Beat, Voice
from commentary.voice.playback import AudioSink, NullSink, PcmSink, default_sink
from commentary.voice.shaping import shape_for_voice
from commentary.voice.speaker import WORDS_PER_SECOND, Utterance

log = logging.getLogger(__name__)

#: Flash is the real-time model: roughly 75 ms to first byte against turbo's
#: ~250 ms. Turbo v2.5 does sound a shade better and is a one-word change via
#: ``model_id``, but the latency budget is already spent elsewhere — the
#: delay buffer, the vision call, the fact gate — and this is the one link in
#: the chain where a quarter of a second buys nothing but polish.
MODEL_ID = "eleven_flash_v2_5"

#: Raw 16-bit mono at 22 050 Hz. Not an mp3, and that is the point: raw
#: samples go straight into the sound card, so there is no decoder to feed and
#: no player process to start, and the sink knows exactly how much speech it
#: is holding. That last part is what the mp3 path could not do — see
#: :class:`~commentary.voice.playback.PcmSink` for the three seconds a line it
#: was costing. ``pcm_16000`` and ``pcm_24000`` are the other two rates a free
#: account may ask for; ``pcm_44100`` needs a paid one. ``mp3_22050_32`` is
#: still here and still a quarter of the bytes, which is the trade to make on
#: a connection too slow to deliver PCM in real time: pass it as
#: ``output_format`` and the ffplay sink takes over.
OUTPUT_FORMAT = "pcm_22050"

#: Stock library voices, picked to be told apart instantly rather than to be
#: the best two voices on the platform. The caller is a British male with some
#: bite to him; the analyst is a British female, calmer. Different register and
#: different pitch means a listener never has to work out who is talking, which
#: matters more in a two-voice booth than either voice does on its own. Taste
#: differs, so both are overridable from the environment.
CALLER_VOICE = "JBFqnCBsd6RMkjVDRZzb"  # George
ANALYST_VOICE = "Xb7hH8MSUJpSbSDYk0k2"  # Alice

#: One line is at most a couple of dozen words; anything slower than this is a
#: hung connection, not a long sentence.
REQUEST_TIMEOUT_S = 30.0

#: What the excitement curve produced for one line, or None when the curve is
#: switched off and the voice is to play at its own library defaults.
VoiceSettings = dict[str, float | bool] | None

#: Given (text, voice id), an async stream of audio chunks. The seam tests
#: inject through, and the reason nothing here monkeypatches a client.
StreamFactory = Callable[[str, str], AsyncIterator[bytes]]

#: Somewhere for those chunks to go. Called once per line.
SinkFactory = Callable[[], AudioSink]


class VoiceUnavailable(RuntimeError):
    """There is no key, so this speaker could never make a sound."""


class ElevenLabsSpeaker:
    """A two-voice ElevenLabs speaker that can be cut off mid-word.

    Implements the :class:`~commentary.voice.speaker.Speaker` protocol, so it
    drops into the director wherever :class:`LogSpeaker` sits.

    Both the stream and the sink are constructor arguments. That is not only
    for tests: it is the reason there is nothing private in here worth
    patching. Hand it a fake chunk iterator and it never touches the network;
    hand it a :class:`~commentary.voice.playback.NullSink` and it makes no
    sound; hand it neither and it does the real thing.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        caller_voice: str | None = None,
        analyst_voice: str | None = None,
        model_id: str = MODEL_ID,
        output_format: str = OUTPUT_FORMAT,
        stream: StreamFactory | None = None,
        sink: SinkFactory | None = None,
        words_per_second: float = WORDS_PER_SECOND,
        timeout_s: float = REQUEST_TIMEOUT_S,
        curve: VoiceConfig | None = None,
        shaping: bool | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("ELEVENLABS_API_KEY", "")
        if not self.api_key and stream is None:
            raise VoiceUnavailable(
                "no ELEVENLABS_API_KEY: set one in .env, or run with --voice log"
            )
        self.caller_voice = caller_voice or os.getenv("ELEVENLABS_CALLER_VOICE") or CALLER_VOICE
        self.analyst_voice = analyst_voice or os.getenv("ELEVENLABS_ANALYST_VOICE") or ANALYST_VOICE
        self.model_id = model_id
        self.output_format = output_format
        self.words_per_second = words_per_second
        self.timeout_s = timeout_s
        # Built here rather than taken from SETTINGS so that the environment
        # is read when a speaker is made: a sweep sets VOICE_CALLER_STYLE_HIGH
        # and builds one, and the process it is running in need not restart.
        self.curve = VoiceConfig() if curve is None else curve
        self.shaping = self.curve.shaping if shaping is None else shaping
        self.said: list[Utterance] = []
        self._stream = stream
        self._sink = sink if sink is not None else lambda: default_sink(self.output_format)
        self._client: Any | None = None
        self._http: Any | None = None

    def voice_id(self, voice: Voice) -> str:
        return self.analyst_voice if voice is Voice.ANALYST else self.caller_voice

    # -- saying things ---------------------------------------------------

    async def say(self, beat: Beat, cancel: asyncio.Event) -> Utterance:
        """Speak a beat, stopping the instant ``cancel`` is set.

        Returns when the sound has actually finished — not when the last byte
        was handed over. The director speaks one beat at a time and treats this
        returning as the channel being free, so returning early would let the
        caller start on top of the analyst's last three words.
        """
        started = time.monotonic()
        text = self.text_for(beat)
        settings = self.curve.settings_for(beat.voice.value, beat.excitement)
        sink = self._sink()
        audio_started: float | None = None
        completed = False
        try:
            stream = self._open_stream(text, self.voice_id(beat.voice), settings)
            await sink.start()
            audio_started, completed = await self._pump(stream, sink, cancel)
        except Exception as exc:
            # A voice failing is not a match failing. The director will call
            # again in a few seconds and the next line may well get through.
            log.warning("ElevenLabs line failed (%s): %s", type(exc).__name__, exc)
            await sink.stop()
            return self._record(
                beat,
                spoken=self._estimate(text, audio_started),
                started=started,
                audio_started=audio_started,
                completed=False,
                settings=settings,
            )

        if completed:
            await sink.finish()
            return self._record(
                beat,
                spoken=text,
                started=started,
                audio_started=audio_started,
                completed=True,
                settings=settings,
            )
        await sink.stop()
        return self._record(
            beat,
            spoken=self._estimate(text, audio_started),
            started=started,
            audio_started=audio_started,
            completed=False,
            settings=settings,
        )

    def text_for(self, beat: Beat) -> str:
        """The words that go to the model: the beat's own, unless shaped.

        The analyst is never shaped. Exclamation marks belong to the caller,
        and the second seat exists in order not to sound like the first one.
        """
        if self.shaping and beat.voice is Voice.CALLER:
            return shape_for_voice(beat.text, beat.excitement)
        return beat.text

    async def _pump(
        self, stream: AsyncIterator[bytes], sink: AudioSink, cancel: asyncio.Event
    ) -> tuple[float | None, bool]:
        """Move chunks from the stream to the sink until one of them runs out.

        Both the wait for the next chunk and the wait for the sink to accept it
        race the cancel event, because either can block for longer than a
        preemption is allowed to take — the network between chunks, and the
        player's pipe once it has as much audio as it can hold.

        A stream that breaks halfway is handled here rather than thrown, so
        that a line which was half heard is still reported as half heard.

        Returns when audio first went out (or None if none did) and whether the
        stream finished of its own accord.
        """
        waiting = asyncio.ensure_future(cancel.wait())
        audio_started: float | None = None
        try:
            while not cancel.is_set():
                chunk_task = asyncio.ensure_future(anext(stream))
                if not await _race(chunk_task, waiting):
                    return audio_started, False
                try:
                    chunk = chunk_task.result()
                except StopAsyncIteration:
                    return audio_started, True
                except Exception as exc:
                    log.warning("ElevenLabs stream broke (%s): %s", type(exc).__name__, exc)
                    return audio_started, False
                if not chunk:
                    continue
                if audio_started is None:
                    audio_started = time.monotonic()
                write_task = asyncio.ensure_future(sink.write(chunk))
                if not await _race(write_task, waiting):
                    return audio_started, False
                try:
                    write_task.result()
                except Exception as exc:
                    log.warning("audio output failed (%s): %s", type(exc).__name__, exc)
                    return audio_started, False
            return audio_started, False
        finally:
            waiting.cancel()
            # Abandoning the generator mid-response is what closes the HTTP
            # connection; without it the rest of the line keeps being paid for
            # and downloaded to nowhere.
            await _aclose(stream)

    def _open_stream(
        self, text: str, voice_id: str, settings: VoiceSettings
    ) -> AsyncIterator[bytes]:
        if self._stream is not None:
            return self._stream(text, voice_id)
        return self._api_stream(text, voice_id, settings)

    def _api_stream(
        self, text: str, voice_id: str, settings: VoiceSettings
    ) -> AsyncIterator[bytes]:
        """The real thing. Imported here so the package stays optional."""
        import httpx
        from elevenlabs.client import AsyncElevenLabs

        if self._client is None:
            # Our own httpx client, purely so :meth:`aclose` has something to
            # close; the SDK's default one outlives the speaker otherwise.
            self._http = httpx.AsyncClient(timeout=self.timeout_s)
            self._client = AsyncElevenLabs(api_key=self.api_key, httpx_client=self._http)
        client: Any = self._client
        # Omitted rather than sent as null when the curve is off: an absent
        # field is the voice's own library settings, which is exactly what
        # "off" has to mean, and a null is not guaranteed to be read that way.
        extra: dict[str, Any] = {} if settings is None else {"voice_settings": settings}
        chunks: AsyncIterator[bytes] = client.text_to_speech.stream(
            voice_id=voice_id,
            text=text,
            model_id=self.model_id,
            output_format=self.output_format,
            **extra,
        )
        return chunks

    # -- bookkeeping -----------------------------------------------------

    def _estimate(self, text: str, audio_started: float | None) -> str:
        """Roughly the words that were actually heard before the cut.

        It is an estimate and cannot be anything else. A streaming TTS API
        hands back audio bytes, not word boundaries — ElevenLabs will return
        character-level timings, but only from a different endpoint that wraps
        the audio in JSON and gives up the low latency this speaker exists for.
        And even exact timings would be timings of the *stream*, while what the
        listener heard depends on how much the player still had buffered when
        it was killed.

        So: elapsed playback time multiplied by a plausible speaking rate. It
        is used for the trace and the eval transcript, where being a word or
        two out about where a preempted line stopped costs nothing. Nothing
        downstream should treat it as a transcript.
        """
        if audio_started is None:
            return ""
        words = text.split()
        heard = int((time.monotonic() - audio_started) * self.words_per_second)
        return " ".join(words[: max(0, heard)])

    def _record(
        self,
        beat: Beat,
        *,
        spoken: str,
        started: float,
        audio_started: float | None,
        completed: bool,
        settings: VoiceSettings = None,
    ) -> Utterance:
        """One line's account of itself, including how long nothing happened.

        ``seconds`` is the whole call and is what the director bills the
        channel for. On its own it cannot say whether a long line was long
        speech or a slow first byte, and those want opposite fixes: the first
        is the model writing too much, the second is the network or the
        format. ``first_audio_s`` separates them, and it is None when no audio
        ever arrived rather than zero, because a line nobody heard did not
        reach the speakers instantly.

        ``voice_settings`` is the third of those numbers and the one being
        tuned: a line that came out flat is not diagnosable from the beat's
        excitement, only from what the curve turned it into.
        """
        utterance = Utterance(
            beat=beat,
            spoken=spoken,
            seconds=time.monotonic() - started,
            completed=completed,
            first_audio_s=None if audio_started is None else audio_started - started,
            voice_settings=settings,
        )
        self.said.append(utterance)
        return utterance

    async def aclose(self) -> None:
        http, self._http = self._http, None
        self._client = None
        if http is not None:
            await http.aclose()


async def _race[T](task: asyncio.Task[T], cancel: asyncio.Future[Any]) -> bool:
    """Wait for ``task``, abandoning it if ``cancel`` resolves first.

    True if the task won. If it lost it is cancelled and its result — or its
    failure — is dropped on the floor, which is the correct treatment for work
    nobody is waiting for any more.
    """
    both: set[asyncio.Future[Any]] = {task, cancel}
    done, _ = await asyncio.wait(both, return_when=asyncio.FIRST_COMPLETED)
    if task in done:
        return True
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task
    return False


async def _aclose(stream: AsyncIterator[bytes]) -> None:
    closer = getattr(stream, "aclose", None)
    if closer is None:
        return
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await closer()


__all__ = [
    "ANALYST_VOICE",
    "CALLER_VOICE",
    "MODEL_ID",
    "OUTPUT_FORMAT",
    "AudioSink",
    "ElevenLabsSpeaker",
    "NullSink",
    "PcmSink",
    "SinkFactory",
    "StreamFactory",
    "VoiceSettings",
    "VoiceUnavailable",
]
