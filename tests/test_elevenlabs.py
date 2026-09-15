"""The speaker's job is mostly about stopping.

Nothing here touches the network. The chunk stream and the audio sink are both
constructor arguments, so a fake iterator and a :class:`NullSink` exercise the
real pump loop — the cancel checks, the abandonment, the estimate — without a
key, a socket, or a sound card.

The PCM sink is the same trick one layer down. Its stream is a constructor
argument too, so the arithmetic that replaced ffplay's timeout — how long the
audio handed over will take to come out, and how little a cancel waits — is
checkable against a fake card that keeps its own books and makes no sound.
"""

from __future__ import annotations

import asyncio
import importlib.util
import shutil
import subprocess
import sys
import time
import types
from collections.abc import AsyncIterator, Sequence

import pytest

from commentary.config import VoiceConfig
from commentary.schemas import Beat, Voice
from commentary.voice import ElevenLabsSpeaker, VoiceUnavailable
from commentary.voice.playback import (
    DRAIN_TIMEOUT_S,
    PCM_SAMPLE_BYTES,
    AudioStalled,
    FFplaySink,
    NullSink,
    PcmSink,
    default_sink,
    pcm_rate,
)


def beat(text: str, *, voice: Voice = Voice.CALLER, excitement: float = 0.0) -> Beat:
    now = time.monotonic()
    return Beat(
        id="t1", voice=voice, text=text, video_ts=0.0, created_ts=now, excitement=excitement
    )


class FakeStream:
    """A scripted audio stream that records how much of it was actually read."""

    def __init__(
        self,
        chunks: Sequence[bytes],
        *,
        delay_s: float = 0.0,
        fail_at: int | None = None,
        stall_at: int | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.delay_s = delay_s
        self.fail_at = fail_at
        self.stall_at = stall_at
        self.yielded = 0
        self.closed = False
        self.calls: list[tuple[str, str]] = []

    def __call__(self, text: str, voice_id: str) -> AsyncIterator[bytes]:
        self.calls.append((text, voice_id))
        return self._gen()

    async def _gen(self) -> AsyncIterator[bytes]:
        try:
            for i, chunk in enumerate(self.chunks):
                if self.fail_at is not None and i == self.fail_at:
                    raise RuntimeError("the stream died")
                if self.stall_at is not None and i == self.stall_at:
                    await asyncio.sleep(30)
                if self.delay_s:
                    await asyncio.sleep(self.delay_s)
                self.yielded += 1
                yield chunk
        finally:
            self.closed = True

    @property
    def voices(self) -> list[str]:
        return [voice for _text, voice in self.calls]


class GatedSink(NullSink):
    """A sink that pulls the plug itself once it has been given enough audio.

    Sidesteps sleeping for a cancel: the event is set from inside the pump's
    own write, so the test knows exactly which chunk the cut lands on.
    """

    def __init__(self, cancel: asyncio.Event, after: int) -> None:
        super().__init__()
        self.cancel = cancel
        self.after = after

    async def write(self, chunk: bytes) -> None:
        await super().write(chunk)
        if len(self.chunks) >= self.after:
            self.cancel.set()


def speaker(stream: FakeStream, sink: NullSink, **kwargs: object) -> ElevenLabsSpeaker:
    return ElevenLabsSpeaker(
        api_key="test-key",
        stream=stream,
        sink=lambda: sink,
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_the_two_voices_are_actually_two_voices() -> None:
    stream = FakeStream([b"a", b"b"])
    sink = NullSink()
    spk = speaker(stream, sink, caller_voice="caller-id", analyst_voice="analyst-id")

    await spk.say(beat("down the left", voice=Voice.CALLER), asyncio.Event())
    await spk.say(beat("they are pressing", voice=Voice.ANALYST), asyncio.Event())

    assert stream.voices == ["caller-id", "analyst-id"]
    assert spk.voice_id(Voice.CALLER) != spk.voice_id(Voice.ANALYST)


@pytest.mark.asyncio
async def test_a_finished_line_reports_the_whole_text() -> None:
    stream = FakeStream([b"a" * 64] * 4)
    sink = NullSink()
    spk = speaker(stream, sink)

    line = "Arsenal break down the left and it is worked inside"
    utterance = await spk.say(beat(line), asyncio.Event())

    assert utterance.completed
    assert utterance.spoken == line
    assert stream.yielded == 4
    assert sink.written == 4 * 64
    # The line played out rather than being killed, which is what keeps the
    # other voice from starting on top of its last few words.
    assert sink.finished and not sink.stopped


@pytest.mark.asyncio
async def test_a_cancel_stops_the_sound_and_leaves_the_rest_of_the_stream_unread() -> None:
    cancel = asyncio.Event()
    stream = FakeStream([bytes([i]) * 32 for i in range(10)])
    sink = GatedSink(cancel, after=2)
    spk = speaker(stream, sink)

    utterance = await spk.say(beat("one two three four five six seven eight"), cancel)

    assert not utterance.completed
    assert stream.yielded == 2, "the rest of the line should never have been downloaded"
    assert stream.closed, "the stream should have been abandoned, not left open"
    assert sink.stopped and not sink.finished
    assert len(sink.chunks) == 2


@pytest.mark.asyncio
async def test_a_cancel_does_not_wait_for_a_chunk_that_never_arrives() -> None:
    """The cut has to land during the silence between chunks too."""
    cancel = asyncio.Event()
    stream = FakeStream([b"a", b"b", b"c"], stall_at=2)
    spk = speaker(stream, NullSink())

    async def cut() -> None:
        await asyncio.sleep(0.05)
        cancel.set()

    asyncio.create_task(cut())
    started = time.monotonic()
    utterance = await spk.say(beat("a long line that is going nowhere"), cancel)

    assert not utterance.completed
    assert time.monotonic() - started < 1.0, "it waited on the stalled stream"
    assert stream.yielded == 2


@pytest.mark.asyncio
async def test_a_stream_that_breaks_does_not_take_the_match_down() -> None:
    stream = FakeStream([b"a", b"b", b"c"], fail_at=1)
    sink = NullSink()
    spk = speaker(stream, sink)

    utterance = await spk.say(beat("it is in the back of the net"), asyncio.Event())

    assert not utterance.completed
    assert sink.stopped
    assert spk.said[-1] is utterance
    # What did arrive before the break was still played, and the beat is still
    # on the record — the director needs to see a cut, not an exception.
    assert sink.chunks == [b"a"]
    assert utterance.beat.text == "it is in the back of the net"


@pytest.mark.asyncio
async def test_a_stream_that_never_opens_does_not_take_the_match_down() -> None:
    def explode(_text: str, _voice: str) -> AsyncIterator[bytes]:
        raise ConnectionError("no route to ElevenLabs")

    spk = ElevenLabsSpeaker(api_key="test-key", stream=explode, sink=NullSink)
    utterance = await spk.say(beat("a line nobody hears"), asyncio.Event())

    assert not utterance.completed
    assert utterance.spoken == ""


def test_no_key_refuses_to_build_a_speaker_that_could_never_speak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(VoiceUnavailable, match="ELEVENLABS_API_KEY"):
        ElevenLabsSpeaker()


@pytest.mark.asyncio
async def test_the_cut_off_estimate_covers_roughly_what_was_heard() -> None:
    """An estimate, not a transcript — so this only checks it is plausible."""
    cancel = asyncio.Event()
    stream = FakeStream([b"x" * 32] * 40, delay_s=0.01)
    sink = GatedSink(cancel, after=20)
    spk = speaker(stream, sink, words_per_second=10.0)

    line = " ".join(f"w{i}" for i in range(40))
    utterance = await spk.say(beat(line), cancel)

    spoken = utterance.spoken.split()
    assert 0 < len(spoken) < 40
    assert line.startswith(utterance.spoken)


@pytest.mark.asyncio
async def test_the_real_call_asks_for_the_right_voice_model_and_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one path a fake stream cannot cover: the SDK call itself.

    Standing a fake module in for ``elevenlabs.client`` is the only way to see
    the keyword arguments without spending a key on it, and a typo in one of
    them would otherwise only show up on matchday.
    """
    seen: dict[str, object] = {}

    class FakeTTS:
        def stream(self, **kwargs: object) -> AsyncIterator[bytes]:
            seen.update(kwargs)
            return FakeStream([b"audio"])._gen()

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            seen["api_key"] = kwargs.get("api_key")
            self.text_to_speech = FakeTTS()

    module = types.ModuleType("elevenlabs.client")
    module.AsyncElevenLabs = FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "elevenlabs.client", module)

    spk = ElevenLabsSpeaker(
        api_key="sk-test",
        analyst_voice="analyst-id",
        model_id="eleven_flash_v2_5",
        output_format="pcm_24000",
        sink=NullSink,
    )
    utterance = await spk.say(beat("a measured point", voice=Voice.ANALYST), asyncio.Event())
    await spk.aclose()

    assert utterance.completed
    assert seen["api_key"] == "sk-test"
    assert seen["voice_id"] == "analyst-id"
    assert seen["model_id"] == "eleven_flash_v2_5"
    assert seen["output_format"] == "pcm_24000"
    assert seen["text"] == "a measured point"


def test_importing_the_voice_package_does_not_need_elevenlabs() -> None:
    """The extra is optional, so the import must not be at module scope."""
    code = (
        "import sys; sys.modules['elevenlabs'] = None; "
        "import commentary.voice; "
        "from commentary.voice import ElevenLabsSpeaker; "
        "assert ElevenLabsSpeaker(api_key='a-key') is not None; print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_ffplay_is_told_the_format_so_it_does_not_stop_to_probe() -> None:
    pcm = FFplaySink(output_format="pcm_24000")._args()
    assert pcm[pcm.index("-f") + 1] == "s16le"
    assert pcm[pcm.index("-ar") + 1] == "24000"
    assert pcm[-2:] == ["-i", "pipe:0"]

    mp3 = FFplaySink(output_format="mp3_22050_32")._args()
    assert mp3[mp3.index("-f") + 1] == "mp3"
    assert "-autoexit" in mp3 and "-nodisp" in mp3


@pytest.mark.asyncio
async def test_stopping_a_player_that_never_started_is_harmless() -> None:
    sink = FFplaySink()
    await sink.write(b"nothing is listening")
    await sink.stop()
    await sink.finish()


# -- the PCM sink -------------------------------------------------------

RATE = 22050
#: Bytes of ``pcm_22050`` in one second of speech.
SECOND = RATE * PCM_SAMPLE_BYTES


class FakeCard:
    """A sound card that keeps its books and makes no sound.

    ``write_available`` is the part that matters: it is finite and only goes
    back up when the test says the card has played something, which is what
    makes the sink's back-pressure real without a device. Writing past it
    fails the test rather than being quietly absorbed, because a sink that
    overruns a real card is a sink that drops audio.
    """

    def __init__(self, room_frames: int = 1 << 20) -> None:
        self._room = room_frames
        self.written = b""
        self.started = False
        self.stopped = False
        self.aborted = False
        self.closed = False

    @property
    def write_available(self) -> int:
        return self._room

    def start(self) -> None:
        self.started = True

    def write(self, data: bytes) -> None:
        frames = len(data) // PCM_SAMPLE_BYTES
        assert frames <= self._room, "wrote more than the card said it had room for"
        self.written += bytes(data)
        self._room -= frames

    def play(self, frames: int) -> None:
        """The card gets through some of what it is holding."""
        self._room += frames

    def stop(self) -> None:
        self.stopped = True

    def abort(self) -> None:
        self.aborted = True

    def close(self) -> None:
        self.closed = True


def pcm_sink(card: FakeCard, **kwargs: float) -> PcmSink:
    return PcmSink("pcm_22050", open_stream=lambda _rate: card, **kwargs)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_finished_pcm_line_waits_for_its_own_audio_and_not_for_a_timeout() -> None:
    """The measurement this sink exists for.

    ffplay's ``-autoexit`` never fired on a closed pipe, so every finished
    line sat out :data:`DRAIN_TIMEOUT_S` and was then killed — three seconds
    of held channel for two seconds of speech. Here the wait is the audio.
    """
    card = FakeCard()
    sink = pcm_sink(card)
    await sink.start()
    await sink.write(b"\0" * (RATE // 4 * PCM_SAMPLE_BYTES))

    started = time.monotonic()
    await sink.finish()
    waited = time.monotonic() - started

    assert waited == pytest.approx(0.25, abs=0.1)
    assert waited < DRAIN_TIMEOUT_S
    assert sink.seconds_written == pytest.approx(0.25, abs=1e-3)
    # The wait covered the audio, so the stream is torn down rather than
    # asked to drain: a blocking drain on a wedged device never returns.
    assert card.aborted and card.closed


@pytest.mark.asyncio
async def test_a_cut_line_drops_what_the_card_is_holding_instead_of_playing_it() -> None:
    card = FakeCard()
    sink = pcm_sink(card)
    await sink.start()
    await sink.write(b"\0" * SECOND)

    started = time.monotonic()
    await sink.stop()

    assert time.monotonic() - started < 0.05, "a preemption waited out the audio it was cancelling"
    assert card.aborted and card.closed


@pytest.mark.asyncio
async def test_the_sink_writes_only_as_much_as_the_card_says_it_has_room_for() -> None:
    """A full card is waited on, on the event loop, not overrun and not blocked.

    Nothing here may occupy a thread: a write stuck inside PortAudio is a
    write a cancel cannot reach, and aborting the stream out from under one is
    how this crashes rather than goes quiet.
    """
    card = FakeCard(room_frames=100)
    sink = pcm_sink(card, poll_s=0.001)
    await sink.start()

    async def play() -> None:
        for _ in range(2):
            await asyncio.sleep(0.01)
            card.play(100)

    playing = asyncio.create_task(play())
    await sink.write(b"\0" * 600)  # 300 frames into a card that holds 100
    await playing

    assert len(card.written) == 600
    assert sink.written == 600


@pytest.mark.asyncio
async def test_a_chunk_that_ends_mid_sample_holds_the_odd_byte_back() -> None:
    # Where a chunk boundary falls is a fact about how the HTTP response was
    # split up, not about the speech. PortAudio counts whole frames.
    card = FakeCard()
    sink = pcm_sink(card)
    await sink.start()

    await sink.write(b"abc")
    assert card.written == b"ab"

    await sink.write(b"de")
    assert card.written == b"abcd"


@pytest.mark.asyncio
async def test_a_stream_slower_than_real_time_still_ends_when_its_audio_does() -> None:
    """The card runs dry between chunks, so the sound ends later than the bytes.

    Total bytes over the rate would say a fifth of a second here. The answer
    is nearer half of one, because the second chunk could not start playing
    until it arrived.
    """
    card = FakeCard()
    sink = pcm_sink(card)
    await sink.start()

    await sink.write(b"\0" * (SECOND // 10))
    await asyncio.sleep(0.25)  # longer than the tenth of a second just written
    await sink.write(b"\0" * (SECOND // 10))

    started = time.monotonic()
    await sink.finish()

    assert time.monotonic() - started == pytest.approx(0.1, abs=0.06)


@pytest.mark.asyncio
async def test_a_sink_nothing_was_written_to_neither_waits_nor_complains() -> None:
    card = FakeCard()
    sink = pcm_sink(card)
    await sink.start()

    started = time.monotonic()
    await sink.finish()

    assert time.monotonic() - started < 0.2
    assert card.closed
    # And a second finish, or a stop after one, is a no-op rather than a crash.
    await sink.finish()
    await sink.stop()


@pytest.mark.asyncio
async def test_a_card_that_has_stopped_playing_ends_the_line_instead_of_the_match() -> None:
    """macOS's audio stack does wedge, and a polite wait is then permanent.

    The director holds the channel until ``say`` returns. A sink that waits
    for room on a dead device never returns, so the commentary does not
    resume when the next beat arrives, or the one after that — the match goes
    silent for good rather than losing one line.
    """
    card = FakeCard(room_frames=0)  # takes nothing, ever
    sink = pcm_sink(card, poll_s=0.001, stall_s=0.05)
    await sink.start()

    started = time.monotonic()
    with pytest.raises(AudioStalled, match="stopped playing"):
        await sink.write(b"\0" * SECOND)

    assert time.monotonic() - started < 1.0
    assert card.written == b""


@pytest.mark.asyncio
async def test_a_stalled_output_is_reported_as_a_cut_line_and_not_as_a_crash() -> None:
    # What the director has to see. An exception out of `say` would take the
    # whole runtime down over one unplayable line.
    class StallingSink(NullSink):
        async def write(self, chunk: bytes) -> None:
            raise AudioStalled("the output took no audio for 5s")

    sink = StallingSink()
    spk = speaker(FakeStream([b"a" * 64] * 4), sink)

    utterance = await spk.say(beat("a line into a dead speaker"), asyncio.Event())

    assert not utterance.completed
    assert sink.stopped


def test_a_raw_format_is_recognised_by_its_name_and_an_mp3_is_not() -> None:
    assert pcm_rate("pcm_22050") == RATE
    assert pcm_rate("pcm_24000") == 24000
    assert pcm_rate("mp3_22050_32") is None
    assert pcm_rate("ulaw_8000") is None


def test_asking_for_a_raw_format_from_the_pcm_sink_is_the_only_thing_it_accepts() -> None:
    with pytest.raises(ValueError, match="not a raw PCM format"):
        PcmSink("mp3_22050_32")


@pytest.mark.skipif(
    importlib.util.find_spec("sounddevice") is None, reason="the audio extra is not installed"
)
def test_the_format_the_voice_asks_for_picks_the_sink_that_can_play_it() -> None:
    # Constructing either is cheap and opens no device; the choice is the
    # whole of what is being asserted.
    assert isinstance(default_sink("pcm_22050"), PcmSink)
    if shutil.which("ffplay") is not None:
        assert isinstance(default_sink("mp3_22050_32"), FFplaySink)


@pytest.mark.asyncio
async def test_a_line_reports_how_long_the_listener_waited_for_the_first_sound() -> None:
    """``seconds`` cannot answer this, and the two want opposite fixes.

    Five seconds of channel for a six-word line is the model writing too much
    if the sound started at once, and the network or the format if it did not.
    """
    stream = FakeStream([b"a" * 64] * 4, delay_s=0.05)
    spk = speaker(stream, NullSink())

    utterance = await spk.say(beat("a slow start"), asyncio.Event())

    assert utterance.first_audio_s is not None
    assert utterance.first_audio_s == pytest.approx(0.05, abs=0.04)
    assert utterance.first_audio_s < utterance.seconds


@pytest.mark.asyncio
async def test_a_line_nobody_heard_reports_no_first_audio_rather_than_none_at_all() -> None:
    # None, not zero: a line that never made a sound did not reach the
    # speakers instantly, and a zero here would average into the latency as
    # though it had.
    def explode(_text: str, _voice: str) -> AsyncIterator[bytes]:
        raise ConnectionError("no route to ElevenLabs")

    spk = ElevenLabsSpeaker(api_key="test-key", stream=explode, sink=NullSink)
    utterance = await spk.say(beat("a line nobody hears"), asyncio.Event())

    assert utterance.first_audio_s is None


# -- the excitement curve -----------------------------------------------


class FakeSdk:
    """Stands in for the installed package, keeping every request body.

    The kwargs are the request: a typo in one of them, or a setting quietly
    not being sent, would otherwise only show up on matchday.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self.calls

        class FakeTTS:
            def stream(self, **kwargs: object) -> AsyncIterator[bytes]:
                calls.append(dict(kwargs))
                return FakeStream([b"audio"])._gen()

        class FakeClient:
            def __init__(self, **_kwargs: object) -> None:
                self.text_to_speech = FakeTTS()

        module = types.ModuleType("elevenlabs.client")
        module.AsyncElevenLabs = FakeClient  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "elevenlabs.client", module)


@pytest.mark.asyncio
async def test_the_request_body_says_how_hard_the_line_should_be_said(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A goal and a throw-in went out at the same library defaults until now."""
    sdk = FakeSdk()
    sdk.install(monkeypatch)
    spk = ElevenLabsSpeaker(api_key="sk-test", sink=NullSink, curve=VoiceConfig())

    await spk.say(beat("Mbappé! Buried!", excitement=1.0), asyncio.Event())
    await spk.say(beat("France push forward down the left.", excitement=0.1), asyncio.Event())
    await spk.aclose()

    goal = sdk.calls[0]["voice_settings"]
    buildup = sdk.calls[1]["voice_settings"]
    assert isinstance(goal, dict) and isinstance(buildup, dict)
    assert goal["stability"] < buildup["stability"]
    assert goal["style"] > buildup["style"]
    assert goal["speed"] > buildup["speed"]
    assert goal["similarity_boost"] == buildup["similarity_boost"]


@pytest.mark.asyncio
async def test_the_curve_switched_off_sends_a_body_with_no_settings_in_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Absent, not null and not defaults written out: the voice plays exactly
    # as it did before any of this existed.
    monkeypatch.setenv("VOICE_CURVE", "off")
    sdk = FakeSdk()
    sdk.install(monkeypatch)
    spk = ElevenLabsSpeaker(api_key="sk-test", sink=NullSink, curve=VoiceConfig())

    utterance = await spk.say(beat("Mbappé! Buried!", excitement=1.0), asyncio.Event())
    await spk.aclose()

    assert "voice_settings" not in sdk.calls[0]
    assert utterance.voice_settings is None


@pytest.mark.asyncio
async def test_the_settings_a_line_was_said_with_land_on_the_utterance() -> None:
    """``excitement`` alone cannot explain a line that came out flat.

    The curve between the two is the thing being tuned, so the trace has to
    carry the numbers that were sent rather than the input they came from.
    """
    curve = VoiceConfig()
    spk = speaker(FakeStream([b"a" * 32] * 2), NullSink(), curve=curve)

    utterance = await spk.say(beat("Mbappé! Buried!", excitement=1.0), asyncio.Event())

    assert utterance.voice_settings == curve.settings_for("caller", 1.0)
    assert utterance.voice_settings is not None
    assert utterance.voice_settings["stability"] == curve.caller.stability_high


@pytest.mark.asyncio
async def test_a_cut_line_still_records_what_it_was_being_said_with() -> None:
    cancel = asyncio.Event()
    sink = GatedSink(cancel, after=1)
    spk = speaker(FakeStream([b"a" * 32] * 6), sink, curve=VoiceConfig())

    utterance = await spk.say(beat("Mbappé! Buri", excitement=1.0), cancel)

    assert not utterance.completed
    assert utterance.voice_settings is not None


@pytest.mark.asyncio
async def test_the_analyst_is_not_sent_the_callers_settings() -> None:
    curve = VoiceConfig()
    spk = speaker(FakeStream([b"a"] * 2), NullSink(), curve=curve)

    caller_line = await spk.say(beat("Mbappé! Buried!", excitement=1.0), asyncio.Event())
    analyst_line = await spk.say(
        beat("That is the run Scaloni wanted.", voice=Voice.ANALYST, excitement=1.0),
        asyncio.Event(),
    )

    assert caller_line.voice_settings != analyst_line.voice_settings
    assert analyst_line.voice_settings == curve.settings_for("analyst", 1.0)


# -- shaping, at the seam where it is switched on -----------------------


@pytest.mark.asyncio
async def test_a_line_is_shaped_only_when_shaping_is_switched_on() -> None:
    """It edits a line the gate has already approved, so it is opt-in."""
    plain = FakeStream([b"a"])
    shaped = FakeStream([b"a"])
    line = beat("Mbappé, buried.", excitement=1.0)

    await speaker(plain, NullSink()).say(line, asyncio.Event())
    utterance = await speaker(shaped, NullSink(), shaping=True).say(line, asyncio.Event())

    assert plain.calls[0][0] == "Mbappé, buried."
    assert shaped.calls[0][0] == "Mbappé! Buried!"
    # And what is reported as spoken is what was actually said, not the beat.
    assert utterance.spoken == "Mbappé! Buried!"


@pytest.mark.asyncio
async def test_the_analysts_line_is_never_shaped() -> None:
    # Exclamation marks belong to the caller. An analyst shouting is the two
    # voices becoming one voice.
    stream = FakeStream([b"a"])
    spk = speaker(stream, NullSink(), shaping=True)

    await spk.say(
        beat("Scaloni, vindicated.", voice=Voice.ANALYST, excitement=1.0), asyncio.Event()
    )

    assert stream.calls[0][0] == "Scaloni, vindicated."
