"""The speaker's job is mostly about stopping.

Nothing here touches the network. The chunk stream and the audio sink are both
constructor arguments, so a fake iterator and a :class:`NullSink` exercise the
real pump loop — the cancel checks, the abandonment, the estimate — without a
key, a socket, or a sound card.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
import types
from collections.abc import AsyncIterator, Sequence

import pytest

from commentary.schemas import Beat, Voice
from commentary.voice import ElevenLabsSpeaker
from commentary.voice.playback import FFplaySink, NullSink


def beat(text: str, *, voice: Voice = Voice.CALLER) -> Beat:
    now = time.monotonic()
    return Beat(id="t1", voice=voice, text=text, video_ts=0.0, created_ts=now)


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
    assert utterance.cut_off
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


@pytest.mark.asyncio
async def test_no_key_is_a_silent_speaker_rather_than_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    spk = ElevenLabsSpeaker()  # constructing must not raise

    assert not spk.available
    utterance = await spk.say(beat("nobody is listening"), asyncio.Event())

    assert not utterance.completed
    assert utterance.spoken == ""
    await spk.aclose()


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
        "assert ElevenLabsSpeaker(api_key='') is not None; print('ok')"
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
