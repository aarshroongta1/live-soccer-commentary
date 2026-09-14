"""The free macOS speaker: same cancel contract as the real one, no network.

A fake executable stands in for ``say`` so these tests neither depend on
macOS nor make sound. It is a tiny shell script that sleeps for roughly as
long as the real ``say`` would take to speak the given words, so the same
race between process exit and cancellation is exercised for real.
"""

from __future__ import annotations

import asyncio
import shutil
import stat
import time
from pathlib import Path

import pytest

from commentary.schemas import Beat, Voice
from commentary.voice.elevenlabs import VoiceUnavailable
from commentary.voice.say import ANALYST_VOICE, CALLER_VOICE, SaySpeaker


def beat(text: str, *, voice: Voice = Voice.CALLER) -> Beat:
    now = time.monotonic()
    return Beat(id="t1", voice=voice, text=text, video_ts=0.0, created_ts=now)


def fake_say(tmp_path: Path, *, seconds_per_word: float = 0.05) -> str:
    """A script that logs its args and sleeps roughly as long as ``say`` would.

    ``say -v <voice> -r <rate> <text>`` gets its word count from ``$4``, the
    text argument, quoted as a single word by ``asyncio.create_subprocess_exec``.
    """
    script = tmp_path / "fake-say"
    script.write_text(
        "#!/bin/sh\n"
        f'echo "$@" >> "{tmp_path}/calls.log"\n'
        "text=$5\n"
        "set -- $text\n"
        f'python3 -c "import time,sys; time.sleep(len(sys.argv[1:]) * {seconds_per_word})" "$@"\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


@pytest.mark.asyncio
async def test_a_finished_line_reports_the_whole_text_and_the_right_voice(tmp_path: Path) -> None:
    spk = SaySpeaker(executable=fake_say(tmp_path))

    utterance = await spk.say(beat("down the left and it is worked inside"), asyncio.Event())

    assert utterance.completed
    assert utterance.spoken == "down the left and it is worked inside"
    log = (tmp_path / "calls.log").read_text()
    assert f"-v {CALLER_VOICE}" in log


@pytest.mark.asyncio
async def test_the_analyst_gets_the_analyst_voice(tmp_path: Path) -> None:
    spk = SaySpeaker(executable=fake_say(tmp_path))

    await spk.say(beat("they are pressing high", voice=Voice.ANALYST), asyncio.Event())

    log = (tmp_path / "calls.log").read_text()
    assert f"-v {ANALYST_VOICE}" in log


@pytest.mark.asyncio
async def test_a_cancel_stops_the_process_and_leaves_a_partial_line(tmp_path: Path) -> None:
    cancel = asyncio.Event()
    # Ten words at a slow per-word rate so there is time to cancel mid-line.
    spk = SaySpeaker(executable=fake_say(tmp_path, seconds_per_word=0.2), words_per_second=5.0)
    line = "one two three four five six seven eight nine ten"

    async def cut() -> None:
        await asyncio.sleep(0.25)
        cancel.set()

    asyncio.create_task(cut())
    started = time.monotonic()
    utterance = await spk.say(beat(line), cancel)
    elapsed = time.monotonic() - started

    assert not utterance.completed
    assert elapsed < 1.5, "the process should have been killed, not waited out"
    words = utterance.spoken.split()
    assert 0 < len(words) < 10
    assert line.startswith(utterance.spoken)


@pytest.mark.asyncio
async def test_the_speaker_keeps_a_record_of_what_it_said(tmp_path: Path) -> None:
    spk = SaySpeaker(executable=fake_say(tmp_path))

    utterance = await spk.say(beat("a short line"), asyncio.Event())

    assert spk.said == [utterance]


def test_a_missing_executable_refuses_to_build_a_speaker_that_could_never_speak() -> None:
    with pytest.raises(VoiceUnavailable, match="macOS only"):
        SaySpeaker(executable="/no/such/say-binary-anywhere")


@pytest.mark.skipif(shutil.which("say") is None, reason="needs the real macOS `say`")
@pytest.mark.asyncio
async def test_the_real_say_binary_can_be_found_and_speaks_a_single_short_word() -> None:
    """The one path a fake script cannot cover: that PATH lookup finds the real thing.

    Rate is cranked up and the word kept to one syllable so this is quick and
    unobtrusive rather than a real demo of the voice.
    """
    spk = SaySpeaker(rate=500)

    utterance = await spk.say(beat("hi"), asyncio.Event())

    assert utterance.completed
    assert utterance.spoken == "hi"
