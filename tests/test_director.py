"""The director's job is mostly about what it refuses to say."""

from __future__ import annotations

import asyncio
import contextlib
import time

import pytest

from commentary.bus import Bus, Topic
from commentary.config import DirectorConfig
from commentary.director import Director, next_beat_id
from commentary.schemas import Beat, Event, Voice
from commentary.voice import LogSpeaker


def beat(
    text: str,
    *,
    voice: Voice = Voice.CALLER,
    event: Event = Event.NONE,
    age: float = 0.0,
    preemptable: bool = True,
) -> Beat:
    now = time.monotonic()
    return Beat(
        id=next_beat_id("t"),
        voice=voice,
        text=text,
        video_ts=0.0,
        created_ts=now - age,
        event=event,
        preemptable=preemptable,
    )


async def run_briefly(director: Director, seconds: float) -> None:
    task = asyncio.create_task(director.run())
    await asyncio.sleep(seconds)
    director.stop()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_speaks_what_it_is_given() -> None:
    speaker = LogSpeaker(words_per_second=200)
    director = Director(speaker=speaker)
    director.submit(beat("Arsenal break down the left"))
    await run_briefly(director, 0.3)
    assert speaker.said[0].completed
    assert director.stats.spoken == 1


@pytest.mark.asyncio
async def test_a_goal_cuts_the_analyst_off_mid_sentence() -> None:
    speaker = LogSpeaker(words_per_second=40)
    bus = Bus()
    director = Director(speaker=speaker, bus=bus)

    long_aside = " ".join(["pressing"] * 40)
    director.submit(beat(long_aside, voice=Voice.ANALYST))
    task = asyncio.create_task(director.run())
    await asyncio.sleep(0.15)

    director.submit(beat("It is in!", event=Event.GOAL, preemptable=False))
    await asyncio.sleep(0.4)
    director.stop()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    aside, goal = speaker.said[0], speaker.said[1]
    assert not aside.completed, "the analyst should have been cut off"
    assert len(aside.spoken.split()) < 40
    assert goal.beat.event is Event.GOAL
    assert director.stats.preempted == 1


@pytest.mark.asyncio
async def test_stale_beats_are_dropped_rather_than_said_late() -> None:
    speaker = LogSpeaker(words_per_second=200)
    director = Director(speaker=speaker, cfg=DirectorConfig(max_beat_age_s=0.5))
    director.submit(beat("that was four seconds ago", age=4.0))
    director.submit(beat("this is now"))
    await run_briefly(director, 0.3)

    assert director.stats.dropped_stale == 1
    assert [u.beat.text for u in speaker.said] == ["this is now"]


@pytest.mark.asyncio
async def test_the_queue_sheds_the_oldest_when_it_overflows() -> None:
    director = Director(cfg=DirectorConfig(queue_depth=2))
    director.submit(beat("one"))
    director.submit(beat("two"))
    director.submit(beat("three"))
    assert director.pending == 2
    assert director.stats.dropped_full == 1


@pytest.mark.asyncio
async def test_an_urgent_beat_jumps_the_queue() -> None:
    director = Director(cfg=DirectorConfig(queue_depth=4))
    director.submit(beat("build up on the right"))
    director.submit(beat("still probing"))
    director.submit(beat("GOAL", event=Event.GOAL))
    assert director._queue[0].text == "GOAL"


@pytest.mark.asyncio
async def test_the_bus_carries_what_happened() -> None:
    bus = Bus()
    seen: list[Topic] = []

    async def watch() -> None:
        async for message in bus.subscribe():
            seen.append(message.topic)

    watcher = asyncio.create_task(watch())
    await asyncio.sleep(0)

    director = Director(speaker=LogSpeaker(words_per_second=200), bus=bus)
    director.submit(beat("a line"))
    await run_briefly(director, 0.3)
    watcher.cancel()

    assert Topic.BEAT in seen
    assert Topic.SPOKEN in seen


@pytest.mark.asyncio
async def test_a_spoken_row_says_how_long_the_line_took_and_how_long_it_was_silent() -> None:
    """Both numbers, or neither is readable.

    ``seconds`` is what the channel was held for and what the rate limiter
    charges against; on its own it cannot say whether a long line was long
    speech or a slow first byte, and those are different bugs. A trace with
    only one of the two is how three seconds a line of player overhead went
    unnoticed for a whole run.
    """
    bus = Bus()
    spoken: list[dict[str, object]] = []

    async def watch() -> None:
        async for message in bus.subscribe():
            if message.topic is Topic.SPOKEN:
                spoken.append(message.payload)

    watcher = asyncio.create_task(watch())
    await asyncio.sleep(0)

    director = Director(speaker=LogSpeaker(words_per_second=200), bus=bus)
    director.submit(beat("a line"))
    await run_briefly(director, 0.3)
    watcher.cancel()

    assert len(spoken) == 1
    assert spoken[0]["seconds"] > 0.0
    # The printed voice has no synthesis to wait through, and says so rather
    # than leaving the field out.
    assert spoken[0]["first_audio_s"] == 0.0
