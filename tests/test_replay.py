"""Watching a finished run again, from its trace and its clip.

A replay is only worth anything if it is faithful: the same rows, in the
order the match produced them, arriving at the cursor they arrived at the
first time. These tests are about that and nothing else — no model is called
here, which is the point of the feature.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from commentary.bus import Message, Topic
from commentary.capture.buffer import Frame
from commentary.config import CaptureConfig, Settings
from commentary.replay import Replay, cues, from_files
from commentary.server import create_app
from commentary.server.app import mjpeg
from commentary.trace import RunTrace
from commentary.voice import LogSpeaker

FPS = 10
DELAY_S = 2.0


def settings() -> Settings:
    return Settings(
        capture=CaptureConfig(width=8, height=8, fps=FPS, delay_s=DELAY_S, history_s=4.0)
    )


class Clip:
    """A clip that decodes instantly, stamped the way ``FileCapture`` stamps one.

    ``FileCapture`` seeks with ``-ss`` and then counts frames from zero, so a
    run that began twenty seconds into a clip still has a trace that counts
    from zero. Nothing here needs ffmpeg to say the same thing.
    """

    def __init__(self, seconds: float) -> None:
        self.count = int(seconds * FPS)
        self.entered = False

    async def __aenter__(self) -> Clip:
        self.entered = True
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.entered = False

    async def frames(self) -> AsyncIterator[Frame]:
        for index in range(self.count):
            yield Frame(ts=index / FPS, image=np.full((8, 8, 3), index % 255, dtype=np.uint8))
            await asyncio.sleep(0)


def row(topic: str, ts: float, **payload: Any) -> dict[str, Any]:
    return {"topic": topic, "ts": ts, **payload}


def state_row(ts: float, home_score: int, away_score: int, clock: str) -> dict[str, Any]:
    return row(
        "state",
        ts,
        home="Argentina",
        away="France",
        home_score=home_score,
        away_score=away_score,
        clock=clock,
        period=1,
        in_replay=False,
    )


def beat_row(
    ts: float,
    text: str,
    *,
    beat_id: str = "b1",
    voice: str = "caller",
    event: str = "none",
    preemptable: bool = True,
    created_ts: float = 941887.047232583,
    live_ts: float | None = None,
) -> dict[str, Any]:
    """A ``beat`` row shaped the way the trace writer shapes one.

    ``created_ts`` defaults to a real reading out of the Di María trace: a
    ``time.monotonic()`` value from a process that is long gone, which is
    exactly the hazard a replay with a voice has to survive.
    """
    return row(
        "beat",
        ts,
        id=beat_id,
        voice=voice,
        text=text,
        video_ts=ts,
        created_ts=created_ts,
        live_ts=DELAY_S + ts if live_ts is None else live_ts,
        event=event,
        urgency=0.0,
        triggers=[],
        preemptable=preemptable,
    )


def replay_of(
    rows: list[dict[str, Any]],
    *,
    seconds: float = 10.0,
    start_s: float = 0.0,
    speaker: Any = None,
) -> Replay:
    return Replay(
        rows=rows,
        source=Clip(seconds),
        settings=settings(),
        start_s=start_s,
        speaker=speaker,
    )


async def collect(replay: Replay) -> list[Message]:
    """Everything the replay publishes, in the order it publishes it."""
    seen: list[Message] = []

    async def watch() -> None:
        async for message in replay.bus.subscribe():
            seen.append(message)

    task = asyncio.create_task(watch())
    await asyncio.sleep(0)
    await replay.run()
    await asyncio.sleep(0.01)
    task.cancel()
    return seen


def test_a_trace_is_sorted_into_timestamp_order_whatever_order_it_was_written_in() -> None:
    # A real trace opens with a board read stamped two seconds in, above a
    # trigger stamped at a quarter of a second: the board reader runs a
    # buffer's length ahead of the tick loop and its row is written first.
    ordered = cues([row("board", 1.98), row("trigger", 0.25), row("caller", 0.25)])

    assert [cue.ts for cue in ordered] == [0.25, 0.25, 1.98]
    # Stable: two rows on the same timestamp keep the order the run made them.
    assert [cue.topic for cue in ordered] == [Topic.TRIGGER, Topic.CALLER, Topic.BOARD]


def test_a_topic_the_bus_no_longer_knows_is_dropped_rather_than_crashing_the_replay() -> None:
    # `tracks` and `gallery` rows are in every trace written before the
    # player tracker was removed, and those traces are still on disk.
    assert [cue.topic for cue in cues([row("tracks", 1.0), row("spoken", 2.0)])] == [Topic.SPOKEN]


def test_the_payload_survives_the_round_trip_intact() -> None:
    only = cues([row("spoken", 3.0, voice="caller", text="Di María strikes.")])[0]

    assert only.payload == {"voice": "caller", "text": "Di María strikes."}
    assert only.message().to_json().startswith('{"topic": "spoken", "ts": 3.0')


@pytest.mark.asyncio
async def test_rows_are_published_in_timestamp_order_as_the_cursor_reaches_them() -> None:
    replay = replay_of(
        [
            row("spoken", 4.0, text="third"),
            row("spoken", 1.0, text="first"),
            row("spoken", 2.5, text="second"),
        ]
    )

    seen = await collect(replay)

    assert [message.payload["text"] for message in seen] == ["first", "second", "third"]
    assert [message.ts for message in seen] == [1.0, 2.5, 4.0]


@pytest.mark.asyncio
async def test_a_row_is_not_published_before_the_cursor_reaches_it() -> None:
    replay = replay_of([row("spoken", 5.0, text="the goal")])

    # The cursor trails the live edge by the buffer depth, so five seconds of
    # clip is not enough to reach a row stamped at five seconds.
    for index in range(5 * FPS):
        replay.buffer.append(Frame(ts=index / FPS, image=np.zeros((8, 8, 3), dtype=np.uint8)))
        replay.publish_due()
    assert replay.played == 0

    for index in range(5 * FPS, 8 * FPS):
        replay.buffer.append(Frame(ts=index / FPS, image=np.zeros((8, 8, 3), dtype=np.uint8)))
        replay.publish_due()
    assert replay.played == 1


@pytest.mark.asyncio
async def test_start_shifts_the_schedule_onto_the_clip_the_run_was_watching() -> None:
    # The screen run played this clip from twenty seconds in, so its trace
    # counts from there: a row stamped at 5.0 describes clip second 25.
    shifted = replay_of([row("spoken", 5.0, text="the goal")], start_s=20.0)
    plain = replay_of([row("spoken", 5.0, text="the goal")])

    for index in range(8 * FPS):
        frame = Frame(ts=index / FPS, image=np.zeros((8, 8, 3), dtype=np.uint8))
        shifted.buffer.append(frame)
        plain.buffer.append(frame)
        if shifted.played == 0:
            shifted.publish_due()
            landed_at = shifted.clip_ts
        if plain.played == 0:
            plain.publish_due()
            plain_landed_at = plain.clip_ts

    assert landed_at == pytest.approx(25.0, abs=0.2)
    assert plain_landed_at == pytest.approx(5.0, abs=0.2)
    # Both fired at the same point in the *run*; only the clip position moved.
    assert shifted.played == plain.played == 1


@pytest.mark.asyncio
async def test_the_last_state_row_seen_is_what_the_snapshot_returns() -> None:
    replay = replay_of(
        [
            state_row(1.0, 1, 0, "34:46"),
            state_row(3.0, 2, 0, "36:02"),
            state_row(30.0, 2, 1, "41:18"),
            row("cost", 3.5, total_usd=0.4137),
        ],
        seconds=8.0,
    )
    client = TestClient(create_app(replay))

    # Before a row has played, the teams are known and nothing else is.
    assert client.get("/api/state").json()["state"]["home"] == "Argentina"
    assert client.get("/api/state").json()["state"]["home_score"] == 0

    await collect(replay)
    body = client.get("/api/state").json()

    # The clip ended before the row at thirty seconds, so the run's own last
    # word is not the answer: what the cursor reached is.
    assert body["state"]["home_score"] == 2
    assert body["state"]["away_score"] == 0
    assert body["state"]["clock"] == "36:02"
    assert body["usage"]["cost_usd"] == 0.4137
    assert body["status"]["replayed"] == 3
    assert replay.remaining == 1


@pytest.mark.asyncio
async def test_a_replay_serves_the_delayed_picture_from_the_cursor_like_a_live_run() -> None:
    replay = replay_of([row("spoken", 1.0, text="a line")], seconds=6.0)
    await collect(replay)

    part = await anext(mjpeg(replay.buffer, fps=60))
    assert part.startswith(b"--frame")
    assert b"image/jpeg" in part

    cursor, live = replay.buffer.cursor_ts, replay.buffer.live_ts
    assert cursor is not None and live is not None
    assert live - cursor == pytest.approx(DELAY_S)


@pytest.mark.asyncio
async def test_with_loop_the_cues_are_published_again_once_the_first_pass_ends() -> None:
    replay = replay_of([row("spoken", 1.0, text="a line")], seconds=6.0)
    replay.loop = True
    seen: list[Message] = []

    async def watch() -> None:
        async for message in replay.bus.subscribe():
            seen.append(message)

    watcher = asyncio.create_task(watch())
    await asyncio.sleep(0)
    runner = asyncio.create_task(replay.run())

    for _ in range(20_000):
        if len(seen) >= 2:
            break
        await asyncio.sleep(0)

    replay.stop()
    await asyncio.wait_for(runner, timeout=1.0)
    watcher.cancel()

    # Republished at the same point in the trace the second time round, not
    # somewhere it never appeared in the first pass.
    assert len(seen) >= 2
    assert [message.payload["text"] for message in seen[:2]] == ["a line", "a line"]
    assert seen[0].ts == seen[1].ts == 1.0


@pytest.mark.asyncio
async def test_looping_seeks_the_clip_back_so_the_buffers_live_ts_restarts() -> None:
    replay = replay_of([row("spoken", 1.0, text="a line")], seconds=1.0)
    replay.loop = True
    runner = asyncio.create_task(replay.run())

    peak = 0.0
    dropped = False
    for _ in range(20_000):
        await asyncio.sleep(0)
        live = replay.buffer.live_ts
        if live is None:
            continue
        if live < peak - 0.05:
            # The live edge fell well below where it had climbed to: the
            # source was reopened at the start rather than kept playing on.
            dropped = True
            break
        peak = max(peak, live)

    replay.stop()
    await asyncio.wait_for(runner, timeout=1.0)

    assert dropped
    assert peak > 0.5  # the first pass actually played most of the one-second clip


@pytest.mark.asyncio
async def test_stop_ends_a_looping_replay() -> None:
    replay = replay_of([row("spoken", 1.0, text="a line")], seconds=1.0)
    replay.loop = True
    runner = asyncio.create_task(replay.run())

    # Enough scheduler turns for the one-second clip to loop several times
    # over before it is asked to stop.
    for _ in range(5000):
        await asyncio.sleep(0)

    replay.stop()
    await asyncio.wait_for(runner, timeout=1.0)

    assert runner.done() and not runner.cancelled()


def test_from_files_reads_a_trace_off_disk_and_points_ffmpeg_at_the_clip(tmp_path: Any) -> None:
    path = tmp_path / "run.jsonl"
    with RunTrace(path=path) as trace:
        trace.event(Topic.SPOKEN, 2.0, text="Glorious goal. Di María.")

    replay = from_files(path, "clips/argfra-dimaria.mp4", start_s=20.0, settings=settings())

    assert [cue.ts for cue in replay.cues] == [2.0]
    assert replay.start_s == 20.0
    assert getattr(replay.source, "start_s", None) == 20.0
    assert "run.jsonl" in replay.status()["message"]


def test_a_row_that_carries_a_live_edge_is_scheduled_at_the_cursor_it_went_out_at() -> None:
    """A beat is stamped with the moment it is *about*, not the moment it aired.

    The caller is handed the cursor, the model thinks for a few seconds, and
    only then does the line exist to be published. ``video_ts`` is the first
    of those instants and ``live_ts`` the second, so the cursor time the run
    actually published at is ``live_ts - delay_s``. On the Di María screen
    run that gap is a median 3.4 s.
    """
    ordered = cues(
        [
            row("beat", 0.25, text="Argentina building from the back", live_ts=13.9),
            row("gate", 0.25, passed=True),
        ],
        delay_s=8.0,
    )
    scheduled = {cue.topic: cue.at for cue in ordered}

    assert scheduled[Topic.BEAT] == pytest.approx(5.9)
    # The row still *says* 0.25: that is the moment the line is about, and the
    # transcript, the eval and every panel align on it.
    assert [cue.ts for cue in ordered] == [0.25, 0.25]
    # A row with no live edge is a panel's account of a moment, not something
    # a viewer hears. It stays where it is.
    assert scheduled[Topic.GATE] == pytest.approx(0.25)


def test_rows_without_a_live_edge_keep_their_order_among_themselves() -> None:
    # The stable sort still holds after the rescheduling: these three share a
    # schedule and come back in the order the run produced them.
    ordered = cues([row("trigger", 2.0), row("caller", 2.0), row("sighting", 2.0)], delay_s=8.0)

    assert [cue.topic for cue in ordered] == [Topic.TRIGGER, Topic.CALLER, Topic.SIGHTING]


@pytest.mark.asyncio
async def test_a_line_is_replayed_when_the_run_said_it_not_when_it_was_written_about() -> None:
    """The whole point of the rescheduling, end to end.

    Replaying a beat at its own timestamp would put it on the page three
    seconds before the play it describes, because the picture is now held
    ``present_offset_s`` behind the cursor to meet the model's round trip.
    """
    replay = replay_of(
        [row("beat", 1.0, text="Di María strikes.", live_ts=DELAY_S + 4.0)],
        seconds=10.0,
    )

    # The beat is about video second 1.0 but went out at cursor 4.0, so the
    # cursor sails past 1.0 with nothing published.
    for index in range(5 * FPS):
        replay.buffer.append(Frame(ts=index / FPS, image=np.zeros((8, 8, 3), dtype=np.uint8)))
        replay.publish_due()
    assert replay.buffer.cursor_ts is not None and replay.buffer.cursor_ts > 1.0
    assert replay.played == 0, "a line replayed at its own timestamp lands before its own play"

    for index in range(5 * FPS, 8 * FPS):
        replay.buffer.append(Frame(ts=index / FPS, image=np.zeros((8, 8, 3), dtype=np.uint8)))
        replay.publish_due()

    assert replay.played == 1


@pytest.mark.asyncio
async def test_the_republished_row_still_carries_the_moment_it_describes() -> None:
    replay = replay_of(
        [row("spoken", 1.0, voice="caller", text="Di María strikes.", live_ts=DELAY_S + 3.4)],
        seconds=10.0,
    )

    seen = await collect(replay)

    assert [message.ts for message in seen] == [1.0]
    assert seen[0].payload["text"] == "Di María strikes."


def test_the_replay_tells_the_page_how_far_behind_the_cursor_the_picture_is() -> None:
    replay = replay_of([row("spoken", 1.0, text="a line")])
    client = TestClient(create_app(replay))

    offset = client.get("/api/state").json()["status"]["present_offset_s"]
    assert offset == pytest.approx(settings().capture.present_offset_s)


# -- replaying with a voice ---------------------------------------------
#
# A trace is a list of lines a model was already paid for. Handing them to a
# real director and a real speaker is the only way to hear a change to either
# without buying the thinking a second time, and the point of these tests is
# that what comes out is the live article and not a recital of the file.


def fast_speaker() -> LogSpeaker:
    """Fast enough that a whole clip's lines fit inside the test."""
    return LogSpeaker(words_per_second=2000.0)


@pytest.mark.asyncio
async def test_with_no_voice_the_trace_is_replayed_exactly_as_before() -> None:
    replay = replay_of([beat_row(1.0, "a line"), row("spoken", 1.0, text="a line", seconds=5.1)])

    seen = await collect(replay)

    assert replay.director is None
    # Both rows go back out untouched: the trace is the whole account.
    assert [message.topic for message in seen] == [Topic.BEAT, Topic.SPOKEN]
    assert seen[1].payload["seconds"] == 5.1


@pytest.mark.asyncio
async def test_a_voice_says_the_traces_lines_again_through_a_real_director() -> None:
    speaker = fast_speaker()
    replay = replay_of([beat_row(1.0, "Di María strikes.")], speaker=speaker)

    seen = await collect(replay)

    assert [utterance.beat.text for utterance in speaker.said] == ["Di María strikes."]
    spoken = [message for message in seen if message.topic is Topic.SPOKEN]
    assert len(spoken) == 1
    assert spoken[0].payload["spoken"] == "Di María strikes."
    # The director's own timing, measured here and now, not the file's.
    assert spoken[0].payload["first_audio_s"] == 0.0
    # And stamped with the moment the line is about, exactly as a live run
    # stamps it, so the transcript still lines up with the picture.
    assert spoken[0].ts == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_the_traces_own_account_of_the_speaking_is_dropped_when_there_is_a_voice() -> None:
    """Two accounts of one line on the page is worse than either alone.

    The trace says the line took 5.1 s through ffplay in September. The
    director says what it took just now. Only one of them is about the audio
    in the room, so only one of them is published.
    """
    replay = replay_of(
        [
            beat_row(1.0, "a line"),
            row("spoken", 1.0, text="a line", spoken="a line", seconds=5.1, live_ts=DELAY_S + 1.0),
            row("preempted", 2.0, text="a cut line", reason="cut", live_ts=DELAY_S + 2.0),
            row("gate", 1.0, passed=True),
        ],
        speaker=fast_speaker(),
    )

    seen = await collect(replay)

    seconds = [message.payload.get("seconds") for message in seen if message.topic is Topic.SPOKEN]
    assert 5.1 not in seconds
    assert len(seconds) == 1 and seconds[0] is not None and seconds[0] < 5.1
    # The file's cut line is gone with it; nothing this playback did was cut.
    assert not [message for message in seen if message.topic is Topic.PREEMPTED]
    # Everything that is not the speaking is still the trace's to report.
    assert [message.topic for message in seen if message.topic is Topic.GATE]


@pytest.mark.asyncio
async def test_a_beat_from_an_old_trace_is_not_thrown_away_as_hours_stale() -> None:
    """The one thing that has to be rewritten on the way back in.

    ``created_ts`` in the file is a ``time.monotonic()`` reading from a
    process that exited weeks ago. The director ages beats against
    ``time.monotonic()`` now, so left as written every line in every trace is
    past its limit and is dropped before a voice ever sees it.
    """
    speaker = fast_speaker()
    replay = replay_of([beat_row(1.0, "a line from an old run")], speaker=speaker)

    await collect(replay)

    assert [utterance.beat.text for utterance in speaker.said] == ["a line from an old run"]


@pytest.mark.asyncio
async def test_a_goal_still_goes_to_the_front_of_the_queue_on_the_way_back_out() -> None:
    """The queueing is the director's, not the file's order.

    A replayed goal has to still be a goal — the event and ``preemptable``
    survive the round trip — or the one behaviour this project is built
    around is the one thing a replay cannot be used to hear.
    """
    speaker = LogSpeaker(words_per_second=20.0)
    replay = replay_of(
        [
            beat_row(1.0, "an aside about the press", beat_id="a1", voice="analyst"),
            beat_row(
                1.0,
                "Di María! The rebound, and it is in!",
                beat_id="b2",
                event="goal",
                preemptable=False,
            ),
        ],
        speaker=speaker,
    )

    seen = await collect(replay)

    urgent = [
        message.payload["text"]
        for message in seen
        if message.topic is Topic.BEAT and message.payload.get("urgent")
    ]
    assert urgent == ["Di María! The rebound, and it is in!"]
    assert speaker.said[0].beat.text == "Di María! The rebound, and it is in!"


@pytest.mark.asyncio
async def test_the_page_is_told_whether_anything_is_actually_coming_out_of_the_speakers() -> None:
    silent = replay_of([row("spoken", 1.0, text="a line")])
    speaking = replay_of([beat_row(1.0, "a line")], speaker=fast_speaker())

    assert TestClient(create_app(silent)).get("/api/state").json()["status"]["speaking"] is False
    assert TestClient(create_app(speaking)).get("/api/state").json()["status"]["speaking"] is True


def test_from_files_carries_the_voice_through_to_the_replay(tmp_path: Any) -> None:
    path = tmp_path / "run.jsonl"
    with RunTrace(path=path) as trace:
        trace.event(Topic.SPOKEN, 2.0, text="Glorious goal. Di María.")

    speaker = fast_speaker()
    replay = from_files(path, "clips/argfra-dimaria.mp4", settings=settings(), speaker=speaker)

    assert replay.director is not None
    assert replay.director.speaker is speaker
