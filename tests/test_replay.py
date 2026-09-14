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


def replay_of(rows: list[dict[str, Any]], *, seconds: float = 10.0, start_s: float = 0.0) -> Replay:
    return Replay(rows=rows, source=Clip(seconds), settings=settings(), start_s=start_s)


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


def test_from_files_reads_a_trace_off_disk_and_points_ffmpeg_at_the_clip(tmp_path: Any) -> None:
    path = tmp_path / "run.jsonl"
    with RunTrace(path=path) as trace:
        trace.event(Topic.SPOKEN, 2.0, text="Glorious goal. Di María.")

    replay = from_files(path, "clips/argfra-dimaria.mp4", start_s=20.0, settings=settings())

    assert [cue.ts for cue in replay.cues] == [2.0]
    assert replay.start_s == 20.0
    assert getattr(replay.source, "start_s", None) == 20.0
    assert "run.jsonl" in replay.status()["message"]
