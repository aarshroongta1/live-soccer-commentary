"""The two streams that leave the process, and the snapshot beside them."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from commentary.bus import Bus, Topic
from commentary.capture.buffer import DelayBuffer, Frame
from commentary.llm.base import Usage
from commentary.schemas import MatchState
from commentary.server import create_app
from commentary.server.app import mjpeg, present_frame


class FakeRuntime:
    """Just enough of a runtime for the web layer to have something to show."""

    def __init__(self, present_offset_s: float = 0.0) -> None:
        self._bus = Bus()
        self._present_offset_s = present_offset_s
        self._buffer = DelayBuffer(fps=10, delay_s=1.0, history_s=1.0)
        self._state = MatchState(home="Ashcombe", away="Verity", home_score=1, clock="37:12")
        for i in range(25):
            image = np.full((32, 64, 3), i * 8 % 255, dtype=np.uint8)
            self._buffer.append(Frame(ts=i / 10.0, image=image))

    @property
    def bus(self) -> Bus:
        return self._bus

    @property
    def buffer(self) -> DelayBuffer:
        return self._buffer

    @property
    def state(self) -> MatchState:
        return self._state

    @property
    def usage(self) -> Usage:
        return Usage(input_tokens=1200, output_tokens=90, cost_usd=0.0034)

    @property
    def present_offset_s(self) -> float:
        return self._present_offset_s

    def status(self) -> dict[str, Any]:
        return {"frames": 25, "spoken": 3}


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(FakeRuntime()))


def test_health_is_health(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}


def test_the_state_snapshot_carries_score_status_and_spend(client: TestClient) -> None:
    body = client.get("/api/state").json()
    assert body["state"]["home_score"] == 1
    assert body["state"]["clock"] == "37:12"
    assert body["status"]["spoken"] == 3
    assert body["usage"]["cost_usd"] == pytest.approx(0.0034)


def test_the_page_renders_without_a_node_build(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert "/api/video" in page.text


@pytest.mark.asyncio
async def test_the_video_stream_serves_from_the_cursor_not_the_live_edge() -> None:
    runtime = FakeRuntime()
    stream = mjpeg(runtime.buffer, fps=60)
    first = await anext(stream)

    # A static buffer yields once and then waits: an unchanged frame is not
    # worth re-encoding, and a stalled stream should look frozen rather than
    # flood the socket. Advancing the buffer is what produces the next part.
    runtime.buffer.append(Frame(ts=9.9, image=np.zeros((32, 64, 3), dtype=np.uint8)))
    second = await anext(stream)
    await stream.aclose()

    for part in (first, second):
        assert part.startswith(b"--frame")
        assert b"image/jpeg" in part

    # The cursor trails the live edge by delay_s, so the frame served is the
    # one the commentator is describing — not the newest one available.
    cursor = runtime.buffer.cursor_ts
    live = runtime.buffer.live_ts
    assert cursor is not None and live is not None
    assert live - cursor == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_a_subscriber_sees_events_as_they_are_published() -> None:
    bus = Bus()
    seen: list[str] = []

    async def watch() -> None:
        async for message in bus.subscribe():
            seen.append(message.sse())

    task = asyncio.create_task(watch())
    await asyncio.sleep(0)
    bus.publish(Topic.SPOKEN, 12.5, None, voice="caller", text="Ashcombe break")
    await asyncio.sleep(0.01)
    task.cancel()

    assert seen and seen[0].startswith("event: spoken\ndata: ")
    assert "Ashcombe break" in seen[0]
    assert seen[0].endswith("\n\n")


@pytest.mark.asyncio
async def test_a_slow_subscriber_loses_messages_rather_than_blocking_the_match() -> None:
    bus = Bus(depth=4)
    agen = bus.subscribe()

    # The queue is registered when the generator first runs, so start a pull
    # and let it reach its await. After that the reader simply stops reading,
    # which is the situation being tested: a browser that has wandered off.
    first = asyncio.create_task(anext(agen))  # type: ignore[arg-type]
    await asyncio.sleep(0)
    assert bus.subscribers == 1

    for i in range(50):
        bus.publish(Topic.BEAT, float(i), None, text=str(i))
    await first

    assert bus.dropped > 0, "a stalled reader should shed messages, not apply back-pressure"
    await agen.aclose()


@pytest.mark.asyncio
async def test_the_video_stream_serves_the_frame_the_line_will_be_about() -> None:
    """The picture is held behind the cursor by the caller's round trip.

    A line is written about the cursor and reaches the page a model call
    later, so a stream served at the cursor exactly puts every line after the
    thing it describes. The frame the viewer gets is the one at
    ``cursor - present_offset_s``, which is the moment the line coming out of
    the model right now was written about.
    """
    import cv2

    offset = 0.8
    runtime = FakeRuntime(present_offset_s=offset)
    cursor = runtime.buffer.cursor_ts
    assert cursor is not None

    wanted = runtime.buffer.nearest(cursor - offset)
    assert wanted is not None
    assert wanted.ts == pytest.approx(cursor - offset)
    assert present_frame(runtime.buffer, offset) is wanted
    # Not the cursor's own frame, which is what the stream used to serve.
    assert wanted.ts != runtime.buffer.nearest(cursor).ts  # type: ignore[union-attr]

    part = await anext(mjpeg(runtime.buffer, offset, fps=60))
    body = part.split(b"\r\n\r\n", 1)[1].rstrip(b"\r\n")
    decoded = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)

    # The frames are flat greys, one value apiece, so the picture on the wire
    # names which frame it came from.
    assert float(decoded.mean()) == pytest.approx(float(wanted.image.mean()), abs=2.0)


@pytest.mark.asyncio
async def test_the_offset_clamps_at_the_oldest_frame_rather_than_running_off_the_end() -> None:
    """Early in a run there is no history to fall back through.

    The buffer has held frames for a fraction of a second and the offset asks
    for one from before the run began. The viewer gets the oldest frame there
    is — a picture a little ahead of where it should be, for a moment — rather
    than nothing at all.
    """
    runtime = FakeRuntime(present_offset_s=5.0)
    oldest = runtime.buffer.nearest(-1e9)
    assert oldest is not None

    assert present_frame(runtime.buffer, 5.0) is oldest

    part = await anext(mjpeg(runtime.buffer, 5.0, fps=60))
    assert part.startswith(b"--frame")


def test_the_snapshot_says_how_far_behind_the_cursor_the_picture_is() -> None:
    # The page reads it from here to label the video stage; no runtime has to
    # remember to put it in its own status dict.
    client = TestClient(create_app(FakeRuntime(present_offset_s=3.5)))

    assert client.get("/api/state").json()["status"]["present_offset_s"] == pytest.approx(3.5)


def test_a_run_with_no_clip_on_disk_has_no_clip_route(client: TestClient) -> None:
    """A live run serves the delay buffer's stream and nothing else."""
    assert client.get("/api/clip").status_code == 404


def test_a_replay_serves_its_clip_for_the_page_to_play_natively(tmp_path: Path) -> None:
    """The first person to watch a lap said the picture and the frame rate
    were poor: 960 wide at quality 72, ticking at 12 against a 15 fps buffer.
    On a replay the clip is on disk, so the page plays the file itself and
    keeps it seeked to the cursor; the route answers Range requests so it
    can seek.
    """
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00" * 4096)
    runtime = FakeRuntime()
    runtime.clip_path = clip  # type: ignore[attr-defined]
    api = TestClient(create_app(runtime))
    whole = api.get("/api/clip")
    assert whole.status_code == 200
    assert whole.headers["content-type"].startswith("video/mp4")
    part = api.get("/api/clip", headers={"Range": "bytes=0-99"})
    assert part.status_code == 206
    assert len(part.content) == 100
    page = api.get("/").text
    assert "/api/clip" in page and "<video" in page
