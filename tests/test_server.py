"""The two streams that leave the process, and the snapshot beside them."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from commentary.bus import Bus, Topic
from commentary.capture.buffer import DelayBuffer, Frame
from commentary.llm.base import Usage
from commentary.schemas import MatchState
from commentary.server import create_app
from commentary.server.app import mjpeg


class FakeRuntime:
    """Just enough of a runtime for the web layer to have something to show."""

    def __init__(self) -> None:
        self._bus = Bus()
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
