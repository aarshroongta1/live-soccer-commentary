"""One broadcast channel from the runtime to everything watching it.

The web page, the trace writer, and the voice queue all want to see the same
stream of things that happened, and none of them may slow the match down. So
every subscriber gets its own bounded queue and a slow reader loses messages
rather than stalling the caller. A dropped frame in the UI is a cosmetic
problem; a caller blocked on a socket write is a missed goal.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class Topic(StrEnum):
    """What kind of thing happened."""

    BEAT = "beat"
    SPOKEN = "spoken"
    PREEMPTED = "preempted"
    STATE = "state"
    BOARD = "board"
    CALLER = "caller"
    ANALYST = "analyst"
    GATE = "gate"
    TRIGGER = "trigger"
    COST = "cost"
    STATUS = "status"
    ERROR = "error"


@dataclass(frozen=True)
class Message:
    """One published thing, ready to become an SSE frame or a trace line."""

    topic: Topic
    ts: float
    payload: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps({"topic": self.topic.value, "ts": self.ts, **self.payload})

    def sse(self) -> str:
        return f"event: {self.topic.value}\ndata: {self.to_json()}\n\n"


def as_payload(value: Any) -> dict[str, Any]:
    """Whatever an agent produced, flattened into something JSON-safe."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return {"value": value}


@dataclass
class Bus:
    """Fan-out with back-pressure that never reaches the publisher."""

    depth: int = 256
    _subscribers: list[asyncio.Queue[Message]] = field(default_factory=list)
    dropped: int = 0

    def publish(self, topic: Topic, ts: float, value: Any = None, **extra: Any) -> Message:
        payload = {**as_payload(value), **extra} if value is not None else dict(extra)
        message = Message(topic=topic, ts=ts, payload=payload)
        for queue in self._subscribers:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                self.dropped += 1
        return message

    async def subscribe(self) -> AsyncIterator[Message]:
        queue: asyncio.Queue[Message] = asyncio.Queue(maxsize=self.depth)
        self._subscribers.append(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers.remove(queue)

    @property
    def subscribers(self) -> int:
        return len(self._subscribers)
