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
    SIGHTING = "sighting"
    CALLER = "caller"
    #: The caller's line and the phraser's rewrite of it, side by side. Not
    #: rendered anywhere: it is on the bus so that the trace carries both,
    #: which is the only way to tell afterwards whether the phrasing stage
    #: helped or whether the caller had already said it well.
    PHRASED = "phrased"
    ANALYST = "analyst"
    GATE = "gate"
    CORRECTION = "correction"
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


@dataclass
class Bus:
    """Fan-out with back-pressure that never reaches the publisher."""

    depth: int = 256
    _subscribers: list[asyncio.Queue[Message]] = field(default_factory=list)
    dropped: int = 0

    def publish(
        self, topic: Topic, ts: float, value: BaseModel | None = None, **extra: Any
    ) -> Message:
        payload = dict(extra) if value is None else {**value.model_dump(mode="json"), **extra}
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
