"""Replay saved visual responses without repaying for vision or exposing later rows."""

from __future__ import annotations

import asyncio
import copy
import math
from collections.abc import Awaitable, Callable
from typing import Any

from commentary.llm.base import Usage
from commentary.recorded_demo import Frame, Observation


class SavedVision:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        rate: float = 1,
        source_rate: float = 1,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if (
            not math.isfinite(rate)
            or rate <= 0
            or not math.isfinite(source_rate)
            or source_rate <= 0
        ):
            raise ValueError("saved vision replay rates must be positive and finite")
        self._rows: dict[tuple[float, ...], dict[str, Any]] = {}
        self._rate = rate
        self._source_rate = source_rate
        self._sleep = sleep
        self.audit: list[dict[str, Any]] = []
        for row in rows:
            if row.get("status", "complete") != "complete":
                raise ValueError("saved vision must contain only completed visual batches")
            stamps = tuple(float(value) for value in row["frames_at_s"])
            if (
                not stamps
                or len(stamps) > 8
                or any(not math.isfinite(t) or t < 0 for t in stamps)
                or any(a >= b for a, b in zip(stamps, stamps[1:], strict=False))
            ):
                raise ValueError("invalid saved frame timeline")
            if stamps in self._rows or float(row["cutoff_s"]) != stamps[-1]:
                raise ValueError("duplicate saved batch or mismatched cutoff")
            latency = float(row["usage"]["latency_s"])
            if not math.isfinite(latency) or latency < 0:
                raise ValueError("invalid saved vision latency")
            previous = stamps[0]
            for raw in row["resolved"]:
                observation = Observation.model_validate(raw)
                if not previous <= observation.at_s <= stamps[-1]:
                    raise ValueError(
                        "saved observations must be chronological and inside their batch"
                    )
                if observation.action and not set(observation.action.support_at_s) <= set(stamps):
                    raise ValueError("saved action support must use supplied frame timestamps")
                previous = observation.at_s
            self._rows[stamps] = copy.deepcopy(row)
        if not self._rows:
            raise ValueError("saved run contains no completed visual batches")

    async def __call__(self, frames: list[Frame], cutoff_s: float) -> list[Observation]:
        stamps = tuple(frame.at_s for frame in frames)
        row = self._rows.get(stamps)
        if row is None or stamps[-1] != cutoff_s:
            raise ValueError("no saved vision for this exact frame batch; no paid fallback")
        delay = float(row["usage"]["latency_s"]) * self._source_rate / self._rate
        await self._sleep(delay)
        current = copy.deepcopy(row)
        current.update(
            source="saved_vision",
            prior_usage=current["usage"],
            usage={**Usage().__dict__, "latency_s": delay},
            latency_basis="replayed_saved_latency",
        )
        self.audit.append(current)
        return [Observation.model_validate(raw) for raw in row["resolved"]]
