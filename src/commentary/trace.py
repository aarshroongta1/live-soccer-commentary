"""Everything that happened in a run, on disk, one JSON object per line.

The eval does not re-run the match; it reads this file. So a trace has to
carry enough to reconstruct every decision after the fact: what the board
read, what the caller filled in, why the gate rejected a line, which trigger
fired, and what each call cost. A run that cannot be graded afterwards was a
run wasted, and there are only so many live matches before the showcase.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from commentary.bus import Message, Topic


@dataclass
class RunTrace:
    """Append-only JSONL for one match, plus a small in-memory tally."""

    path: Path
    run_id: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S"))
    counts: dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    _fh: Any = None

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, message: Message) -> None:
        self.counts[message.topic.value] = self.counts.get(message.topic.value, 0) + 1
        if message.topic is Topic.COST:
            self.cost_usd = float(message.payload.get("total_usd", self.cost_usd))
        self._fh.write(message.to_json() + "\n")
        self._fh.flush()

    def event(self, topic: Topic, ts: float, **payload: Any) -> None:
        self.write(Message(topic=topic, ts=ts, payload=payload))

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> RunTrace:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def summary(self) -> str:
        parts = [f"{k}={v}" for k, v in sorted(self.counts.items())]
        return f"run {self.run_id}: " + " ".join(parts) + f" cost=${self.cost_usd:.2f}"


def read_trace(path: Path) -> list[dict[str, Any]]:
    """Load a trace back. Tolerates a truncated last line from a killed run."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def rows_of(rows: list[dict[str, Any]], topic: str) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("topic") == topic]
