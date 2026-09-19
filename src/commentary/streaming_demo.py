"""A bounded per-batch commentary proof built from the recorded-demo contracts."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, field_validator, model_validator

from commentary.llm.base import LLMBackend, Parsed, Usage, image_block, text_block
from commentary.recorded_demo import (
    Frame,
    Observation,
    _resolve_observation_actors,
    _writer_pack,
    safe_pack,
)
from commentary.recorded_research import ResearchBrief

MAX_BATCH_FRAMES = 8
MAX_BATCH_OBSERVATIONS = 3


class StreamingObservation(Observation):
    """Streaming permits an empty actor object as an explicit unreadable identity."""

    @model_validator(mode="before")
    @classmethod
    def drop_empty_actor(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        actor = value.get("actor")
        if (
            isinstance(actor, dict)
            and not actor.get("shirt_number")
            and not actor.get("shirt_name")
        ):
            return {**value, "actor": None}
        return value


class StreamingObserverResult(BaseModel):
    """Observer output with streaming-only handling for unreadable actor objects."""

    observations: list[StreamingObservation] = Field(max_length=MAX_BATCH_OBSERVATIONS)

    def __init__(self, **data: object) -> None:
        raw_observations = data.get("observations", [])
        if isinstance(raw_observations, list):
            data["observations"] = [
                item.model_dump() if isinstance(item, Observation) else item
                for item in raw_observations
            ]
        super().__init__(**data)


class StreamingLine(BaseModel):
    """One exact, evidence-cited line, or no line for a silent batch."""

    voice: Literal["caller", "analyst"]
    text: str = Field(min_length=1, max_length=180)
    evidence_ids: list[str] = Field(default_factory=list, max_length=3)
    research_ids: list[str] = Field(default_factory=list, max_length=2)

    @field_validator("text")
    @classmethod
    def short_spoken_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        if len(value.split()) > 22:
            raise ValueError("text must contain at most 22 words")
        return value


class WriterResult(BaseModel):
    line: StreamingLine | None = None


class BatchState(TypedDict, total=False):
    frames: list[Frame]
    playback_s: float
    observations: list[Observation]
    raw_observer: StreamingObserverResult
    raw_writer: WriterResult
    desired_voice: Literal["caller", "analyst"]
    line: StreamingLine | None
    observer_usage: Usage
    writer_usage: Usage
    observation_latency_s: float
    writing_latency_s: float
    warnings: list[str]


OBSERVER_SYSTEM = """Observe only these currently available football frames. Return at most three
chronological, concrete observations. Each timestamp must be within the supplied batch bounds.
Use the supplied frame timestamps exactly; they are continuous across batches, never reset to zero.
Describe only visible play, goal, celebration, or replay; do not infer future events, names,
scores, tactics, or outcomes. Focus on the newest visible ball movement; do not report clocks
or scores. Do not decide attacking/defending direction unless the teams and goalkeeper are clear.
Attach an actor only for a readable shirt number/name or an action-linked graphic at that
observation; otherwise set actor to null. Do not set resolved_name. A celebration never
identifies an earlier action. Raised arms near a touchline may be a throw-in, not a celebration.
Call a goal only when a supplied frame clearly shows the ball inside the net or across the
goal line; never infer a goal from a pose, crowd, celebration, or scoreboard.
If uncertain, say so in the description."""

WRITER_SYSTEM = """Write one short spoken football-commentary line, or return line null when no
meaningful line is supported. Follow desired_voice exactly. Cite only supplied evidence IDs and
research IDs. Caller lines require visual evidence and never research. Analyst lines require
visual evidence or research. Use a player name only when that same cited observation supplies a
resolved_name. Never infer a name from a team list, research, another observation, kit colours,
or a later celebration: a named celebrant is not evidence of who scored or assisted.
Describe that person celebrating unless their identity is directly attached to the earlier action.
Never use stripes or generic runner/player/attacker labels. Never use kit colours or 'white shirt'
as a person's name: say Argentina/France, Argentinian/French, or leave the person out.
Those team names are examples, not defaults: without supplied team context, omit the actor
rather than guessing a team or nationality.
Prioritize the newest relevant observation, not a stale action from the start of the window.
Use restrained, natural commentary, not promotional excitement or a guessed match narrative.
Do not mention future play, score, or unseen
outcomes. Prefer a newly resolved name at its supported prominent action, avoid repeating recent
lines, and make analyst reactions or research relevant to the current play rather than standalone
facts. Preserve a research fact's stated scope exactly. Return exact spoken words, at most 22
words, without headings or explanation."""


def _json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"))


def _writer_observation(observation: Observation) -> dict[str, Any]:
    """Expose writer-safe evidence without its roster-resolution input."""
    item = observation.model_dump(mode="json", exclude={"actor"})
    if observation.actor is not None and observation.actor.resolved_name:
        item["actor"] = {
            "resolved_name": observation.actor.resolved_name,
            "action_role": observation.actor.action_role,
        }
    return item


class StreamingCommentator:
    """One observe → write graph invocation per caller-supplied frame batch."""

    def __init__(
        self,
        backend: LLMBackend,
        *,
        pack: dict[str, Any] | None = None,
        research: ResearchBrief | None = None,
        observer_model: str = "gpt-5.6-terra",
        writer_model: str = "gpt-5.6-terra",
    ) -> None:
        self.backend = backend
        self.pack = safe_pack(pack) if pack is not None else None
        self.research = research
        self.observer_model = observer_model
        self.writer_model = writer_model
        self._lines: list[StreamingLine] = []
        self._observations: list[Observation] = []
        self._batches = 0
        self.last_observer: dict[str, Any] | None = None
        self._graph = self._build_graph()

    def _desired_voice(self, observations: Sequence[Observation]) -> Literal["caller", "analyst"]:
        if any(item.kind == "goal" for item in observations):
            return "caller"
        caller_words = sum(len(line.text.split()) for line in self._lines if line.voice == "caller")
        analyst_words = sum(
            len(line.text.split()) for line in self._lines if line.voice == "analyst"
        )
        return "analyst" if caller_words > analyst_words * 2 else "caller"

    def _build_graph(self) -> Any:
        async def observe(state: BatchState) -> BatchState:
            frames = state["frames"]
            started = time.monotonic()
            parsed = await self.backend.parse(
                model=self.observer_model,
                system=OBSERVER_SYSTEM,
                blocks=self._observer_blocks(frames, state["playback_s"]),
                output_format=StreamingObserverResult,
                max_tokens=2048,
                effort="low" if self.observer_model == "gpt-6-astra" else "none",
                tag="streaming_observer",
            )
            self.last_observer = parsed.value.model_dump(mode="json")
            observations = self._validate_observations(parsed, frames, state["playback_s"])
            return {
                "raw_observer": parsed.value,
                "observations": observations,
                "desired_voice": self._desired_voice(observations),
                "observer_usage": parsed.usage,
                "observation_latency_s": time.monotonic() - started,
                "warnings": [],
            }

        async def write(state: BatchState) -> BatchState:
            observations = state["observations"]
            if not observations:
                return {
                    "raw_writer": WriterResult(line=None),
                    "line": None,
                    "writer_usage": Usage(),
                    "writing_latency_s": 0.0,
                    "warnings": [
                        *state.get("warnings", []),
                        "observer returned no observations; writer skipped",
                    ],
                }
            started = time.monotonic()
            parsed = await self.backend.parse(
                model=self.writer_model,
                system=WRITER_SYSTEM,
                blocks=[
                    text_block(_json(self._writer_payload(observations, state["desired_voice"])))
                ],
                output_format=WriterResult,
                max_tokens=1024,
                effort="low" if self.writer_model == "gpt-6-astra" else "none",
                tag="streaming_writer",
            )
            line = self._validate_line(parsed.value.line, observations, state["desired_voice"])
            warnings = [*state.get("warnings", [])]
            if line is None:
                warnings.append("writer chose silence")
            return {
                "raw_writer": parsed.value,
                "line": line,
                "writer_usage": parsed.usage,
                "writing_latency_s": time.monotonic() - started,
                "warnings": warnings,
            }

        graph = StateGraph(BatchState)
        graph.add_node("observe", observe)
        graph.add_node("write", write)
        graph.add_edge(START, "observe")
        graph.add_edge("observe", "write")
        graph.add_edge("write", END)
        return graph.compile(name="streaming_demo")

    def _observer_blocks(self, frames: Sequence[Frame], playback_s: float) -> list[dict[str, Any]]:
        payload = {
            "playback_s": playback_s,
            "batch_start_s": frames[0].at_s,
            "batch_end_s": frames[-1].at_s,
            "pack": self.pack,
        }
        blocks: list[dict[str, Any]] = [text_block(_json(payload))]
        for frame in frames:
            blocks.extend((text_block(f"frame at_s={frame.at_s:.2f}"), image_block(frame.jpeg)))
        return blocks

    def _validate_observations(
        self, parsed: Parsed[StreamingObserverResult], frames: Sequence[Frame], playback_s: float
    ) -> list[Observation]:
        observations = parsed.value.observations
        if len(observations) > MAX_BATCH_OBSERVATIONS:
            raise ValueError("observer returned more than three observations")
        ids = [item.id for item in observations]
        if len(ids) != len(set(ids)):
            raise ValueError("observer returned duplicate observation IDs")
        start_s, end_s = frames[0].at_s, frames[-1].at_s
        previous = start_s
        for item in observations:
            if not math.isfinite(item.at_s) or not start_s <= item.at_s <= end_s:
                raise ValueError("observer returned an observation outside the batch")
            if item.at_s < previous:
                raise ValueError("observer observations must be chronological")
            if item.at_s > playback_s:
                raise ValueError("observer returned future evidence")
            previous = item.at_s
        namespaced = [
            item.model_copy(update={"id": f"b{self._batches + 1}-o{index}"})
            for index, item in enumerate(observations, start=1)
        ]
        return _resolve_observation_actors(namespaced, self.pack)

    def _writer_payload(
        self, observations: Sequence[Observation], desired_voice: str
    ) -> dict[str, Any]:
        evidence = [*self._observations[-3:], *observations]
        return {
            "playback_s": observations[-1].at_s,
            "desired_voice": desired_voice,
            "recent_lines": [line.model_dump(mode="json") for line in self._lines[-4:]],
            "observations": [_writer_observation(item) for item in evidence],
            "teams": _writer_pack(self.pack),
            "research_facts": (
                [item.model_dump(mode="json") for item in self.research.facts]
                if self.research is not None
                else []
            ),
        }

    def _validate_line(
        self,
        line: StreamingLine | None,
        observations: Sequence[Observation],
        desired_voice: str,
    ) -> StreamingLine | None:
        if line is None:
            return None
        if line.voice != desired_voice:
            raise ValueError("writer did not use the requested voice")
        evidence = {item.id: item for item in [*self._observations[-3:], *observations]}
        research_ids = {item.id for item in self.research.facts} if self.research else set()
        if any(item not in evidence for item in line.evidence_ids):
            raise ValueError("writer cited unsupported or future evidence")
        if any(item not in research_ids for item in line.research_ids):
            raise ValueError("writer cited unsupported research")
        if line.voice == "caller" and (not line.evidence_ids or line.research_ids):
            raise ValueError("caller line must cite only visual evidence")
        if line.voice == "analyst" and not (line.evidence_ids or line.research_ids):
            raise ValueError("analyst line needs evidence or research")
        return line

    async def process(self, frames: list[Frame], *, playback_s: float) -> dict[str, Any]:
        """Process one bounded batch; this method never samples frames or starts background work."""
        self.last_observer = None
        self._validate_frames(frames, playback_s)
        state = await self._graph.ainvoke({"frames": frames, "playback_s": playback_s})
        observations = state["observations"]
        line = state["line"]
        self._observations.extend(observations)
        self._lines.extend([line] if line is not None else [])
        self._batches += 1
        observer_usage = state["observer_usage"]
        writer_usage = state["writer_usage"]
        return {
            "observations": [item.model_dump(mode="json") for item in observations],
            "line": line.model_dump(mode="json") if line is not None else None,
            "usage": {
                "cumulative": asdict(self.backend.total),
                "call_delta": {"observer": asdict(observer_usage), "writer": asdict(writer_usage)},
            },
            "observation_latency_s": state["observation_latency_s"],
            "writing_latency_s": state["writing_latency_s"],
            "warnings": state.get("warnings", []),
            "audit": {
                "playback_s": playback_s,
                "frame_timestamps": [frame.at_s for frame in frames],
                "desired_voice": state["desired_voice"],
                "observer_raw": state["raw_observer"].model_dump(mode="json"),
                "observer_actor_normalization": {
                    "caveat": (
                        "empty actor objects are normalized to null before identity validation"
                    )
                },
                "writer_raw": state["raw_writer"].model_dump(mode="json"),
            },
        }

    @staticmethod
    def _validate_frames(frames: Sequence[Frame], playback_s: float) -> None:
        if not frames or len(frames) > MAX_BATCH_FRAMES:
            raise ValueError("frames must contain 1 to 8 items")
        if not math.isfinite(playback_s) or playback_s < 0:
            raise ValueError("playback_s must be a finite value at or above zero")
        previous = -1.0
        for frame in frames:
            if not math.isfinite(frame.at_s) or not 0 <= frame.at_s <= playback_s:
                raise ValueError("frame timestamp must be finite and available at playback_s")
            if frame.at_s <= previous:
                raise ValueError("frame timestamps must be strictly ordered")
            previous = frame.at_s
