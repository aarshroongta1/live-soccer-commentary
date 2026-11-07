"""Shared, serializable contracts for replay, agents, and evaluation.

Reference transcripts and labels deliberately have no field in RuntimeSnapshot.
Snapshots must be deep copied by their producer before a role receives them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from commentary.recorded_demo import Observation
from commentary.recorded_research import ResearchBrief

Role = Literal["caller", "analyst"]


class SpeechLine(BaseModel):
    voice: Role
    text: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(default_factory=list, max_length=8)
    research_ids: list[str] = Field(default_factory=list, max_length=4)

    @field_validator("text")
    @classmethod
    def spoken_words(cls, value: str) -> str:
        if not value.strip() or len(value.split()) > 60:
            raise ValueError("speech must contain between one and 60 words")
        return value.strip()


class RuntimeSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: int = Field(ge=0)
    cutoff_s: float = Field(ge=0, allow_inf_nan=False)
    observations: list[Observation] = Field(default_factory=list)
    research: ResearchBrief | None = None
    recent_lines: list[SpeechLine] = Field(default_factory=list)
    teams: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def visible_evidence(self) -> RuntimeSnapshot:
        ids = [item.id for item in self.observations]
        if len(ids) != len(set(ids)):
            raise ValueError("snapshot evidence IDs must be unique")
        previous = -1.0
        for item in self.observations:
            if not 0 <= item.at_s <= self.cutoff_s or item.at_s < previous:
                raise ValueError("snapshot evidence must be chronological and visible")
            if item.action and any(t > self.cutoff_s for t in item.action.support_at_s):
                raise ValueError("action support must be visible at the snapshot cutoff")
            previous = item.at_s
        return self


class ToolCall(BaseModel):
    name: Literal["recent_events", "research", "player_context"]
    query: str = Field(default="", max_length=160)


class AgentAction(BaseModel):
    """Model-selected action, executed by a bounded application tool loop."""

    kind: Literal["tool", "speak", "abstain"]
    tool: ToolCall | None = None
    line: SpeechLine | None = None
    reason: str = Field(default="", max_length=300)

    @model_validator(mode="after")
    def one_action(self) -> AgentAction:
        if self.kind == "tool" and (self.tool is None or self.line is not None):
            raise ValueError("tool action requires only a tool")
        if self.kind == "speak" and (self.line is None or self.tool is not None):
            raise ValueError("speak action requires only a line")
        if self.kind == "abstain" and (self.tool is not None or self.line is not None):
            raise ValueError("abstain action cannot contain a tool or line")
        return self


class TraceEvent(BaseModel):
    event: str
    at_s: float = Field(ge=0, allow_inf_nan=False)
    role: Role | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class TurnResult(BaseModel):
    role: Role
    snapshot_version: int = Field(ge=0)
    cutoff_s: float = Field(ge=0, allow_inf_nan=False)
    line: SpeechLine | None = None
    reason: str = ""
    trace: list[TraceEvent] = Field(default_factory=list)


class ReviewResult(BaseModel):
    accepted: bool
    feedback: str = Field(default="", max_length=600)


class SpeechRecord(BaseModel):
    id: str
    line: SpeechLine
    snapshot_version: int = Field(ge=0)
    event_at_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    started_s: float = Field(ge=0, allow_inf_nan=False)
    ended_s: float = Field(ge=0, allow_inf_nan=False)
    interrupted: bool = False

    @model_validator(mode="after")
    def ordered(self) -> SpeechRecord:
        if self.ended_s < self.started_s:
            raise ValueError("speech cannot end before it starts")
        return self


class ReplayResult(BaseModel):
    mode: Literal["causal_replay"] = "causal_replay"
    observations: list[Observation] = Field(default_factory=list)
    speech: list[SpeechRecord] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
